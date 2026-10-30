/* CLI-backed commands use the dashboard's existing durable queue. */
(() => {
  const root = document.getElementById('rsi-execution');
  if (!root) return;
  const $ = id => document.getElementById(id);
  const status = text => { $('execution-status').textContent = text; };
  async function api(url, options) {
    const response = await fetch(url, options);
    const value = await response.json();
    if (!response.ok) throw new Error(value.error || 'Instance command unavailable.');
    return value;
  }
  const pause = () => new Promise(resolve => setTimeout(resolve, 350));
  async function submit(payload) {
    const queued = await api('/api/rsi/execute', {method: 'POST',
      headers: {'Content-Type': 'application/json', 'X-CSRF-Token': document.body.dataset.csrf,
        'Idempotency-Key': crypto.randomUUID()}, body: JSON.stringify(payload)});
    status(`Queued command ${queued.operation_id}.`);
    for (;;) {
      const operation = await api(`/api/operations/${encodeURIComponent(queued.operation_id)}`);
      if (operation.status === 'succeeded') { status('CLI command completed. Refresh retained jobs for project status.'); return; }
      if (operation.status === 'failed') throw new Error('CLI command failed. Check the pinned release, inputs and retained evidence.');
      await pause();
    }
  }
  function bind(id, action, transform = value => value) {
    const form = $(id);
    form.addEventListener('submit', async event => {
      event.preventDefault();
      if (root.dataset.enabled !== 'true') return;
      const button = form.querySelector('button'); button.disabled = true;
      try {
        await submit({action, ...transform(Object.fromEntries(new FormData(form)))});
        await refresh();
      } catch (error) { status(error.message); }
      finally { button.disabled = false; }
    });
  }
  bind('phase-form', 'phase', value => {
    if (value.mode === 'inference' && value.operation === 'start') {
      value.project = JSON.parse(value.project);
      value.recover = Boolean(value.recover);
    } else { delete value.project; delete value.recover; }
    return value;
  });
  bind('repo-form', 'add-repo'); bind('export-form', 'export');
  async function inspect(id) {
    try {
      const value = await api(`/api/rsi/jobs/${encodeURIComponent(id)}`);
      // textContent prevents source paths/project metadata becoming markup.
      $('job-inspection').textContent = JSON.stringify(value, null, 2);
    } catch (error) { status(error.message); }
  }
  async function refresh() {
    const value = await api('/api/rsi/jobs');
    const selected = $('retained-job').value;
    $('retained-job').replaceChildren(new Option('Select retained job', ''));
    $('jobs-list').replaceChildren();
    for (const job of value.jobs) {
      const label = `${job.mode} / ${job.cycle_id} / ${job.state} / ${job.job_id}`;
      $('retained-job').append(new Option(label, job.job_id));
      const button = document.createElement('button'); button.type = 'button';
      button.textContent = label; button.addEventListener('click', () => inspect(job.job_id));
      $('jobs-list').append(button);
    }
    $('retained-job').value = selected;
  }
  $('refresh-jobs').addEventListener('click', () => refresh().catch(error => status(error.message)));
  refresh().catch(error => status(error.message));
})();
