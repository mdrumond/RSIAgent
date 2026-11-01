/* Read existing CLI projections; this page never starts or resumes work. */
(() => {
  const root = document.getElementById('instance-campaigns');
  if (!root) return;
  const list = document.getElementById('instance-campaigns-list');
  const status = document.getElementById('instance-campaigns-status');
  const refreshButton = document.getElementById('instance-campaigns-refresh');
  const add = (parent, tag, text) => {
    const node = document.createElement(tag);
    node.textContent = text;
    parent.append(node);
    return node;
  };
  const text = value => value == null ? 'unknown' : String(value);
  const seconds = value => typeof value === 'number' && Number.isFinite(value)
    ? value.toFixed(1) + ' s' : 'unknown';
  const budget = (row, key) => !Object.hasOwn(row, key) ? 'unknown'
    : row[key] === null ? 'unlimited' : seconds(row[key]);
  async function api(url) {
    const response = await fetch(url, {cache: 'no-store'});
    const value = await response.json();
    if (!response.ok) throw new Error(value.error || 'Instance evidence unavailable.');
    return value;
  }
  function projects(parent, inspection) {
    for (const wave of inspection.waves || []) {
      add(parent, 'h4', 'Wave ' + text(wave.wave_id) + ' · ' + text(wave.status));
      const table = document.createElement('table');
      parent.append(table);
      const header = document.createElement('tr');
      table.append(header);
      for (const name of ['Project', 'Lifecycle / outcome', 'Stage', 'Device',
        'Actor elapsed / allocation / remaining', 'Observation']) add(header, 'th', name);
      for (const project of wave.projects || []) {
        const row = document.createElement('tr');
        table.append(row);
        add(row, 'td', text(project.project_id));
        add(row, 'td', text(project.status) + ' / ' + (project.outcome || 'not settled'));
        add(row, 'td', project.stage || 'not started');
        add(row, 'td', project.device != null ? String(project.device)
          : project.assigned_device != null ? 'released ' + project.assigned_device : 'unassigned');
        const timing = add(row, 'td', seconds(project.actor_seconds) + ' / ' +
          budget(project, 'actor_allocation') + ' / ' + budget(project, 'remaining_actor_seconds'));
        if (Object.hasOwn(project, 'wave_actor_ceiling')) {
          add(timing, 'small', 'Wave ceiling: ' + budget(project, 'wave_actor_ceiling'));
        }
        const facts = [];
        if (project.pending) facts.push('Pending: ' + (project.pending.handle || project.pending.id || 'observation required'));
        if (project.admission && project.admission.admitted === false) {
          facts.push('Admission: ' + (project.admission.failure?.type || 'denied'));
        }
        if (project.recovery_authorized) facts.push('Harness recovery grant retained');
        add(row, 'td', facts.join(' · ') || '—');
      }
    }
    for (const task of inspection.tasks || []) {
      add(parent, 'p', 'Task ' + text(task.task_id) + ' · ' + text(task.accounting_state) +
        ' · actor elapsed ' + seconds(task.actor_union_seconds));
    }
    if (!inspection.available) add(parent, 'p', inspection.reason || 'Project evidence not yet available.');
    else if (!(inspection.waves || []).length && !(inspection.tasks || []).length) {
      add(parent, 'p', 'No retained project rows yet.');
    }
  }
  function card(job, evidence) {
    const article = document.createElement('article');
    article.className = 'card';
    add(article, 'h3', job.cycle_id || job.job_id);
    add(article, 'p', text(job.mode) + ' · job ' + text(job.job_id) + ' · ' + text(job.state) +
      (job.service_status ? ' / ' + job.service_status : ''));
    const link = add(article, 'a', 'Inspect in RSI execution');
    link.href = root.dataset.executionUrl;
    if (evidence.deferred) add(article, 'p', 'Open execution to inspect retained project evidence.');
    else if (evidence.error) add(article, 'p', evidence.error);
    else projects(article, evidence.inspection || {});
    return article;
  }
  let busy = false, timer;
  async function refresh() {
    if (busy) return;
    busy = true;
    refreshButton.disabled = true;
    clearTimeout(timer);
    try {
      const value = await api('/api/rsi/jobs');
      // A resume/reconcile supersedes its predecessor; UUID ordering is not time.
      const superseded = new Set(value.jobs.map(job => job.previous_job).filter(Boolean));
      const current = value.jobs.filter(job => !superseded.has(job.job_id));
      const cards = [];
      for (const job of current) {
        let evidence = {deferred: true};
        if (['running', 'launching', 'draining'].includes(job.state)) {
          try { evidence = await api('/api/rsi/jobs/' + encodeURIComponent(job.job_id)); }
          catch (error) { evidence = {error: error.message}; }
        }
        cards.push(card(evidence.job || job, evidence));
      }
      list.replaceChildren(...cards);
      status.textContent = cards.length ? 'Updated ' + new Date().toLocaleTimeString() + '.'
        : 'No CLI jobs are registered for this instance.';
    } catch (error) {
      // Keep the last successful view visible during CLI/transport failures.
      status.textContent = error.message + ' Last successful view retained; retrying automatically.';
    } finally {
      busy = false;
      refreshButton.disabled = false;
      timer = setTimeout(refresh, 10000);
    }
  }
  refreshButton.addEventListener('click', refresh);
  return refresh();
})();
