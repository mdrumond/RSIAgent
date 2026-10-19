/* Only compact status is polled; event bodies are fetched when expanded. */
(() => {
  const form = document.querySelector('#optimizer-create');
  const job = document.querySelector('#optimizer-job');
  const message = document.querySelector('#optimizer-message');
  const key = () => crypto.randomUUID();
  async function read(url) {
    const response = await fetch(url);
    const data = await response.json();
    if (!response.ok) throw Error(data.error || `Request failed (${response.status})`);
    return data;
  }
  async function action(url, body, identity) {
    const multipart = body instanceof FormData;
    const response = await fetch(url, {method: 'POST', headers: {
      'X-CSRF-Token': document.body.dataset.csrf, 'Idempotency-Key': identity,
      ...(multipart ? {} : {'Content-Type': 'application/json'})},
      body: multipart ? body : JSON.stringify(body)});
    const data = await response.json();
    if (!response.ok) throw Error(data.error || `Request failed (${response.status})`);
    return data;
  }
  if (form) {
    read('/api/optimizer/sources').then(sources => {
      const select = form.elements.memory_root;
      select.replaceChildren();
      for (const source of sources) {
        const option = document.createElement('option');
        option.value = source.root;
        option.disabled = !source.available;
        option.textContent = source.available ? `${source.root} — ${source.memory_count} memories` : `${source.root} — unavailable: ${source.reason}`;
        select.append(option);
      }
      const eligible = sources.find(source => source.available);
      select.disabled = !eligible;
      document.querySelector('#optimizer-start').disabled = !eligible;
      if (eligible) select.value = eligible.root;
      const describe = () => {
        const source = sources.find(row => row.root === select.value);
        document.querySelector('#optimizer-memory-status').textContent = source?.available ? `Completed compatible A3 memory · snapshot ${source.snapshot_id}` : 'No completed compatible Phase 2 memory is available. Complete and configure a native A3 Phase 2 source first.';
      };
      select.addEventListener('change', describe); describe();
    }).catch(error => {message.textContent = error.message;});
    let identity = key();
    form.addEventListener('input', () => {identity = key();});
    form.addEventListener('submit', async event => {
      event.preventDefault();
      if (!confirm('Start optimization with this source, memory snapshot and budget?')) return;
      const button = document.querySelector('#optimizer-start'); button.disabled = true;
      try {
        const result = await action('/api/optimizer/jobs', new FormData(form), identity);
        location.href = `/optimizer/${result.job_id}`;
      } catch (error) {message.textContent = error.message; button.disabled = false;}
    });
  }
  if (!job) return;
  const base = `/api/optimizer/jobs/${job.dataset.jobId}`;
  async function refresh() {
    try {
      const state = await read(base);
      document.querySelector('#optimizer-progress').textContent = `${state.status} · ${state.stage}${state.stop_requested ? ' · stop requested' : ''}`;
      document.querySelector('#optimizer-budget').textContent = `${state.attempts} / ${state.max_attempts} attempts; ${Math.round(state.elapsed_seconds)} / ${state.max_seconds} seconds`;
      document.querySelector('#optimizer-handles').textContent = state.handles.map(row => `${row.operation}: ${row.handle}`).join('\n');
      document.querySelector('#optimizer-stop').disabled = !(state.worker_active || ['running', 'queued'].includes(state.status)) || state.stop_requested;
      document.querySelector('#optimizer-resume').hidden = state.worker_active || !['stopped', 'failed', 'budget-exhausted'].includes(state.status);
      document.querySelector('#optimizer-recover').hidden = state.worker_active || !['running', 'budget-exhausted'].includes(state.status) || state.stage !== 'observation-unavailable';
    } catch (error) {message.textContent = error.message;}
  }
  if (globalThis.RSI_DEMO) {
    // Snapshot templates omit all execution controls.
  } else {
  document.querySelector('#optimizer-stop').addEventListener('click', async () => {
    if (!confirm('Request this job to stop? Retained evidence will remain available.')) return;
    try {await action(`${base}/stop`, {}, key()); await refresh();}
    catch (error) {message.textContent = error.message;}
  });
  document.querySelector('#optimizer-recover').addEventListener('click', async () => {
    if (!confirm('Reconnect to the retained operation? This does not add budget or resubmit work.')) return;
    try {await action(`${base}/recover`, {}, key()); await refresh();}
    catch (error) {message.textContent = error.message;}
  });
  const resume = document.querySelector('#optimizer-resume');
  let resumeIdentity = key();
  resume.addEventListener('input', () => {resumeIdentity = key();});
  resume.addEventListener('submit', async event => {
    event.preventDefault();
    if (!confirm('Resume this job with additional budget?')) return;
    const payload = Object.fromEntries(new FormData(resume));
    Object.keys(payload).forEach(name => {payload[name] = Number(payload[name]);});
    try {await action(`${base}/resume`, payload, resumeIdentity); resumeIdentity = key(); await refresh();}
    catch (error) {message.textContent = error.message;}
  });
  }
  const trace = document.querySelector('#optimizer-trace');
  const submission = document.querySelector('#optimizer-submission');
  submission.parentElement.addEventListener('toggle', async () => {
    if (!submission.parentElement.open || submission.childNodes.length) return;
    try {
      const data = await read(`${base}/submission`);
      const pre = document.createElement('pre'); pre.textContent = JSON.stringify(data, null, 2);
      submission.append(pre);
    } catch (error) {message.textContent = error.message;}
  });
  async function refreshTrace() {
    if (!trace.open) return;
    try {
      const events = await read(`${base}/events`);
      const container = document.querySelector('#optimizer-events');
      for (const event of events) {
        if (container.querySelector(`[data-sequence="${event.sequence}"]`)) continue;
        const detail = document.createElement('details'); detail.dataset.sequence = event.sequence;
        const summary = document.createElement('summary');
        summary.textContent = `${event.sequence} · ${event.kind} · ${new Date(event.timestamp * 1000).toLocaleString()}`;
        detail.append(summary);
        detail.addEventListener('toggle', async () => {
          if (!detail.open || detail.querySelector('pre')) return;
          try {
            const data = await read(`${base}/events/${event.sequence}`);
            const pre = document.createElement('pre'); pre.textContent = JSON.stringify(data.payload, null, 2); detail.append(pre);
            if (data.payload?.sha256 && data.payload?.name) {
              const button = document.createElement('button'); button.textContent = `Load retained ${data.payload.name}`;
              button.addEventListener('click', async () => {
                try {
                  const response = await fetch(`${base}/artifacts/${data.payload.sha256}`);
                  if (!response.ok) throw Error(`Artifact unavailable (${response.status})`);
                  const body = await response.text();
                  try {pre.textContent = JSON.stringify(JSON.parse(body), null, 2);}
                  catch (_) {pre.textContent = body;}
                  button.remove();
                } catch (error) {message.textContent = error.message;}
              }); detail.append(button);
            }
          } catch (error) {message.textContent = error.message;}
        }); container.append(detail);
      }
    } catch (error) {message.textContent = error.message;}
  }
  trace.addEventListener('toggle', refreshTrace);
  if (!globalThis.RSI_DEMO) { refresh(); setInterval(() => {refresh(); refreshTrace();}, 3000); }
})();
