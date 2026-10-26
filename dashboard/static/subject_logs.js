document.querySelectorAll('[data-live-logs]').forEach(section => {
  const output = section.querySelector('[data-log-output]');
  const status = section.querySelector('[data-log-status]');
  const follow = section.querySelector('[data-log-follow]');
  const earlier = section.querySelector('[data-log-earlier]');
  let cursor = null, history = 'start', following = true, busy = false;
  const entries = new Map();
  async function load(old = false) {
    if (busy) return;
    busy = true;
    const selected = old ? history : cursor;
    try {
      const query = selected === null ? '' : `?cursor=${encodeURIComponent(selected)}`;
      const response = await fetch(section.dataset.liveLogs + query);
      if (!response.ok) throw new Error(`Log unavailable (${response.status}); retrying`);
      const page = await response.json();
      if (page.reset) entries.clear();
      page.events.forEach(row => entries.set(row.id, row));
      if (old) history = page.cursor; else cursor = page.cursor;
      output.textContent = [...entries.values()].sort((a, b) => a.stream.localeCompare(b.stream) || a.offset - b.offset)
        .map(row => `${row.stream}\n${JSON.stringify(row.event, null, 2)}`).join('\n\n');
      status.textContent = entries.size ? `${entries.size} retained entries${following ? ' · following' : ' · paused'}` : 'Waiting for actor log entries';
      if (!old && following) output.scrollTop = output.scrollHeight;
      if (old && !page.events.length) earlier.disabled = true;
    } catch (error) { status.textContent = error.message; }
    finally { busy = false; }
  }
  follow.addEventListener('click', () => { following = !following; follow.textContent = following ? 'Pause' : 'Follow'; if (following) load(); });
  earlier.addEventListener('click', () => load(true));
  load();
  setInterval(() => { if (following) load(); }, 2000);
});
