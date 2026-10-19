// Keep live status useful without reloading long code pages.
if (document.querySelector('.status.running, .operation.running, .operation.queued')) setTimeout(() => location.reload(), 5000);

const csrf = document.body.dataset.csrf;
async function control(action, subject, body) {
  if (!confirm(`Confirm ${action} for ${subject}?`)) return;
  body.confirmation = `${action}:${subject}`;
  const response = await fetch(`/api/control/${action}`, {
    method: 'POST', headers: {'Content-Type': 'application/json', 'X-CSRF-Token': csrf,
      'Idempotency-Key': crypto.randomUUID()}, body: JSON.stringify(body)
  });
  if (!response.ok) alert((await response.json()).error || `Request failed: ${response.status}`);
  else location.reload();
}
document.querySelectorAll('button[data-action]').forEach(button => button.addEventListener('click', () => {
  const action = button.dataset.action, subject = button.dataset.subject;
  control(action, subject, action === 'start' ? {preset_id: button.dataset.presetId} : {campaign_id: button.dataset.campaignId});
}));
document.querySelectorAll('[data-proposal-action]').forEach(button => button.addEventListener('click', event => {
  event.preventDefault();
  const form = button.closest('form'), action = button.dataset.proposalAction;
  const body = {campaign_id: form.dataset.campaignId, proposal_id: form.dataset.proposalId};
  if (action === 'approve') body.project_ids = [...form.querySelectorAll('input:checked')].map(item => item.value);
  control(action, form.dataset.proposalId, body);
}));

document.querySelectorAll('.trace-event[data-trace-url]').forEach(section => section.addEventListener('toggle', async () => {
  if (!section.open || section.dataset.loaded) return;
  const output = section.querySelector('.trace-payload');
  try {
    const response = await fetch(section.dataset.traceUrl);
    if (!response.ok) throw new Error(`payload unavailable (${response.status})`);
    const payload = await response.json();
    try { output.textContent = JSON.stringify(JSON.parse(payload.text), null, 2); }
    catch { output.textContent = payload.text; }
    section.dataset.loaded = 'true';
  } catch (error) {
    output.textContent = error.message;
  }
}));

const svgNS = 'http://www.w3.org/2000/svg';
const svgNode = (name, attrs = {}, text = '') => {
  const node = document.createElementNS(svgNS, name);
  Object.entries(attrs).forEach(([key, value]) => node.setAttribute(key, value));
  node.textContent = text;
  return node;
};
function chartFrame(label) {
  const svg = svgNode('svg', {viewBox: '0 0 720 260', class: 'performance-chart', role: 'img', 'aria-label': label});
  svg.append(svgNode('line', {x1: 55, y1: 220, x2: 700, y2: 220, class: 'axis'}));
  svg.append(svgNode('line', {x1: 55, y1: 20, x2: 55, y2: 220, class: 'axis'}));
  return svg;
}
function point(svg, x, y, title, unavailable = false) {
  const mark = svgNode('circle', {cx: x, cy: y, r: unavailable ? 5 : 4,
    class: unavailable ? 'plot-point unavailable' : 'plot-point', tabindex: 0, 'aria-label': title});
  mark.append(svgNode('title', {}, title)); svg.append(mark);
}
function renderTiming(target, captures, shapeMode) {
  const svg = chartFrame(shapeMode ? 'Latency by logical shape' : 'Latency over attempts');
  if (!captures.length) { target.textContent = 'No retained timing captures.'; return; }
  const maximum = Math.max(...captures.map(row => row.statistics.max_us), 1);
  const y = value => 220 - value / maximum * 190;
  const ordered = [...captures];
  if (shapeMode) ordered.sort((a, b) => a.shape_axis.kind === 'log-length' && b.shape_axis.kind === 'log-length' ? a.shape_axis.value - b.shape_axis.value : String(a.shape_axis.value).localeCompare(String(b.shape_axis.value)));
  const allLog = shapeMode && ordered.every(row => row.shape_axis.kind === 'log-length');
  const logs = allLog ? ordered.map(row => Math.log10(row.shape_axis.value)) : [];
  const x = (row, index) => allLog && Math.max(...logs) !== Math.min(...logs) ? 70 + (Math.log10(row.shape_axis.value) - Math.min(...logs)) / (Math.max(...logs) - Math.min(...logs)) * 610 : 70 + (ordered.length === 1 ? 305 : index / (ordered.length - 1) * 610);
  ordered.forEach((row, index) => {
    const px = x(row, index), s = row.statistics;
    svg.append(svgNode('line', {x1: px, y1: y(s.min_us), x2: px, y2: y(s.max_us), class: 'whisker'}));
    svg.append(svgNode('rect', {x: px - 6, y: y(s.q3_us), width: 12, height: Math.max(1, y(s.q1_us) - y(s.q3_us)), class: 'quartile'}));
    const label = shapeMode ? String(row.shape_axis.value) : `#${row.attempt}`;
    point(svg, px, y(s.median_us), `${label}: median ${s.median_us.toFixed(3)} µs; range ${s.min_us.toFixed(3)}–${s.max_us.toFixed(3)} µs`);
    svg.append(svgNode('text', {x: px, y: 242, class: 'tick'}, label));
  });
  target.replaceChildren(svg);
}
function renderPipes(target, profiles, engine = 'all') {
  const svg = chartFrame('AIC and AIV pipe utilization');
  const metrics = profiles.flatMap(row => row.metrics).filter(metric => engine === 'all' || metric.engine === engine);
  if (!metrics.length) { target.textContent = 'No retained PipeUtilization capture.'; return; }
  metrics.forEach((metric, index) => {
    const x = 70 + (metrics.length === 1 ? 305 : index / (metrics.length - 1) * 610);
    const value = metric.available ? metric.value : 0, y = 220 - Math.min(Math.max(value, 0), 100) * 1.9;
    const duration = metric.time_us == null ? 'time NA' : `time ${metric.time_us.toFixed(3)} µs`;
    const title = metric.available ? `${metric.engine} ${metric.label}: ${value.toFixed(3)}%; ${duration}` : `${metric.engine} ${metric.label}: NA; ${duration}`;
    point(svg, x, y, title, !metric.available);
    svg.append(svgNode('text', {x, y: 242, class: `tick ${metric.engine.toLowerCase()}`}, `${metric.engine} ${metric.label}`));
  });
  target.replaceChildren(svg);
}
function pipeTable(target, profiles) {
  if (!profiles.length) return;
  const table = document.createElement('table');
  table.innerHTML = '<caption>Retained pipe values</caption><thead><tr><th>Capture</th><th>Engine</th><th>Metric</th><th>Ratio</th><th>Time µs</th></tr></thead>';
  const body = document.createElement('tbody');
  profiles.forEach(profile => profile.metrics.forEach(metric => {
    const values = [profile.capture, metric.engine, metric.label,
      metric.available ? `${metric.value.toFixed(3)}%` : 'NA',
      metric.time_us == null ? 'NA' : metric.time_us.toFixed(3)];
    const tr = document.createElement('tr');
    values.forEach(value => { const td = document.createElement('td'); td.textContent = value; tr.append(td); });
    body.append(tr);
  })); table.append(body); target.replaceChildren(table);
}
function performanceTable(target, captures) {
  if (!captures.length) return;
  const table = document.createElement('table');
  table.innerHTML = '<caption>Retained timing values</caption><thead><tr><th>Attempt</th><th>Shape</th><th>Samples</th><th>Median µs</th><th>Useful logical-I/O GiB/s</th></tr></thead>';
  const body = document.createElement('tbody');
  captures.forEach(row => {
    const cells = [row.attempt, row.logical_shape.join('×'), row.statistics.count, row.statistics.median_us.toFixed(3), row.useful_logical_io_gib_s == null ? 'NA' : row.useful_logical_io_gib_s.toFixed(3)];
    const tr = document.createElement('tr');
    cells.forEach(value => { const td = document.createElement('td'); td.textContent = value; tr.append(td); }); body.append(tr);
  }); table.append(body); target.replaceChildren(table);
}
document.querySelectorAll('.performance-view[data-performance-url]').forEach(section => section.addEventListener('toggle', async () => {
  if (!section.open || section.dataset.loaded) return;
  const status = section.querySelector('.performance-status');
  try {
    const response = await fetch(section.dataset.performanceUrl);
    if (!response.ok) throw new Error(`performance unavailable (${response.status})`);
    const data = await response.json();
    renderTiming(section.querySelector('[data-performance-chart="attempts"]'), data.captures, false);
    renderTiming(section.querySelector('[data-performance-chart="shapes"]'), data.captures, true);
    renderPipes(section.querySelector('[data-performance-chart="pipes"]'), data.profiles);
    pipeTable(section.querySelector('.pipe-table'), data.profiles);
    section.querySelectorAll('[data-pipe-engine]').forEach(button => button.addEventListener('click', () =>
      renderPipes(section.querySelector('[data-performance-chart="pipes"]'), data.profiles, button.dataset.pipeEngine)));
    performanceTable(section.querySelector('.performance-table'), data.captures);
    section.querySelector('.bandwidth-note').textContent = 'Useful logical-I/O bandwidth is derived from declared bytes and launch-plus-synchronize latency. Actual MTE bandwidth: NA (no retained Memory capture).';
    section.querySelector('.performance-plots').hidden = false; status.remove(); section.dataset.loaded = 'true';
  } catch (error) { status.textContent = error.message; }
}));
