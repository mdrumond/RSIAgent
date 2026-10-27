// Render descriptions outside retained JSON/source text, using text nodes only.
window.renderStatusExplanations = function(container, payload) {
  const rows = [];
  function visit(value, path = '') {
    if (!value || typeof value !== 'object') return;
    for (const item of value.status_explanations || []) rows.push({item, path});
    for (const [key, child] of Object.entries(value)) {
      if (!['status_explanations', 'payload', 'source', 'submission', 'presentation'].includes(key)) visit(child, path ? `${path}.${key}` : key);
    }
  }
  visit(payload);
  const block = document.createElement('div'); block.className = 'status-explanations';
  for (const {item, path} of rows) {
    const p = document.createElement('p'); p.className = 'status-explanation';
    p.textContent = `${path ? path + ': ' : ''}${item.code} — ${item.description} Next: ${item.next_action}`;
    block.append(p);
  }
  container.append(block);
};

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
  control(action, subject, action === 'start' ? {preset_id: button.dataset.presetId} : action === 'import-feedback' ? {} : {campaign_id: button.dataset.campaignId});
}));
document.querySelectorAll('[data-feedback-action]').forEach(button => button.addEventListener('click', event => {
  event.preventDefault();
  const form = button.closest('form'), preset = form.elements.preset_id.value;
  control(button.dataset.feedbackAction, preset, {preset_id: preset,
    feedback_ids: [...form.querySelectorAll('input[name="feedback_id"]:checked')].map(item => item.value),
    project_target: Number(form.elements.project_target.value)});
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
    window.renderStatusExplanations(section, payload.presentation || payload);
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
function chartFrame(label, xLabel, yLabel, maximum) {
  const svg = svgNode('svg', {viewBox: '0 0 720 290', class: 'performance-chart', role: 'img', 'aria-label': `${label}; ${xLabel}; ${yLabel}`});
  svg.append(svgNode('line', {x1: 55, y1: 220, x2: 700, y2: 220, class: 'axis'}));
  svg.append(svgNode('line', {x1: 55, y1: 20, x2: 55, y2: 220, class: 'axis'}));
  svg.append(svgNode('text', {x: 377, y: 278, class: 'axis-label'}, xLabel));
  svg.append(svgNode('text', {transform: 'translate(13 120) rotate(-90)', class: 'axis-label'}, yLabel));
  [0, maximum / 2, maximum].forEach(value => {
    const y = 220 - value / maximum * 190;
    svg.append(svgNode('line', {x1: 51, y1: y, x2: 55, y2: y, class: 'axis'}));
    svg.append(svgNode('text', {x: 47, y: y + 3, class: 'axis-value'}, Number(value.toPrecision(3)).toString()));
  });
  return svg;
}
function point(svg, x, y, title, unavailable = false, extraClass = '') {
  const mark = svgNode('circle', {cx: x, cy: y, r: unavailable ? 5 : 4,
    class: `${unavailable ? 'plot-point unavailable' : 'plot-point'}${extraClass}`, tabindex: 0, 'aria-label': title});
  mark.append(svgNode('title', {}, title)); svg.append(mark);
}
function renderTiming(target, captures, shapeMode) {
  if (!captures.length) { target.textContent = 'No retained timing captures.'; return; }
  const maximum = Math.max(...captures.map(row => row.statistics.max_us), 1);
  const y = value => 220 - value / maximum * 190;
  const ordered = [...captures];
  if (shapeMode) ordered.sort((a, b) => a.shape_axis.kind === 'log-length' && b.shape_axis.kind === 'log-length' ? a.shape_axis.value - b.shape_axis.value : String(a.shape_axis.value).localeCompare(String(b.shape_axis.value)));
  const allLog = shapeMode && ordered.every(row => row.shape_axis.kind === 'log-length');
  const xLabel = !shapeMode ? 'Measurement attempt (#)' : allLog ? 'Logical length (elements, log scale)' : 'Logical shape (elements per dimension)';
  const svg = chartFrame(shapeMode ? 'Latency by logical shape' : 'Latency over attempts', xLabel, 'Latency (µs)', maximum);
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
  const svg = chartFrame('AIC and AIV pipe utilization', 'Pipe / profiling capture (category)', 'Utilization (%)', 100);
  const metrics = profiles.flatMap(row => row.metrics.map(metric => ({...metric, capture: row.capture}))).filter(metric => engine === 'all' || metric.engine === engine);
  if (!metrics.length) { target.textContent = 'No retained PipeUtilization capture.'; return; }
  metrics.forEach((metric, index) => {
    const x = 70 + (metrics.length === 1 ? 305 : index / (metrics.length - 1) * 610);
    const value = metric.available ? metric.value : 0, y = 220 - Math.min(Math.max(value, 0), 100) * 1.9;
    const duration = metric.time_us == null ? 'time NA' : `time ${metric.time_us.toFixed(3)} µs`;
    const title = metric.available ? `Capture ${metric.capture} · ${metric.engine} ${metric.label}: ${value.toFixed(3)}%; ${duration}` : `Capture ${metric.capture} · ${metric.engine} ${metric.label}: NA; ${duration}`;
    point(svg, x, y, title, !metric.available, ` capture-${metric.capture}`);
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


// Failure-case bodies remain lazy and exact downloads remain available in full.
document.querySelectorAll('[data-failure-cases-url]').forEach(section => section.addEventListener('toggle', async () => {
  if (!section.open || section.dataset.loaded) return;
  const status = section.querySelector('.failure-cases-status');
  try {
    const response = await fetch(section.dataset.failureCasesUrl);
    if (!response.ok) throw Error(`Failing examples could not be read (${response.status}). Inspect the retained feedback record.`);
    const cases = await response.json();
    const container = section.querySelector('.failure-cases-content');
    container.replaceChildren();
    for (const item of cases) {
      const article = document.createElement('article');
      const description = document.createElement('p'); description.textContent = item.description;
      const availability = document.createElement('p');
      availability.textContent = item.availability?.complete ? 'The retained example is available. Its scope and comparison policy are recorded below.' : `Incomplete evidence: ${item.availability?.reason || 'The failing example is unavailable.'}`;
      article.append(description, availability); window.renderStatusExplanations(article, item);
      const metadata = document.createElement('pre');
      const {artifacts, status_explanations, ...context} = item;
      metadata.textContent = JSON.stringify(context, null, 2); article.append(metadata);
      for (const [role, ref] of Object.entries(artifacts || {})) {
        const detail = document.createElement('details'), summary = document.createElement('summary');
        summary.textContent = `${role === 'observed' ? 'Actual output (observed)' : role} · SHA-256 ${ref.sha256}`;
        const url = section.dataset.artifactBase + ref.sha256;
        const download = document.createElement('a'); download.href = url + '/download'; download.textContent = 'Download exact complete artifact';
        const pre = document.createElement('pre');
        const more = document.createElement('button'); more.type = 'button'; more.textContent = 'Read next 8192 bytes'; more.hidden = true;
        const note = document.createElement('p'); note.textContent = 'Text preview is paged. Download preserves exact bytes, including binary tensor data.';
        detail.append(summary, download, note, pre, more);
        let offset = 0, loaded = false, loading = false;
        async function page() {
          if (loading) return; loading = true; more.disabled = true;
          try {
            const response = await fetch(`${url}?offset=${offset}`);
            if (!response.ok) throw Error(`Artifact unavailable or failed integrity verification (${response.status}). The example cannot be treated as complete.`);
            const body = await response.json(); pre.textContent += body.text;
            offset = body.next_offset ?? body.size; more.hidden = !body.has_more;
            note.textContent = `Showing ${Math.min(offset, body.size)} of ${body.size} bytes as text. Download preserves exact complete bytes.`;
            loaded = true;
          } catch (error) { note.textContent = error.message; }
          finally { loading = false; more.disabled = false; }
        }
        detail.addEventListener('toggle', () => {if (detail.open && !loaded) page();});
        more.addEventListener('click', page); article.append(detail);
      }
      container.append(article);
    }
    status.textContent = cases.length ? 'Retained evidence belongs to the original observed failure; it is separate from reusable memory.' : 'No failing example was retained for this observation.';
    section.dataset.loaded = 'true';
  } catch (error) {status.textContent = error.message;}
}));
