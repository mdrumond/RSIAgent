/* No report trees or source bodies are fetched until requested. */
(() => {
  const job = document.querySelector('#optimizer-job');
  if (!job) return;
  const base = `/api/optimizer/jobs/${job.dataset.jobId}`;
  const summary = document.querySelector('#optimizer-result-summary');
  async function read(path) {
    const response = await fetch(base + path);
    if (!response.ok) throw Error(`Evidence unavailable (${response.status})`);
    return response.json();
  }
  const number = value => value == null ? 'N/A' : Number(value).toPrecision(5);
  function chart(target, title, rows, series, yLabel, xLabel) {
    const container = document.querySelector(target);
    container.replaceChildren();
    const heading = document.createElement('h3'); heading.textContent = title;
    container.append(heading);
    if (!rows.length) {
      const note = document.createElement('p'); note.textContent = 'No measurements available yet.';
      container.append(note); return;
    }
    const ns = 'http://www.w3.org/2000/svg';
    const svg = document.createElementNS(ns, 'svg');
    svg.setAttribute('viewBox', '0 0 840 320');
    svg.setAttribute('role', 'img'); svg.setAttribute('aria-label', `${title}. X axis: ${xLabel}. Y axis: ${yLabel}.`);
    svg.style.cssText = 'width:100%;min-width:500px;max-height:380px';
    const append = (type, attrs, label) => {
      const element = document.createElementNS(ns, type);
      Object.entries(attrs).forEach(([name, value]) => element.setAttribute(name, value));
      if (label != null) element.textContent = label;
      svg.append(element); return element;
    };
    const values = rows.flatMap(row => series.map(s => row[s.key])).filter(v => Number.isFinite(v));
    const low = Math.min(0, ...values), high = Math.max(0, ...values);
    const range = high - low || 1;
    const y = v => 245 - (v - low) / range * 190;
    const colors = ['#4f87ed', '#e89443'];
    append('line', {x1:90, y1:45, x2:90, y2:245, stroke:'currentColor'});
    append('line', {x1:90, y1:245, x2:800, y2:245, stroke:'currentColor'});
    append('text', {x:445,y:310,'text-anchor':'middle',fill:'currentColor'}, xLabel);
    append('text', {transform:'translate(18 150) rotate(-90)','text-anchor':'middle',fill:'currentColor'}, yLabel);
    for (let i = 0; i <= 4; i++) {
      const value = low + range * i / 4;
      append('text', {x:80,y:y(value)+4,'text-anchor':'end',fill:'currentColor','font-size':11}, number(value));
    }
    series.forEach((s, i) => {
      append('text', {x:100+i*250,y:25,fill:colors[i%2],'font-size':12}, s.label);
    });
    rows.forEach((row, index) => {
      const center = 90 + (index + .5) * 710 / rows.length;
      append('text', {x:center,y:265,'text-anchor':'middle',fill:'currentColor','font-size':10}, row.label);
      series.forEach((s, i) => {
        const value = row[s.key];
        if (!Number.isFinite(value)) return;
        const width = Math.min(36, 520 / rows.length / series.length);
        const rect = append('rect', {x:center+(i-series.length/2)*width,
          y:Math.min(y(value),y(0)),width:Math.max(1,width-2),height:Math.max(1,Math.abs(y(value)-y(0))),fill:colors[i%2]});
        const tooltip = document.createElementNS(ns, 'title');
        tooltip.textContent = `${row.label}: ${s.label} ${number(value)} ${yLabel}`; rect.append(tooltip);
      });
    });
    container.append(svg);
    const details = document.createElement('details'), caption = document.createElement('summary');
    caption.textContent = 'Exact values / unavailable measurements'; details.append(caption);
    const table = document.createElement('table');
    const header = table.insertRow();
    [xLabel,...series.map(s=>`${s.label} (${yLabel})`)].forEach(label => {
      const cell = document.createElement('th'); cell.textContent = label; header.append(cell);
    });
    rows.forEach(row=>{const tr=table.insertRow(); [row.label,...series.map(s=>number(row[s.key]))].forEach(value=>{
      const td=tr.insertCell(); td.textContent=value;
    });});
    details.append(table); container.append(details);
  }
  document.querySelector('#optimizer-load-results').addEventListener('click', async () => {
    try {
      const result = await read('/results');
      summary.textContent = result.best ? `Best verified geomean speedup: ${number(result.best.speedup)}× · ${result.best.summary || ''}` : 'No fully verified candidate with complete timings yet.';
      const cases = result.cases.map(row=>({...row,label:row.case}));
      chart('#optimizer-case-plot','CANNBot device latency',cases,
        [{key:'torch_us',label:'Torch reference'},{key:'ascendc_us',label:'Best Ascend C'}], 'Latency (µs)', 'Input case');
      chart('#optimizer-speedup-plot','Per-case speedup',cases,[{key:'speedup',label:'Torch / Ascend C'}], 'Speedup (×)', 'Input case');
      chart('#optimizer-attempt-plot','Performance over candidate attempts',result.attempts.map(row=>({...row,label:String(row.attempt)})),
        [{key:'speedup',label:'Candidate geomean'},{key:'best_speedup',label:'Best so far'}], 'Geomean speedup (×)', 'Candidate attempt (count)');
      const select = document.querySelector('#optimizer-metric'); select.replaceChildren();
      const groups = new Map();
      for (const metric of result.metrics) {
        const key = JSON.stringify([metric.group,metric.column,metric.unit]);
        if (!groups.has(key)) groups.set(key, []);
        groups.get(key).push(metric);
      }
      for (const [key, rows] of groups) {
        const option = document.createElement('option'); option.value = key;
        option.textContent = `${rows[0].column} · ${rows[0].group} · ${rows[0].unit || 'unit not exported'}`;
        select.append(option);
      }
      const profile = () => {
        const rows = groups.get(select.value) || [];
        chart('#optimizer-metric-plot','Diagnostic profiler metric',rows.map(row=>({
          label:`${row.case || 'case'} / ${row.engine} ${row.attempt ? 'attempt '+row.attempt : ''} / row ${row.row}`, value:row.available ? row.value : null})),
          [{key:'value',label:rows[0]?.column || 'Unavailable'}], rows[0]?.unit || 'Unit not exported', 'Case / engine / operator row');
      };
      select.onchange = profile; profile();
      document.querySelector('#optimizer-unavailable').textContent = result.unavailable.map(row=>
        `${row.case || ''} ${row.engine} / ${row.group}: N/A — ${row.reason}`).join('\n');
      const history = document.querySelector('#optimizer-source-list'); history.replaceChildren();
      for (const source of result.sources) {
        const detail = document.createElement('details'), title = document.createElement('summary');
        title.textContent = `Candidate ${source.index+1} · ${source.sha256.slice(0,12)}`; detail.append(title);
        detail.addEventListener('toggle', async () => {
          if (!detail.open || detail.dataset.loaded) return;
          try {
            const evidence = await read(`/sources/${source.index}`);
            const code = document.createElement('div'); code.innerHTML = evidence.html; // Escaped server-side Pygments output.
            const link = document.createElement('a'); link.href = `${base}/sources/${source.index}/download`; link.textContent = 'Download kernel';
            const diff = document.createElement('pre'); diff.textContent = evidence.diff;
            detail.append(link,code,diff); detail.dataset.loaded='true';
          } catch (error) {summary.textContent=error.message;}
        }); history.append(detail);
      }
    } catch (error) {summary.textContent = error.message;}
  });
})();
