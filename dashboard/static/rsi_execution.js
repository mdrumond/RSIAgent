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
  let baseline, selectedSkill;
  const editable = root.dataset.enabled === 'true';
  const pause = () => new Promise(resolve => setTimeout(resolve, 350));
  async function submit(payload) {
    const queued = await api('/api/rsi/execute', {method: 'POST',
      headers: {'Content-Type': 'application/json', 'X-CSRF-Token': document.body.dataset.csrf,
        'Idempotency-Key': crypto.randomUUID()}, body: JSON.stringify(payload)});
    status(`Queued command ${queued.operation_id}.`);
    for (;;) {
      const operation = await api(`/api/operations/${encodeURIComponent(queued.operation_id)}`);
      if (operation.status === 'succeeded') { status('CLI command completed. Refresh retained jobs for project status.'); return operation.result; }
      if (operation.status === 'failed') throw new Error('CLI command failed. Reload the baseline before retrying a revision conflict; check retained evidence for jobs.');
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
  bind('tune-form', 'manual-tuning');
  bind('repo-form', 'add-repo'); bind('export-form', 'export');
  function renderLedger(inspection) {
    const ledger = $('time-ledger'); ledger.replaceChildren();
    const add = (tag, text, parent = ledger) => {
      const node = document.createElement(tag); node.textContent = text; parent.append(node); return node;
    };
    if (!inspection.available) { add('p', inspection.reason); return; }
    for (const task of inspection.tasks) {
      add('h4', task.task_id || 'Retained task');
      add('p', `Accounting: ${task.accounting_state || 'not supplied'}`);
      const table = document.createElement('table'); ledger.append(table);
      const header = document.createElement('tr'); table.append(header);
      for (const text of ['Activity', 'Charged seconds', 'Excluded seconds']) add('th', text, header);
      const charged = task.charged_seconds || {}, excluded = task.excluded_seconds || {};
      for (const category of new Set([...Object.keys(charged), ...Object.keys(excluded)])) {
        const row = document.createElement('tr'); table.append(row);
        add('td', category, row);
        add('td', Object.hasOwn(charged, category) ? String(charged[category]) : '—', row);
        add('td', Object.hasOwn(excluded, category) ? String(excluded[category]) : '—', row);
      }
      add('p', `Actor interval union: ${task.actor_union_seconds == null ? 'unknown / unresolved' : task.actor_union_seconds + ' seconds'}`);
      for (const [key, label] of [['actor_seconds', 'Actor remaining'], ['wall_seconds', 'Wall remaining']]) {
        const remaining = task.remaining || {};
        add('p', `${label}: ${!Object.hasOwn(remaining, key) ? 'unknown' : remaining[key] === null ? 'unlimited' : remaining[key] + ' seconds'}`);
      }
      add('p', `Accepted skills: ${task.skills_ref || 'not supplied'}; candidate: ${task.candidate_ref || 'not supplied'}`);
    }
  }
  async function inspect(id) {
    try {
      const value = await api(`/api/rsi/jobs/${encodeURIComponent(id)}`);
      // textContent prevents source paths/project metadata becoming markup.
      $('job-inspection').textContent = JSON.stringify(value, null, 2);
      renderLedger(value.inspection);
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
  function acceptRevision(result) {
    if (!result || !result.bundle_ref) throw new Error('CLI omitted the new immutable bundle reference; reload before editing again.');
    baseline.revision = result.bundle_ref;
    $('skills-revision').textContent = `Baseline revision: ${result.bundle_ref}`;
  }
  function saveBuffer() {
    if (baseline && selectedSkill) baseline.skills[selectedSkill] = $('skill-content').value;
  }
  function selectSkill(id) {
    saveBuffer(); selectedSkill = id;
    $('skill-selector').value = id || '';
    $('skill-content').value = baseline.skills[id] || '';
  }
  function skillOptions(id) {
    $('skill-selector').replaceChildren();
    for (const name of Object.keys(baseline.skills).sort()) $('skill-selector').append(new Option(name, name));
    selectedSkill = undefined; selectSkill(id || $('skill-selector').value);
  }
  async function loadSkills() {
    const loaded = await api('/api/rsi/skills');
    baseline = loaded;
    $('skills-revision').textContent = `Baseline revision: ${loaded.revision}`;
    $('skill-map').value = JSON.stringify(loaded.step_map, null, 2);
    skillOptions();
    $('skills-fields').disabled = !editable;
  }
  $('skill-selector').addEventListener('change', () => selectSkill($('skill-selector').value));
  $('reload-skills').addEventListener('click', () => loadSkills().catch(error => status(error.message)));
  $('add-skill').addEventListener('click', () => {
    const name = $('new-skill-id').value.trim();
    if (!/^[a-z0-9]+(?:-[a-z0-9]+)*$/.test(name) || Object.hasOwn(baseline.skills, name)) {
      status('Enter a new skill ID using lowercase words separated by hyphens.'); return;
    }
    saveBuffer(); baseline.skills[name] = `---\nname: ${name}\ndescription: Describe when to use this skill.\n---\n\n`;
    skillOptions(name); $('new-skill-id').value = '';
  });
  $('save-skill').addEventListener('click', async () => {
    if (!editable || !baseline || !selectedSkill) return;
    $('skills-fields').disabled = true;
    try {
      saveBuffer();
      const result = await submit({action: 'skill-content', revision: baseline.revision, name: selectedSkill,
                    content: baseline.skills[selectedSkill]});
      acceptRevision(result);
    } catch (error) { status(error.message); }
    finally { $('skills-fields').disabled = !editable; }
  });
  $('skills-form').addEventListener('submit', async event => {
    event.preventDefault(); if (!editable || !baseline) return;
    $('skills-fields').disabled = true;
    try {
      saveBuffer();
      const stepMap = JSON.parse($('skill-map').value);
      if (!stepMap || Array.isArray(stepMap) || typeof stepMap !== 'object' ||
          Object.values(stepMap).some(ids => !Array.isArray(ids) || ids.some(id => !Object.hasOwn(baseline.skills, id)))) {
        throw new Error('Every workflow mapping must be an array of existing skill IDs.');
      }
      const result = await submit({action: 'skill-map', revision: baseline.revision, step_map: stepMap});
      acceptRevision(result); baseline.step_map = stepMap;
    } catch (error) { status(error.message); }
    finally { $('skills-fields').disabled = !editable; }
  });
  async function capabilities() {
    const value = await api('/api/rsi/capabilities');
    $('tuning-fields').disabled = !editable || !value.manual_tuning;
    $('tuning-capability').textContent = value.manual_tuning ? 'Canonical evaluator CLI available.' : 'Unavailable: this pinned CLI has not registered manual tuning. Draft inputs remain visible.';
    $('reload-skills').disabled = !value.skill_content_editing;
    $('skills-capability').textContent = value.skill_content_editing ? 'Immutable baseline revision CLI available.' : 'Unavailable: this pinned CLI has not registered immutable baseline editing.';
    if (value.skill_content_editing) await loadSkills();
  }
  capabilities().catch(error => status(error.message));
  $('refresh-jobs').addEventListener('click', () => refresh().catch(error => status(error.message)));
  refresh().catch(error => status(error.message));
})();
