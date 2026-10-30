/* Input forms retain the full configuration while updating only edited fields. */
(() => {
  const root = document.getElementById('rsi-inputs');
  if (!root) return;
  const $ = id => document.getElementById(id);
  const editable = root.dataset.enabled === 'true';
  let snapshot, payload, folderTarget, folderPath, busy = false, generation = 0;
  const pending = new Set();
  const status = message => { $('input-status').textContent = message; };
  const copy = value => JSON.parse(JSON.stringify(value));
  async function api(url, options) {
    const response = await fetch(url, options);
    const value = await response.json();
    if (!response.ok) throw new Error(value.error || 'Request could not be completed.');
    return value;
  }
  function element(tag, text) {
    const node = document.createElement(tag);
    if (text !== undefined) node.textContent = text;
    return node;
  }
  function button(text, action) {
    const node = element('button', text); node.type = 'button';
    node.addEventListener('click', action); return node;
  }
  function field(parent, title, object, key, options = {}) {
    const label = element('label', title);
    const input = element(options.multiline || options.json ? 'textarea' : 'input');
    input.setAttribute('aria-label', title);
    if (options.json) input.value = JSON.stringify(object[key] ?? options.fallback ?? {}, null, 2);
    else if (options.boolean) { input.type = 'checkbox'; input.checked = object[key] === true; }
    else { input.value = object[key] ?? ''; if (options.number) { input.type = 'number'; input.min = '1'; input.step = options.integer ? '1' : 'any'; } }
    label.append(input); parent.append(label);
    input.addEventListener('input', () => {
      try {
        let value = options.json ? JSON.parse(input.value) : options.boolean ? input.checked : options.number ? Number(input.value) : input.value;
        if (options.number && (!Number.isFinite(value) || value <= 0 || (options.integer && !Number.isInteger(value)))) throw new Error('positive number');
        if (options.json && options.array && (!Array.isArray(value) || value.some(v => typeof v !== 'string'))) throw new Error('array of arguments');
        if (options.json && !options.array && (value === null || Array.isArray(value) || typeof value !== 'object')) throw new Error('object');
        if (options.nullable && value === '') value = null;
        if (options.json && !options.array && object[key] && typeof object[key] === 'object') {
          for (const name of Object.keys(object[key])) delete object[key][name];
          Object.assign(object[key], value);
        } else object[key] = value;
        pending.delete(input); input.setCustomValidity(''); sync();
      } catch (_) { pending.add(input); input.setCustomValidity('Enter valid ' + (options.json ? 'JSON' : 'positive number') + '.'); }
    });
    if (options.json && !options.array) input.addEventListener('change', () => { if (!pending.has(input)) render(); });
    if (options.folder) label.append(button('Browse folders', () => {
      folderPath = null; $('select-folder').disabled = true;
      folderTarget = path => { object[key] = path; input.value = path; sync(); };
      $('folder-browser').showModal(); browse();
    }));
    return input;
  }
  const limits = [ ['design_iterations', 'Design iterations'], ['implementation_iterations', 'Implementation iterations'],
    ['execution_tries', 'Execution tries'], ['actor_seconds', 'Actor seconds'], ['wall_seconds', 'Wall seconds'] ];
  function budgets(parent, object) {
    const defaults = {design_iterations: 3, implementation_iterations: 5, execution_tries: 20, actor_seconds: 1800, wall_seconds: null};
    for (const [key, value] of Object.entries(defaults)) if (!(key in object)) object[key] = value;
    for (const [key, title] of limits) {
      const row = element('div'); row.className = 'budget-row'; parent.append(row);
      const input = field(row, title, object, key, {number: true, integer: !key.endsWith('seconds')});
      const label = element('label', 'No limit'); const check = element('input');
      check.type = 'checkbox'; check.checked = object[key] === null; input.disabled = check.checked;
      label.append(check); row.append(label);
      check.addEventListener('change', () => {
        input.disabled = check.checked;
        if (check.checked) { object[key] = null; pending.delete(input); input.setCustomValidity(''); }
        else { input.value = Number(input.value) > 0 ? input.value : '1'; input.dispatchEvent(new Event('input')); }
        sync();
      });
    }
  }
  function projects(name, target) {
    const parent = $(target); parent.replaceChildren();
    (payload[name] || []).forEach((project, index) => {
      const card = element('article'); card.className = 'project-editor'; parent.append(card);
      card.append(element('h3', 'Project ' + (index + 1)));
      for (const [key, title] of [['project_id', 'Project ID'], ['goal', 'Goal'], ['command', 'Command arguments (JSON array)'],
        ['starting_folder', 'Starting folder'], ['repository_ref', 'Repository reference'], ['correctness_ref', 'Correctness reference'], ['performance_goal', 'Performance goal']]) {
        field(card, title, project, key, {json: key === 'command', array: key === 'command', fallback: [], multiline: key === 'goal',
          folder: key === 'starting_folder', nullable: ['starting_folder', 'repository_ref', 'correctness_ref', 'performance_goal'].includes(key)});
      }
      project.budgets ||= copy(payload.budgets);
      const detail = element('details'); detail.append(element('summary', 'Project budgets')); card.append(detail); budgets(detail, project.budgets);
      card.append(button('Remove project', () => { payload[name].splice(index, 1); render(); }));
    });
  }
  function sync() { $('advanced-settings').value = JSON.stringify(payload, null, 2); }
  function render() {
    pending.clear();
    for (const id of ['prompt-fields', 'baseline-fields', 'actor-fields', 'budget-fields', 'policy-fields']) $(id).replaceChildren();
    payload.prompts ||= {}; payload.provider ||= {}; payload.roles ||= {}; payload.budgets ||= {};
    payload.recovery ||= {}; payload.tuning ||= {};
    for (const [key, title] of [['curriculum', 'Curriculum preparation'], ['actor', 'Actor instructions'], ['waves', 'Wave preparation'], ['skillwriter', 'Skill writer guidance']]) field($('prompt-fields'), title, payload.prompts, key, {multiline: true});
    for (const [key, title] of [['baseline_kdb', 'KDB baseline'], ['baseline_skills', 'Skill baseline'], ['skill_map_ref', 'Skill map reference']]) {
      field($('baseline-fields'), title, payload, Object.hasOwn(payload, key) ? key : {baseline_kdb: 'kdb', baseline_skills: 'skills', skill_map_ref: 'skill_map'}[key], {folder: true});
    }
    field($('actor-fields'), 'Provider profile', payload.provider, 'profile_id');
    field($('actor-fields'), 'Model', payload.provider, 'model');
    field($('actor-fields'), 'Roles interpreter reference', payload, 'roles_python', {nullable: true});
    field($('actor-fields'), 'Actor settings (JSON)', payload, 'actor', {json: true});
    field($('actor-fields'), 'Role settings (JSON)', payload, 'roles', {json: true});
    budgets($('budget-fields'), payload.budgets);
    projects('phase1_projects', 'phase1-projects'); projects('phase2_projects', 'phase2-projects');
    field($('policy-fields'), 'Enable recovery', payload.recovery, 'enabled', {boolean: true});
    for (const [key, title] of [['every_projects', 'Evaluate every N projects'], ['heldout_projects', 'Held-out projects'], ['runs_per_arm', 'Runs per arm']]) field($('policy-fields'), title, payload.tuning, key, {number: true, integer: true});
    field($('policy-fields'), 'Evaluate at wave end', payload.tuning, 'at_wave_end', {boolean: true});
    field($('policy-fields'), 'Advanced recovery (JSON)', payload, 'recovery', {json: true});
    field($('policy-fields'), 'Advanced tuning (JSON)', payload, 'tuning', {json: true});
    sync(); $('input-fields').disabled = !editable || busy;
  }
  async function load() {
    try { snapshot = await api('/api/rsi/config'); payload = copy(snapshot.payload); render(); status('Settings loaded.'); }
    catch (error) { status(error.message); }
  }
  async function browse(path) {
    const requestGeneration = ++generation;
    folderPath = null; $('select-folder').disabled = true;
    try {
      const listing = await api('/api/rsi/folders' + (path ? '?path=' + encodeURIComponent(path) : ''));
      if (requestGeneration !== generation) return;
      folderPath = listing.path; $('folder-status').textContent = listing.path;
      $('folder-roots').replaceChildren(...listing.roots.map(value => button(value, () => browse(value))));
      $('folder-list').replaceChildren(...listing.folders.map(value => button(value.name, () => browse(value.path))));
      $('select-folder').disabled = false;
    } catch (error) { if (requestGeneration !== generation) return; folderPath = null; $('folder-status').textContent = error.message; $('select-folder').disabled = true; }
  }
  $('select-folder').onclick = () => { if (folderPath) { folderTarget(folderPath); $('folder-browser').close(); } };
  $('close-folders').onclick = () => { generation++; $('folder-browser').close(); };
  $('reload-inputs').onclick = () => { if (!busy) load(); };
  root.querySelectorAll('[data-add]').forEach(node => node.onclick = () => {
    const key = node.dataset.add; payload[key] ||= [];
    payload[key].push({project_id: '', goal: '', command: ['python', 'main.py'], budgets: copy(payload.budgets)}); render();
  });
  $('apply-advanced').onclick = () => {
    try {
      const value = JSON.parse($('advanced-settings').value);
      if (!value || Array.isArray(value) || typeof value !== 'object') throw new Error('object');
      payload = value; render(); status('Advanced settings applied to form. Save to publish.');
    } catch (_) { status('Enter a configuration JSON object.'); }
  };
  $('inputs-form').onsubmit = async event => {
    event.preventDefault(); if (busy || !editable || pending.size) return;
    busy = true; $('input-fields').disabled = true; status('Saving settings…');
    try {
      const queued = await api('/api/rsi/config', {method: 'POST', headers: {'Content-Type': 'application/json',
        'X-CSRF-Token': document.body.dataset.csrf, 'Idempotency-Key': crypto.randomUUID()},
        body: JSON.stringify({revision: snapshot.revision, payload})});
      let operation;
      do { await new Promise(resolve => setTimeout(resolve, 250)); operation = await api('/api/operations/' + queued.operation_id); }
      while (['queued', 'running'].includes(operation.status));
      if (operation.status !== 'succeeded') throw new Error(operation.error || 'Save failed.');
      snapshot = operation.result; if (typeof snapshot === 'string') snapshot = JSON.parse(snapshot);
      payload = copy(snapshot.payload); render(); status('Settings saved for future projects.');
    } catch (error) { status(error.message); }
    finally { busy = false; $('input-fields').disabled = !editable; }
  };
  load();
})();
