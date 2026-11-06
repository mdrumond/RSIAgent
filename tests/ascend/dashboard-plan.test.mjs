import {readFileSync} from 'node:fs';
import vm from 'node:vm';
import {test} from 'node:test';
import assert from 'node:assert/strict';

const source = readFileSync(new URL('../../web/app.js', import.meta.url), 'utf8');
const tick = () => new Promise(resolve => setImmediate(resolve));
async function dashboard(progress) {
  const elements = new Map();
  function element(tagName) {
    return {tagName, value: '', children: [], listeners: {},
      append(...nodes) { this.children.push(...nodes); },
      replaceChildren(...nodes) { this.children = nodes; },
      addEventListener(type, listener) { this.listeners[type] = listener; }};
  }
  const get = id => {
    if (!elements.has(id)) elements.set(id, element());
    return elements.get(id);
  };
  let config = {revision: 1, skills: {activeVersion: 'active', candidate: 'candidate'},
    projects: [], campaign_plan: [
      {id: 'first', phase: 'phase1', new_projects: 2},
      {id: 'next', phase: 'phase2', new_projects: 3},
      {id: 'last', phase: 'inference', new_projects: 4}]};
  let state = {progress, wait: [{project: 'old', kind: 'retry'}]};
  const requests = [];
  const fetch = async (url, options) => {
    const body = options.body && JSON.parse(options.body);
    requests.push({url, method: options.method, body});
    if (url === '/api/config' && body) config = {...body, revision: config.revision + 1};
    return {ok: true, json: async () => url === '/api/config' ? config : url === '/api/status' ? state : {}};
  };
  vm.runInNewContext(source, {document: {getElementById: get,
    createElement: element, querySelectorAll: () => []}, fetch});
  await tick();
  return {get, requests, setState(value) { state = value; },
    rows: () => get('campaign-entries').children,
    async submit(id) { get(id).onsubmit({preventDefault() {}}); await tick(); }};
}
const field = (row, index) => row.children[index].children[0];
const currentEntry = {id: 'first', phase: 'phase1', new_projects: 2, index: 0,
  finishedOriginals: 1, totalOriginals: 2, remainingOriginals: 1, remainingAdmissions: 0};

test('future plan actions preserve IDs, save config, and use campaign/resume routes', async () => {
  const ui = await dashboard({currentEntry, plan: {lockedPrefix: 1}});
  assert.equal(field(ui.rows()[0], 0).disabled, true);
  assert.equal(ui.rows()[1].children[3].disabled, true);
  field(ui.rows()[1], 0).value = 'editable'; field(ui.rows()[1], 0).onchange();
  field(ui.rows()[1], 1).value = 'inference'; field(ui.rows()[1], 1).onchange();
  field(ui.rows()[1], 2).value = '7'; field(ui.rows()[1], 2).onchange();
  ui.rows()[1].children[4].onclick();
  ui.rows()[1].children[5].onclick();
  ui.get('campaign-add').onclick();
  assert.equal(field(ui.rows()[2], 0).value, 'entry-1');
  await ui.get('resume').onclick();
  assert.equal(ui.requests.some(r => r.url === '/api/resume'), false);
  await ui.submit('settings');
  const saved = ui.requests.find(r => r.method === 'PUT').body;
  assert.deepEqual(saved.campaign_plan, [{id: 'first', phase: 'phase1', new_projects: 2},
    {id: 'editable', phase: 'inference', new_projects: 7}, {id: 'entry-1', phase: 'phase1', new_projects: 1}]);
  assert.deepEqual(saved.skills, {activeVersion: 'active', candidate: 'candidate'});
  await ui.get('campaign').onclick(); await ui.get('resume').onclick();
  assert.deepEqual(ui.requests.filter(r => ['/api/campaign', '/api/resume'].includes(r.url))
    .map(r => [r.url, r.body]), [['/api/campaign', {mode: 'phase1'}], ['/api/resume', {}]]);
  ui.get('project').value = JSON.stringify({id: 'p', campaign_entry: 'editable'});
  await ui.submit('project-form');
  assert.deepEqual(ui.requests.find(r => r.url === '/api/project').body, {id: 'p', campaign_entry: 'editable'});
});

test('refresh keeps drafts and updates progress locks and recovery display', async () => {
  const ui = await dashboard({currentEntry, plan: {lockedPrefix: 1}});
  field(ui.rows()[2], 2).value = '9'; field(ui.rows()[2], 2).onchange();
  const next = {...currentEntry, id: 'next', index: 1};
  ui.setState({skills: {activeVersion: 'active', candidate: 'draft', candidateBase: 'older'}, progress: {currentEntry: next, plan: {lockedPrefix: 2}}, recoveries: [{project: 'old'}]});
  await ui.get('refresh').onclick();
  assert.equal(JSON.parse(ui.get('skills-status').textContent).candidate, 'draft');
  assert.equal(JSON.parse(ui.get('skills-status').textContent).activeVersion, 'active');
  assert.equal(field(ui.rows()[1], 2).disabled, true);
  assert.equal(field(ui.rows()[2], 2).value, 9);
  assert.deepEqual(JSON.parse(ui.get('campaign-status').textContent), {
    currentEntry: next, progress: {currentEntry: next, plan: {lockedPrefix: 2}}, recoveriesIncludingCarriedWork: [{project: 'old'}]});
});

test('completed plan allows appended future entries at continuation', async () => {
  const ui = await dashboard({currentEntry: null, done: true, plan: {lockedPrefix: 3}});
  assert.equal(field(ui.rows()[2], 0).disabled, true);
  ui.get('campaign-add').onclick();
  await ui.get('refresh').onclick();
  assert.equal(field(ui.rows()[3], 0).disabled, false);
  field(ui.rows()[3], 0).value = 'continuation'; field(ui.rows()[3], 0).onchange();
  await ui.submit('settings');
  assert.equal(ui.requests.find(r => r.method === 'PUT').body.campaign_plan[3].id, 'continuation');
});


test('status before first admission leaves the first entry editable', async () => {
  const ui = await dashboard({currentEntry, plan: {lockedPrefix: 0}});
  assert.equal(field(ui.rows()[0], 0).disabled, false);
  ui.rows()[0].children[4].onclick();
  assert.equal(field(ui.rows()[1], 0).value, 'first');
});

test('skills metadata is read-only, bounded, and never adopts candidates', async t => {
  const {statusSkills} = await import('../../cli/main.mjs');
  const {mkdtemp, mkdir, writeFile, readFile, rm} = await import('node:fs/promises');
  const {tmpdir} = await import('node:os');
  const {join} = await import('node:path');
  const root = await mkdtemp(join(tmpdir(), 'dashboard-skills-'));
  t.after(() => rm(root, {recursive: true, force: true}));
  const config = {skills: {activeVersion: 'active'}};
  assert.deepEqual(await statusSkills(root, config), {activeVersion: 'active', candidate: null, candidateBase: null});
  await mkdir(join(root, 'skills'));
  const path = join(root, 'skills/evolution.json');
  const text = JSON.stringify({candidate: 'draft', candidateBase: 'older', history: [{project: 'private'}]});
  await writeFile(path, text);
  assert.deepEqual(await statusSkills(root, config), {activeVersion: 'active', candidate: 'draft', candidateBase: 'older'});
  assert.equal(await readFile(path, 'utf8'), text);
  assert.equal(config.skills.activeVersion, 'active');
  for (const value of ['invalid JSON', ' '.repeat(1024 * 1024 + 1)]) {
    await writeFile(path, value);
    const result = await statusSkills(root, config);
    assert.equal(result.candidate, null); assert.equal(result.activeVersion, 'active');
    assert.ok(result.metadataError);
  }
});


test('status projects recovery metadata without initializing or changing queues', async t => {
  const {statusRecoveries} = await import('../../cli/main.mjs');
  const {State} = await import('../../controller/state.mjs');
  const {RecoveryQueue} = await import('../../controller/recovery.mjs');
  const state = new State(':memory:'); t.after(() => state.close());
  assert.deepEqual(statusRecoveries(state), []);
  assert.equal(state.db.prepare("SELECT name FROM sqlite_master WHERE name='recovery_queue'").get(), undefined);
  new RecoveryQueue(state);
  state.add({id: 'old', goal: 'g', actor_seconds: 30});
  const attempt = state.admit('old', {});
  state.db.prepare("INSERT INTO recovery_queue(project,origin,kind,eligible_wave) VALUES (?,?,'retry',4)").run('old', attempt.id);
  const before = state.db.prepare('SELECT * FROM recovery_queue').all();
  assert.deepEqual(statusRecoveries(state).map(row => ({...row})), [{project: 'old', origin: attempt.id,
    kind: 'retry', eligible_wave: 4, attempt: null, failed_rounds: 0}]);
  assert.deepEqual(state.db.prepare('SELECT * FROM recovery_queue').all(), before);
});
