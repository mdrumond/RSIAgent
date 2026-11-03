'use strict';
const el = id => document.getElementById(id);
const format = value => JSON.stringify(value, null, 2);
const parse = id => JSON.parse(el(id).value);
const fields = ['skillsmap', 'curriculum', 'environment'];
let dirty = false;
let loadedMode;

async function api(path, method = 'GET', data) {
  const response = await fetch(`/api/${path}`, {
    method, headers: {'Content-Type': 'application/json'},
    body: data === undefined ? undefined : JSON.stringify(data)
  });
  const value = await response.json();
  if (!response.ok) throw new Error(value.error || `Request failed (${response.status})`);
  return value;
}

async function perform(action) {
  const buttons = [...document.querySelectorAll('button')];
  buttons.forEach(button => { button.disabled = true; });
  el('message').textContent = 'Working…';
  try {
    await action();
    el('message').textContent = 'Done';
  } catch (error) {
    el('message').textContent = error.message;
  } finally {
    buttons.forEach(button => { button.disabled = false; });
  }
}

function loadFields(config) {
  const mode = config.mode ?? 'phase1';
  if (loadedMode !== mode) el('mode').value = mode;
  loadedMode = mode;
  el('config').value = format(config);
  el('baseline').value = config.baselineKDB ?? '';
  const scopes = Object.keys(config.skillsmap ?? {});
  const selected = el('skill-scope').value;
  el('skill-scope').replaceChildren(...scopes.map(scope => {
    const option = document.createElement('option');
    option.value = scope; option.textContent = scope;
    return option;
  }));
  el('skill-scope').value = scopes.includes(selected) ? selected : scopes[0] ?? '';
  fields.forEach(key => { el(key).value = format(config[key] ?? {}); });
  el('actors').value = format({actor: config.actor ?? {},
    curriculum: config.curriculum_actor ?? {}, writer: config.skill_writer_actor ?? {}});
  el('userinfo').value = typeof config.userinfo === 'string'
    ? config.userinfo : format(config.userinfo ?? '');
}

function table(title, rows) {
  const heading = document.createElement('h3');
  heading.textContent = title;
  const output = document.createElement('pre');
  output.textContent = format(rows ?? []);
  el('status').append(heading, output);
}

async function refresh() {
  const state = await api('status');
  el('status').replaceChildren();
  table('Projects', state.projects);
  table('Attempts', state.attempts);
  table('Background operations', state.requests);
}

el('settings').addEventListener('input', () => { dirty = true; });
el('apply').onclick = () => perform(async () => { loadFields(parse('config')); dirty = true; });
el('settings').onsubmit = event => {
  event.preventDefault();
  perform(async () => {
    const config = parse('config');
    config.baselineKDB = el('baseline').value;
    fields.forEach(key => { config[key] = parse(key); });
    const actors = parse('actors');
    config.actor = actors.actor;
    config.curriculum_actor = actors.curriculum;
    config.skill_writer_actor = actors.writer;
    delete config.actors;
    const saved = await api('config', 'PUT', config);
    loadFields(saved);
    dirty = false;
  });
};
el('refresh').onclick = () => perform(refresh);
el('skill-scope').onchange = () => { el('skill-content').value = ''; };
for (const action of ['load', 'save']) {
  el(`skill-${action}`).onclick = () => perform(async () => {
    const payload = {scope: el('skill-scope').value};
    if (!payload.scope) throw new Error('Choose a skill scope');
    if (action === 'save') {
      if (dirty) throw new Error('Save settings before saving a skill');
      payload.content = el('skill-content').value;
    }
    const result = await api('skill', 'POST', payload);
    if (action === 'save') loadFields(await api('config'));
    el('skill-scope').value = result.scope;
    el('skill-content').value = result.content;
  });
}
for (const action of ['wave', 'campaign', 'stop', 'resume']) {
  el(action).onclick = () => perform(async () => {
    if (dirty && ['wave', 'campaign'].includes(action)) throw new Error('Save settings before starting');
    const value = await api(action, 'POST', ['wave', 'campaign'].includes(action) ? {mode: el('mode').value} : {});
    el('action-result').textContent = format(value);
    await refresh();
  });
}
async function browse(path) {
  el('path').value = path;
  const result = await api('browse', 'POST', {path});
  el('browse-result').textContent = format(result);
  el('folders').replaceChildren();
  for (const entry of result.entries ?? []) {
    if (!(entry.isDirectory || entry.type === 'directory')) continue;
    const button = document.createElement('button');
    button.textContent = entry.name;
    button.onclick = () => perform(() => browse(entry.path));
    el('folders').append(button);
  }
}
el('browse').onsubmit = event => {
  event.preventDefault();
  perform(() => browse(el('path').value));
};
el('project-form').onsubmit = event => {
  event.preventDefault();
  perform(async () => {
    await api('project', 'POST', parse('project'));
    // Refresh the base config so later settings saves retain the added project.
    const config = await api('config');
    if (dirty) {
      const edited = parse('config');
      edited.projects = config.projects;
      el('config').value = format(edited);
    } else loadFields(config);
    await refresh();
  });
};
for (const action of ['tune', 'kdb']) {
  el(`${action}-form`).onsubmit = event => {
    event.preventDefault();
    perform(async () => {
      const payload = parse(`${action}-action`);
      if (action === 'tune') payload.userinfo = el('userinfo').value;
      const result = await api(action, 'POST', payload);
      if (action === 'tune') {
        const config = parse('config');
        config.userinfo = payload.userinfo;
        el('config').value = format(config);
      }
      el('action-result').textContent = format(result);
      await refresh();
    });
  };
}
perform(async () => { loadFields(await api('config')); await refresh(); });
