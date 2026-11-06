'use strict';
const el = id => document.getElementById(id);
const format = value => JSON.stringify(value, null, 2);
const parse = id => JSON.parse(el(id).value);
const fields = ['skillsmap', 'curriculum', 'environment'];
let dirty = false;
let loadedMode;
let needsReload = false;
let campaignPlan = [];
let campaignProgress;
let planLoaded = false;

function lockedThrough() {
  if (!planLoaded) return Infinity;
  return (campaignProgress?.plan?.lockedPrefix ?? 0) - 1;
}

function renderPlan(rebuild = false) {
  if (!rebuild) {
    [...el('campaign-entries').children].forEach((row, index) => {
      const locked = index <= lockedThrough();
      for (let column = 0; column < 3; column++) row.children[column].children[0].disabled = locked;
      row.children[3].disabled = locked || index - 1 <= lockedThrough();
      row.children[4].disabled = locked || index + 1 >= campaignPlan.length;
      row.children[5].disabled = locked;
    });
    el('campaign-add').disabled = !planLoaded;
    return;
  }
  el('campaign-entries').replaceChildren();
  campaignPlan.forEach((entry, index) => {
    const row = document.createElement('div');
    const locked = index <= lockedThrough();
    const input = (key, type) => {
      const label = document.createElement('label');
      label.textContent = key;
      const field = document.createElement(type === 'select' ? 'select' : 'input');
      if (type === 'select') for (const phase of ['phase1', 'phase2', 'inference']) {
        const option = document.createElement('option');
        option.value = phase; option.textContent = phase; field.append(option);
      }
      else field.type = type;
      if (type === 'number') { field.min = '1'; field.step = '1'; }
      field.value = entry[key]; field.disabled = locked;
      field.oninput = field.onchange = () => {
        if (index <= lockedThrough()) return renderPlan();
        entry[key] = type === 'number' ? Number(field.value) : field.value;
        dirty = true;
      };
      label.append(field); row.append(label);
    };
    input('id', 'text'); input('phase', 'select'); input('new_projects', 'number');
    for (const [title, offset] of [['Move up', -1], ['Move down', 1], ['Remove', 0]]) {
      const button = document.createElement('button');
      button.type = 'button'; button.textContent = title;
      const destination = index + offset;
      button.disabled = locked || destination <= lockedThrough() || destination >= campaignPlan.length;
      button.onclick = () => {
        if (index <= lockedThrough() || destination <= lockedThrough()) return;
        if (!offset) campaignPlan.splice(index, 1);
        else [campaignPlan[index], campaignPlan[destination]] = [campaignPlan[destination], campaignPlan[index]];
        dirty = true; renderPlan(true);
      };
      row.append(button);
    }
    el('campaign-entries').append(row);
  });
  el('campaign-add').disabled = !planLoaded;
}

el('campaign-add').onclick = () => {
  if (!planLoaded) return;
  let suffix = 1;
  while (campaignPlan.some(entry => entry.id === `entry-${suffix}`)) suffix++;
  campaignPlan.push({id: `entry-${suffix}`, phase: 'phase1', new_projects: 1});
  dirty = true; renderPlan(true);
};

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
    renderPlan();
  }
}

function loadFields(config) {
  const mode = config.mode ?? 'phase1';
  if (loadedMode !== mode) el('mode').value = mode;
  loadedMode = mode;
  el('config').value = format(config);
  campaignPlan = JSON.parse(JSON.stringify(config.campaign_plan ?? []));
  renderPlan(true);
  el('skills-status').textContent = format(config.skills ?? {});
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

async function reloadSettings() {
  loadFields(await api('config'));
  dirty = false; needsReload = false;
  el('settings-conflict').hidden = true;
}

async function settingsChanged() {
  if (!dirty) return reloadSettings();
  needsReload = true;
  el('settings-conflict').hidden = false;
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
  campaignProgress = state.progress;
  planLoaded = true;
  renderPlan();
  if (state.skills) el('skills-status').textContent = format({ ...state.skills,
    candidateState: !state.skills.candidate ? null : state.skills.candidate === state.skills.activeVersion
      ? 'active' : state.skills.candidateDecision ?? 'unknown'});
  el('campaign-status').textContent = format({currentEntry: campaignProgress?.currentEntry ?? null,
    progress: campaignProgress, recoveriesIncludingCarriedWork: state.recoveries ?? []});
  el('status').replaceChildren();
  table('Controller / wave', state.controller);
  table('Phase progress', state.progress);
  table('Waiting for', state.wait);
  table('Host memory', state.memory);
  table('Projects', state.projects);
  table('Attempts', state.attempts);
  table('Background operations', state.requests);
  await refreshStorage();
}

el('settings').addEventListener('input', () => { dirty = true; });
el('apply').onclick = () => perform(async () => { loadFields(parse('config')); dirty = true; });
el('settings').onsubmit = event => {
  event.preventDefault();
  perform(async () => {
    if (needsReload) throw new Error('Reload settings before saving; your edits are still available');
    const config = parse('config');
    if (campaignPlan.length || config.campaign_plan !== undefined) config.campaign_plan = JSON.parse(JSON.stringify(campaignPlan));
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
el('reload-settings').onclick = () => perform(reloadSettings);
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
for (const action of ['wave', 'campaign', 'pause', 'stop', 'resume']) {
  el(action).onclick = () => perform(async () => {
    if (dirty && ['wave', 'campaign', 'resume'].includes(action)) throw new Error('Save settings before starting');
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
    await settingsChanged();
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
      if (action === 'tune') await settingsChanged();
      el('action-result').textContent = format(result);
      await refresh();
    });
  };
}


async function refreshStorage() {
  const state = await api('storage');
  el('storage-summary').textContent = format({campaign: state.campaign, usage: state.usage, revisions: state.revisions});
  el('storage-inventory').replaceChildren();
  for (const row of state.artifacts ?? []) {
    const link = document.createElement('button');
    link.textContent = `${row.id} · ${row.project_id ?? ''} · ${row.attempt ?? ''} · ${row.kind} · ${row.missing ? 'missing evidence' : row.status}`;
    link.onclick = () => {
      el('storage-action').value = format({action: 'inspect', artifact: row.id});
      el('storage-result').textContent = format(row);
    };
    el('storage-inventory').append(link);
  }
}
el('storage-form').onsubmit = event => {
  event.preventDefault();
  perform(async () => {
    el('storage-result').textContent = format(await api('storage', 'POST', parse('storage-action')));
    await refreshStorage();
  });
};

perform(async () => { await reloadSettings(); await refresh(); });
