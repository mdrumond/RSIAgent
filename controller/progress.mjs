const phases = ['phase1', 'phase2', 'inference'];

function positive(value) {
  return Number.isSafeInteger(value) && value > 0;
}

// Plain configuration policy shared by completion and generation callers.
export function generationSettings(config, phase=config.mode) {
  const base=config.curriculum?.generation;
  const settings={...base,...base?.by_phase?.[phase]};
  return {...settings,enabled:['phase1','phase2'].includes(phase)&&settings.enabled===true};
}

// Read-only projection: never changes admission, recovery ownership or config.
export function campaignProgress(state, config) {
  if (config.campaign_plan !== undefined) return plannedProgress(state, config);
  const mode = config.mode;
  const sequence = config.phase_sequence ?? [mode];
  if (!phases.includes(mode) || !Array.isArray(sequence) || !sequence.length ||
      sequence.some(phase => !phases.includes(phase)) || new Set(sequence).size !== sequence.length ||
      !sequence.includes(mode)) throw new Error('Invalid phase sequence');
  if (config.phase_targets !== undefined && (!config.phase_targets ||
      typeof config.phase_targets !== 'object' || Array.isArray(config.phase_targets) ||
      Object.entries(config.phase_targets).some(([phase, target]) => !phases.includes(phase) || !positive(target))))
    throw new Error('Invalid phase targets');
  if (config.completion_target !== undefined && !positive(config.completion_target))
    throw new Error('Invalid completion target');
  const target = config.phase_targets?.[mode] ?? config.completion_target ?? null;
  const lists = Object.fromEntries(phases.map(phase => [phase,
    Array.isArray(config.curriculum?.[phase]) ? [...new Set(config.curriculum[phase])] :
      [...new Set((config.projects??[]).filter(project=>project.phase===phase||(!project.phase&&phase===mode)).map(project=>project.id))]]));
  // Only project IDs from configuration enter SQL; histories stay in SQLite.
  const membership = phases.flatMap(phase => lists[phase].map(id => ({id, phase})));
  const hasRecovery = state.db.prepare("SELECT 1 FROM sqlite_master WHERE type='table' AND name='recovery_attempts'").get();
  const hasLessons = state.db.prepare("SELECT 1 FROM sqlite_master WHERE type='table' AND name='recovery_lessons'").get();
  const original = hasRecovery ? 'NOT EXISTS (SELECT 1 FROM recovery_attempts r WHERE r.attempt=a.id)' : '1';
  const acknowledged = hasLessons ? 'EXISTS (SELECT 1 FROM recovery_lessons l WHERE l.attempt=a.id AND l.recorded=1)' : '0';
  const rows = state.db.prepare(`
    WITH membership AS (
      SELECT json_extract(value,'$.id') AS id, json_extract(value,'$.phase') AS phase FROM json_each(?)
    ), supplied AS (
      SELECT id, MIN(phase) AS phase FROM membership GROUP BY id HAVING COUNT(*)=1
    ), originals AS (
      SELECT a.project, a.rowid AS admission_order,
        CASE WHEN json_valid(ad.data) THEN json_extract(ad.data,'$.mode') END AS admission_mode,
        CASE WHEN json_valid(p.spec) THEN json_extract(p.spec,'$.phase') END AS spec_phase,
        s.phase AS supplied_phase,
        CASE WHEN a.outcome IN ('passed','failed') AND ${acknowledged}
          AND EXISTS (SELECT 1 FROM events post WHERE post.attempt=a.id AND post.type='postprocess-completed'
            AND json_valid(post.data) AND json_type(post.data)='object')
          AND NOT EXISTS (SELECT 1 FROM events settlement WHERE settlement.sequence=(
            SELECT sequence FROM events WHERE attempt=a.id AND type='failure-settlement' ORDER BY sequence DESC LIMIT 1)
            AND CASE WHEN json_valid(settlement.data) THEN
              COALESCE(json_type(settlement.data,'$.closed')!='true',1) ELSE 1 END)
          THEN 1 ELSE 0 END AS finished
      FROM attempts a JOIN projects p ON p.id=a.project LEFT JOIN supplied s ON s.id=p.id
      LEFT JOIN events ad ON ad.sequence=(SELECT sequence FROM events
        WHERE attempt=a.id AND type='job-admitted' ORDER BY sequence LIMIT 1)
      WHERE ${original}
        AND (ad.sequence IS NULL OR CASE WHEN json_valid(ad.data) THEN
          json_extract(ad.data,'$.kind')='original' ELSE 0 END)
    ), pinned AS (
      SELECT project, finished, FIRST_VALUE(COALESCE(admission_mode,spec_phase,supplied_phase))
        OVER (PARTITION BY project ORDER BY admission_order) AS phase FROM originals
    ), scoped AS (
      SELECT project, phase, MAX(finished) AS finished FROM pinned GROUP BY project, phase
    )
    SELECT phase, COUNT(*) AS total, SUM(finished) AS finished,
      SUM(CASE WHEN finished=1 AND EXISTS (SELECT 1 FROM membership m
        WHERE m.id=scoped.project AND m.phase=scoped.phase) THEN 1 ELSE 0 END) AS supplied_finished
    FROM scoped WHERE phase IN ('phase1','phase2','inference') GROUP BY phase
  `).all(JSON.stringify(membership));
  const phaseCounts = Object.fromEntries(phases.map(phase => [phase, {finishedOriginals: 0, totalOriginals: 0}]));
  for (const row of rows) phaseCounts[row.phase] = {finishedOriginals: row.finished, totalOriginals: row.total};
  const {finishedOriginals, totalOriginals} = phaseCounts[mode];
  const generation = generationSettings(config,mode).enabled;
  const phaseComplete = target !== null ? finishedOriginals >= target : !generation && lists[mode].length > 0 &&
    (rows.find(row => row.phase === mode)?.supplied_finished ?? 0) === lists[mode].length;
  const nextMode = phaseComplete ? sequence[sequence.indexOf(mode) + 1] ?? null : null;
  return {mode, finishedOriginals, totalOriginals, phaseCounts, target, phaseComplete, nextMode,
    done: phaseComplete && nextMode === null};
}


export function validateCampaignPlan(config) {
  const plan = config.campaign_plan;
  if (plan === undefined) return;
  if (!Array.isArray(plan) || !plan.length || plan.some(entry => !entry ||
      typeof entry.id !== 'string' || !/^[a-zA-Z0-9_-]+$/.test(entry.id) ||
      !phases.includes(entry.phase) || !positive(entry.new_projects)) ||
      new Set(plan.map(entry => entry.id)).size !== plan.length)
    throw new Error('Invalid campaign plan');
  for (const project of config.projects ?? []) {
    const entry = plan.find(entry => entry.id === project.campaign_entry);
    if (!entry || (project.phase !== undefined && project.phase !== entry.phase))
      throw new Error('Project requires a matching campaign_entry: ' + project.id);
  }
}

// Persisted through the same control ledger as pause, progress and scheduler state.
// The cursor names a batch, never a phase: repeated phases are independent batches.
function atomic(state, operation) {
  if (state.db.isTransaction) return operation();
  state.db.exec('BEGIN IMMEDIATE');
  try {const result=operation();state.db.exec('COMMIT');return result;}
  catch(error){state.db.exec('ROLLBACK');throw error;}
}

function savedPlan(state) {
  if (!state.db.prepare("SELECT 1 FROM sqlite_master WHERE name='control'").get()) return null;
  return JSON.parse(state.db.prepare("SELECT value FROM control WHERE key='campaign-plan'").get()?.value ?? 'null');
}

export function syncCampaignPlan(state, config) {
  return atomic(state, () => reconcilePlan(state, config));
}

function reconcilePlan(state, config) {
  validateCampaignPlan(config);
  if (config.campaign_plan === undefined &&
      !state.db.prepare("SELECT 1 FROM sqlite_master WHERE name='control'").get()) return null;
  state.db.exec('CREATE TABLE IF NOT EXISTS control (key TEXT PRIMARY KEY,value TEXT NOT NULL)');
  const previous = savedPlan(state);
  if (config.campaign_plan === undefined) {
    if ((previous?.started ?? -1) >= 0) throw new Error('Cannot remove a started campaign plan');
    state.db.prepare("DELETE FROM control WHERE key='campaign-plan'").run();
    return null;
  }
  const cursor = previous?.index ?? 0;
  if (previous) for (let index = 0; index <= (previous.started ?? -1); index++) {
    if (!['id','phase','new_projects'].every(key => previous.plan[index][key] === config.campaign_plan[index]?.[key]))
      throw new Error('Cannot edit or remove a started campaign entry');
  }
  for (const project of config.projects ?? []) {
    const admitted = state.db.prepare('SELECT 1 FROM attempts WHERE project=? LIMIT 1').get(project.id);
    const admission = admitted && state.db.prepare(`SELECT json_extract(e.data,'$.campaign_entry') entry
      FROM events e JOIN attempts a ON a.id=e.attempt WHERE a.project=? AND e.type='job-admitted'
      ORDER BY e.sequence LIMIT 1`).get(project.id);
    if (admitted && (admission?.entry ?? state.project(project.id)?.spec.campaign_entry) !== project.campaign_entry)
      throw new Error('Cannot change campaign_entry of an admitted project');
  }
  const value = {plan: config.campaign_plan, index: cursor, started: previous?.started ?? -1};
  state.db.prepare("INSERT OR REPLACE INTO control(key,value) VALUES ('campaign-plan',?)").run(JSON.stringify(value));
  return value;
}

// PR3 generation and PR4 status contract. remainingOriginals excludes completed
// originals; remainingAdmissions also excludes active originals in this batch.
export function currentCampaignEntry(state, config) {
  validateCampaignPlan(config);
  if (config.campaign_plan === undefined) return null;
  const persisted = savedPlan(state) ?? {plan:config.campaign_plan,index:0};
  const entry = persisted.plan[persisted.index];
  if (!entry) return null;
  const hasQueue = state.db.prepare("SELECT 1 FROM sqlite_master WHERE name='recovery_attempts'").get();
  const original = hasQueue ? 'NOT EXISTS (SELECT 1 FROM recovery_attempts r WHERE r.attempt=a.id)' : '1';
  const acknowledged = hasQueue ? 'EXISTS (SELECT 1 FROM recovery_lessons l WHERE l.attempt=a.id AND l.recorded=1)' : '0';
  const finished = state.db.prepare(`SELECT COUNT(DISTINCT a.project) n FROM attempts a
    JOIN events ad ON ad.sequence=(SELECT MIN(sequence) FROM events WHERE attempt=a.id AND type='job-admitted')
    WHERE json_extract(ad.data,'$.campaign_entry')=? AND json_extract(ad.data,'$.kind')='original'
    AND ${original}
    AND a.outcome IN ('passed','failed')
    AND ${acknowledged}
    AND EXISTS (SELECT 1 FROM events e WHERE e.attempt=a.id AND e.type='postprocess-completed'
      AND json_valid(e.data) AND json_type(e.data)='object')
    AND NOT EXISTS (SELECT 1 FROM events e WHERE e.sequence=(SELECT MAX(sequence) FROM events
      WHERE attempt=a.id AND type='failure-settlement') AND
      CASE WHEN json_valid(e.data) THEN COALESCE(json_type(e.data,'$.closed')!='true',1) ELSE 1 END)`)
    .get(entry.id).n;
  const total = state.db.prepare(`SELECT COUNT(DISTINCT a.project) n FROM attempts a
    JOIN events ad ON ad.sequence=(SELECT MIN(sequence) FROM events WHERE attempt=a.id AND type='job-admitted')
    WHERE json_extract(ad.data,'$.campaign_entry')=? AND json_extract(ad.data,'$.kind')='original' AND ${original}`).get(entry.id).n;
  return {...entry, index: persisted.index, finishedOriginals: finished, totalOriginals: total,
    remainingOriginals: Math.max(0, entry.new_projects - finished),
    remainingAdmissions: Math.max(0, entry.new_projects - total)};
}

export function campaignPlanStatus(state, config) {
  if (config.campaign_plan === undefined) return null;
  validateCampaignPlan(config);
  const saved = savedPlan(state) ?? {plan:config.campaign_plan,index:0,started:-1};
  return {currentIndex:saved.index, lockedPrefix:(saved.started ?? -1)+1,
    entries:saved.plan.map((entry,index)=>({...entry,index,
      state:index<saved.index?'completed':index<=(saved.started ?? -1)?'started':'pending'}))};
}

function plannedProgress(state, config) {
  const entry = currentCampaignEntry(state, config);
  const phaseComplete = !entry || entry.remainingOriginals === 0;
  const next = entry && phaseComplete ? config.campaign_plan[entry.index + 1] : null;
  return {mode: entry?.phase ?? config.mode, currentEntry: entry, plan:campaignPlanStatus(state,config),
    finishedOriginals: entry?.finishedOriginals ?? 0, totalOriginals: entry?.totalOriginals ?? 0,
    target: entry?.new_projects ?? null, phaseComplete, nextMode: next?.phase ?? null,
    done: phaseComplete && !next};
}

export function startCampaignEntry(state, config) {
  return atomic(state, () => {
    const value = reconcilePlan(state, config);
    if (!value || !value.plan[value.index]) return false;
    if(value.started < value.index)state.event(null,'campaign-entry-started',{...value.plan[value.index],index:value.index});
    value.started = Math.max(value.started, value.index);
    state.db.prepare("UPDATE control SET value=? WHERE key='campaign-plan'").run(JSON.stringify(value));
    return true;
  });
}

export function advanceCampaignEntry(state, config) {
  return atomic(state, () => {
    const progress = plannedProgress(state, config);
    if (!progress.phaseComplete || !progress.currentEntry) return false;
    const value = reconcilePlan(state, config);
    value.started = Math.max(value.started, value.index);
    state.event(null,'campaign-entry-completed',{...value.plan[value.index],index:value.index});
    value.index++;
    state.db.prepare("UPDATE control SET value=? WHERE key='campaign-plan'").run(JSON.stringify(value));
    return true;
  });
}
