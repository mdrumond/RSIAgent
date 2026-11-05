const phases = ['phase1', 'phase2', 'inference'];

function positive(value) {
  return Number.isSafeInteger(value) && value > 0;
}

// Read-only projection: never changes admission, recovery ownership or config.
export function campaignProgress(state, config) {
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
    Array.isArray(config.curriculum?.[phase]) ? [...new Set(config.curriculum[phase])] : []]));
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
  const generation = mode === 'phase1' && config.curriculum?.generation?.enabled === true;
  const phaseComplete = target !== null ? finishedOriginals >= target : !generation && lists[mode].length > 0 &&
    (rows.find(row => row.phase === mode)?.supplied_finished ?? 0) === lists[mode].length;
  const nextMode = phaseComplete ? sequence[sequence.indexOf(mode) + 1] ?? null : null;
  return {mode, finishedOriginals, totalOriginals, phaseCounts, target, phaseComplete, nextMode,
    done: phaseComplete && nextMode === null};
}
