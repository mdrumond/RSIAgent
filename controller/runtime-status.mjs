// Compact dashboard/CLI projection. Raw Pi sessions and event payloads stay in
// their original stores; status never reads project logs or provider streams.
export function attemptMetadataBlocked(state,{limit=100}={}) {
  if(!Number.isSafeInteger(limit)||limit<1||limit>1000)throw new Error('Bounded status limit required');
  return state.db.prepare(`SELECT a.id AS attempt,a.project,json_extract(e.data,'$.reason') AS reason
    FROM attempts a JOIN events e ON e.sequence=(SELECT MAX(sequence) FROM events WHERE attempt=a.id
      AND type IN ('attempt-metadata-unavailable','attempt-metadata-restored'))
    WHERE e.type='attempt-metadata-unavailable' ORDER BY e.sequence DESC LIMIT ?`).all(limit);
}
export function foregroundResourceBlocked(state,{limit=100}={}) {
  if(!Number.isSafeInteger(limit)||limit<1||limit>1000)throw new Error('Bounded status limit required');
  return state.db.prepare(`SELECT a.id AS attempt,a.project,json_extract(e.data,'$.stage') AS stage,
      json_extract(e.data,'$.code') AS code,json_extract(e.data,'$.message') AS message
    FROM attempts a JOIN events e ON e.sequence=(SELECT MAX(sequence) FROM events WHERE attempt=a.id
      AND type IN ('foreground-resource-blocked','foreground-resource-migrated'))
    WHERE a.outcome IS NULL AND e.type='foreground-resource-blocked' ORDER BY e.sequence DESC LIMIT ?`).all(limit);
}
export function runtimeStatus(state,{limit=100}={}) {
  if(!Number.isSafeInteger(limit)||limit<1||limit>1000)throw new Error('Bounded status limit required');
  const exists=table=>Boolean(state.db.prepare("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?").get(table));
  const count=(table,where='1')=>exists(table)?state.db.prepare(`SELECT COUNT(*) n FROM ${table} WHERE ${where}`).get().n:0;
  const leases=state.db.prepare(`SELECT a.id AS attempt,a.project,
      json_extract(e.data,'$.pool') AS pool,json_extract(e.data,'$.physicalDevice') AS physicalDevice,
      json_extract(e.data,'$.receipt') AS receipt,
      CASE WHEN a.outcome IS NULL THEN 'running' ELSE 'unsettled' END AS state
    FROM attempts a JOIN events e ON e.sequence=(SELECT MAX(sequence) FROM events
      WHERE attempt=a.id AND type='device-lease-acquired')
    WHERE NOT EXISTS(SELECT 1 FROM events released WHERE released.attempt=a.id
      AND released.type='device-lease-released' AND released.sequence>e.sequence
      AND json_extract(released.data,'$.token') IS json_extract(e.data,'$.token'))
    ORDER BY e.sequence DESC LIMIT ?`).all(limit);
  const actorWaits=state.db.prepare(`SELECT a.id AS attempt,a.project,json_extract(e.data,'$.pool') AS pool,
      json_extract(e.data,'$.reason') AS reason FROM attempts a JOIN events e ON e.sequence=(
      SELECT MAX(sequence) FROM events WHERE attempt=a.id AND type='device-wait')
    WHERE a.outcome IS NULL AND json_extract(e.data,'$.waiting')=1 ORDER BY e.sequence DESC LIMIT ?`).all(limit);
  const postprocessing=state.db.prepare(`SELECT COUNT(*) n FROM attempts a WHERE outcome IS NOT NULL
    AND NOT EXISTS(SELECT 1 FROM events e WHERE e.attempt=a.id AND e.type='postprocess-completed')`).get().n;
  const metadataBlocked=attemptMetadataBlocked(state,{limit});
  return {leases,actorWaits,metadataBlocked,resourceBlocked:foregroundResourceBlocked(state,{limit}),queues:{fresh:count('projects',"status='queued' AND frozen=0"),
    recovery:count('recovery_queue','attempt IS NULL'),memory:count('recovery_lessons','recorded=0'),postprocessing},
    hostSlots:exists('resource_leases')?state.db.prepare('SELECT resource,attempt FROM resource_leases LIMIT ?').all(limit):[]};
}
