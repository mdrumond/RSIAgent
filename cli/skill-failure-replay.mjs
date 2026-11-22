import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { isDeepStrictEqual } from 'node:util';

const hash = bytes => createHash('sha256').update(bytes).digest('hex');
const isHash = value => typeof value === 'string' && /^[a-f0-9]{64}$/.test(value);
const controlOutcomes = new Map([
  ['positive', 'pass'], ['negative', 'exact-failure'], ['calibration', 'pass'],
]);
const outcomes = new Set(['pass', 'exact-failure', 'other-failure', 'unscored', 'infrastructure']);

function requireVerdict(verdict) {
  assert(outcomes.has(verdict?.outcome), 'Replay verdict requires a recognized outcome');
}

/**
 * Replay a frozen cohort through a single corrected checker. The trusted
 * adapter reads retained sources and evaluates supplied bytes without actors.
 * The caller can persist the returned audit alongside the original cohort.
 */
export async function replayFailureCohort({ original, correctedIdentity }, { readSource, score }) {
  const { identity, records } = structuredClone(original);
  const revisedIdentity = structuredClone(correctedIdentity);
  const { checker: oldChecker, ...oldBinding } = identity;
  const { checker: newChecker, ...newBinding } = revisedIdentity;
  for (const key of ['fixture', 'oracle', 'runtime', 'skills']) {
    assert(oldBinding[key], 'Missing frozen replay identity: ' + key);
  }
  assert(isDeepStrictEqual(oldBinding, newBinding), 'Only the checker may change during replay');
  assert(isHash(oldChecker) && isHash(newChecker) && oldChecker !== newChecker,
    'Replay requires distinct checker hashes');
  for (const arm of [...controlOutcomes.keys(), 'baseline', 'candidate']) {
    assert(records.some(record => record.arm === arm), 'Missing replay arm: ' + arm);
  }

  const ids = new Set();
  const staged = new Map();
  for (const record of records) {
    assert(typeof record.id === 'string' && record.id.length && !ids.has(record.id), 'Invalid replay record ID');
    ids.add(record.id);
    requireVerdict(record.verdict);
    assert(controlOutcomes.has(record.arm) || ['baseline', 'candidate'].includes(record.arm), 'Unknown replay arm');
    const expected = controlOutcomes.get(record.arm);
    if (expected) assert.equal(record.verdict.outcome, expected, 'Original ' + record.arm + ' control failed');
    const entries = Object.entries(record.files ?? {});
    assert(entries.length && entries.every(([name, sha]) => name && isHash(sha)), 'Missing frozen source hashes');
    const sources = {};
    for (const [name, sha] of entries) {
      const bytes = await readSource(structuredClone(record), name);
      assert(bytes instanceof Uint8Array, 'Replay sources must be bytes');
      assert.equal(hash(bytes), sha, 'Source hash mismatch for ' + record.id + '/' + name);
      Object.defineProperty(sources, name, { value: Buffer.from(bytes), enumerable:true });
    }
    staged.set(record.id, sources);
  }

  const revised = new Map();
  const scoreRecord = async record => {
    const verdict = await score({
      record:{ id:record.id, arm:record.arm },
      files:staged.get(record.id),
      identity:structuredClone(revisedIdentity),
    });
    requireVerdict(verdict);
    for (const [name, sha] of Object.entries(record.files)) {
      assert.equal(hash(staged.get(record.id)[name]), sha, 'Checker changed source for ' + record.id + '/' + name);
    }
    revised.set(record.id, structuredClone(verdict));
  };
  for (const record of records.filter(record => controlOutcomes.has(record.arm))) {
    await scoreRecord(record);
    assert.equal(revised.get(record.id).outcome, controlOutcomes.get(record.arm),
      'Revised ' + record.arm + ' control failed');
  }
  for (const record of records.filter(record => !controlOutcomes.has(record.arm))) await scoreRecord(record);

  return {
    identity,
    revisedIdentity,
    freshConfirmationRequired:true,
    records:records.map(({ verdict, ...record }) => ({
      ...record, original:verdict, revised:revised.get(record.id),
    })),
  };
}
