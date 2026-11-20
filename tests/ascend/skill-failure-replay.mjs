import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import {test} from 'node:test';
import {replayFailureCohort} from '../../cli/skill-failure-replay.mjs';

const hash = text => createHash('sha256').update(text).digest('hex');
const identity = {fixture:'fixture', oracle:'oracle', runtime:'a5', skills:'version', checker:hash('old')};
const submission = (id, arm, source, outcome) => ({
  id, arm, files:{'kernel.py':hash(source)}, verdict:{outcome},
});
const records = [
  submission('positive', 'positive', 'yes', 'pass'),
  submission('negative', 'negative', 'no', 'exact-failure'),
  submission('calibration', 'calibration', 'ready', 'pass'),
  submission('a', 'baseline', 'baseline', 'unscored'),
  submission('b', 'candidate', 'candidate', 'unscored'),
];
const sources = new Map(records.map((record, index) =>
  [record.id, ['yes', 'no', 'ready', 'baseline', 'candidate'][index]]));
const input = {
  original:{identity, records},
  correctedIdentity:{...identity, checker:hash('corrected')},
};
const readSource = async record => Buffer.from(sources.get(record.id));

test('same frozen bytes go through one checker, retaining both verdicts', async () => {
  const seen = [];
  const result = await replayFailureCohort(input, {
    readSource,
    score:async ({record, files, identity:scoringIdentity}) => {
      assert.equal(files['kernel.py'].toString(), sources.get(record.id));
      assert.deepEqual(scoringIdentity, input.correctedIdentity);
      seen.push(record.id);
      return {outcome:record.arm==='baseline'||record.arm==='negative'?'exact-failure':'pass'};
    },
  });
  assert.deepEqual(seen, records.map(record => record.id));
  assert.deepEqual(result.records.slice(-2).map(record => [record.original.outcome, record.revised.outcome]),
    [['unscored', 'exact-failure'], ['unscored', 'pass']]);
  assert.equal(result.freshConfirmationRequired, true);
});

test('changed source, altered oracle and missing arm fail before scoring', async () => {
  const score = async () => assert.fail('unexpected score');
  await assert.rejects(replayFailureCohort(input, {
    readSource: async record => record.id==='b'?Buffer.from('changed'):readSource(record), score,
  }), /hash mismatch/i);
  await assert.rejects(replayFailureCohort({
    ...input, correctedIdentity:{...input.correctedIdentity, oracle:'different'},
  }, {readSource, score}), /only the checker/i);
  await assert.rejects(replayFailureCohort({
    ...input, original:{identity, records:records.slice(0,-1)},
  }, {readSource, score}), /candidate/i);
});

test('a failed revised control stops scoring before actor artifacts', async () => {
  const seen=[];
  await assert.rejects(replayFailureCohort(input, {readSource, score:async ({record}) => {
    seen.push(record.id);
    return {outcome:'pass'};
  }}), /negative.*control/i);
  assert.deepEqual(seen, ['positive', 'negative']);
});
