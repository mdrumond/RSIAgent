import assert from 'node:assert/strict';
import {resolve} from 'node:path';
import {pathToFileURL} from 'node:url';
import {rowSumHandoffs, rowSumPackets} from './packets.mjs';

const [consumer] = process.argv.slice(2);
if (!consumer) throw Error('Usage: node validate.mjs PATH_TO_FIXTURE_REPLAY_MODULE');
const {publicPacket, validateDesign, checkControls} = await import(pathToFileURL(resolve(consumer)));
const {input, test, probe, withhold} = await rowSumPackets();
const packet = publicPacket(input);
const examples = await rowSumHandoffs();
validateDesign(test, packet);
validateDesign(probe, packet);
validateDesign(withhold, publicPacket({
  ...input,
  records: input.records.map(record => record.reference === 'failing-source'
    ? {...record, text: '# Only the launch wrapper survived.'} : record)
}));
assert.throws(() => validateDesign(examples.designer.rejected, packet));
assert.throws(() => validateDesign(examples.feedback.rejected, packet));
validateDesign(examples.feedback.accepted, packet, examples.feedback.input);
validateDesign(examples.designer.withhold.accepted, publicPacket(examples.designer.withhold.input));
const measured = examples.reviewer.input.measurements;
assert.deepEqual(checkControls(test, measured, packet), []);
assert.deepEqual(checkControls(test, {...measured, bad: measured.starter}, packet).map(x => x.role), ['bad']);
console.log('Validated public packet, test/probe/withhold, targeted feedback and three measured control outcomes.');
