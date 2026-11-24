import {readFile} from 'node:fs/promises';

const file=name=>readFile(new URL(name,import.meta.url),'utf8');
export async function rowSumPackets(){
 const example=JSON.parse(await file('example.json'));
 const [starter,good,bad,checker,code]=await Promise.all(['starter.py','good.py','bad.py','checker.py','probe.py'].map(file));
 const {operation,originalContract,diagnostic,checkId}=example;
 const records=Object.entries({'public-task':originalContract,'failing-source':starter,'observed-output':diagnostic,'inherited-guidance':''})
  .map(([reference,text])=>({reference,text}));
 return {
  input:{operation,originalContract,publicTask:originalContract,records},
  test:{decision:'test',reason:example.testReason,prompt:originalContract,starter,good,bad,checker,diagnosticRef:'observed-output:L1',checkId},
  probe:{decision:'probe',reason:example.probeReason,code},
  withhold:example.withhold,review:example.review,
 };
}

// Worked examples use their own synthetic source and measurements.
export async function rowSumHandoffs() {
  const {input, test, probe, withhold, review} = await rowSumPackets();
  const measurements = JSON.parse(await file('measurements.json'));
  const incomplete = {...input, records: input.records.map(r => r.reference === 'failing-source'
    ? {...r, text: '# The archived wrapper does not include the executed kernel.'} : r)};
  const previous = {...test, bad: test.good};
  return {
    designer: {
      input, accepted: test,
      probe: {input: {question: 'Which arguments does the installed tensor bridge accept?'}, accepted: probe,
        disposition: 'Run the probe and supply its observed signature before continuing.'},
      withhold: {input: incomplete, accepted: withhold,
        disposition: 'Record the missing executed source for archive recovery.'},
      disposition: 'Run all three controls with the independent checker.',
      rejected: {...test, diagnosticRef: 'row load failed'},
      correction: 'Select the observed-output line that identifies the historical failure.'
    },
    reviewer: {
      input: {fixture: test, measurements}, accepted: review,
      disposition: 'Admit the measured fixture for baseline trials.',
      rejected: {verdict: 'approve', reason: 'The Python files compile.', reference: 'unavailable-private-check'},
      correction: 'Select an artifact the reviewer actually received and explain how its measurements support the decision.'
    },
    feedback: {
      input: {
        previousFixture: previous,
        assessment: {bad: {expected: {exit: 'nonzero', checkId: test.checkId},
          observed: measurements.good}},
        feedback: 'The wrong control duplicates the working source. Its measured output passes all checks.'
      },
      accepted: test,
      disposition: 'Rerun calibration after changing the wrong control.',
      rejected: previous,
      correction: 'Change the wrong control so that it computes a different numerical result through the same native route.'
    }
  };
}
