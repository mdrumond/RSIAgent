import test from 'node:test';
import assert from 'node:assert/strict';
import {mkdtemp,readFile,rm} from 'node:fs/promises';
import {join} from 'node:path';
import {tmpdir} from 'node:os';
import {skillObservations,writerEvidence,validateSkillGrounding} from '../../controller/skill-evidence.mjs';
import {generateCandidate} from '../../controller/skill-proposal.mjs';
import {candidateVersion} from '../../controller/skill-evolution.mjs';
import {Workflow} from '../../controller/workflow.mjs';
import {nativeWriter} from './skill-proposal-fixture.mjs';

const task={id:'failed-bsa',events:[
  {type:'knowledge-retrieved',data:{actor_elapsed:1169.782,durationSeconds:1.69}},
  {type:'failure-recorded',data:{message:'Device reservations require reconciliation: missing-receipt',actor_elapsed:1944}},
  {type:'rejected',data:{remaining:0,actor_elapsed:28800}},
]};
const change={action:'upsert',id:'observe-blocker',scope:'preflight',text:'Report the device blocker and retain work.',
  critical:false,negative:true,evidenceRefs:['failed-bsa:1'],response:'report-blocker'};

test('writer observations separate cumulative charged clocks, measured duration, and failure ownership',()=>{
  const observations=skillObservations(task);
  assert.equal(observations[0].actorClockSeconds,1169.782);
  assert.equal(observations[0].durationSeconds,1.69);
  assert.equal(observations[1].responsibility,'harness');
  assert.equal(observations[1].actorResponse,'report-blocker');
  assert.equal(observations[2].responsibility,'actor-workflow');
  assert.equal(observations[2].actorResponse,'reserve-validation-runway');
  assert.equal(skillObservations({events:[{type:'attempt-error',data:{message:'unclassified compiler failure',actor_elapsed:10}}]})[0].responsibility,'unknown');
  assert.equal(skillObservations({events:[{type:'knowledge-retrieved',data:{actor_elapsed:1169}}]})[0].durationSeconds,undefined);
  const frozen=structuredClone(task),writer=writerEvidence(task);
  assert.deepEqual(task,frozen);
  assert.equal(writer.events[0].data.actorClockSeconds,1169.782);
  assert.equal(writer.events[0].data.actor_elapsed,undefined);
});

test('grounded proposals cannot turn harness ownership into actor repair or invent operation durations',()=>{
  const observations=skillObservations(task);
  assert.throws(()=>validateSkillGrounding([{...change,response:'actor-change',scope:'implementation'}],observations),/Harness failures/);
  assert.throws(()=>validateSkillGrounding([{...change,scope:'implementation'}],observations),/before device work/);
  assert.throws(()=>validateSkillGrounding([{...change,evidenceRefs:['failed-bsa:2'],response:'reserve-validation-runway',scope:'acceptance'}],observations),/before the attempt is spent/);
  assert.throws(()=>validateSkillGrounding([{...change,measurements:[{reference:'failed-bsa:0',durationSeconds:1169.782}]}],observations),/measured operation duration/);
  assert.throws(()=>validateSkillGrounding([{...change,evidenceRefs:['not-observed']}],observations),/Unknown skill evidence/);
  assert.doesNotThrow(()=>validateSkillGrounding([{...change,measurements:[{reference:'failed-bsa:0',durationSeconds:1.69}]}],observations));
  assert.doesNotThrow(()=>validateSkillGrounding([{...change,evidenceRefs:['failed-bsa:2'],response:'reserve-validation-runway',scope:'global-kdb'}],observations));
});

test('native writer receives normalized evidence and can correct a rejected harness-repair submission',async t=>{
  const directory=await mkdtemp(join(tmpdir(),'grounded-writer-'));t.after(()=>rm(directory,{recursive:true,force:true}));
  const {native}=nativeWriter();let submit,disposed=false;const create=native.createAgentSession;
  const original=native.DefaultResourceLoader;
  native.DefaultResourceLoader=class extends original {
    async reload(){await super.reload();for(const factory of this.options.extensionFactories)
      await factory({registerTool(tool){if(tool.name==='submit_changes')submit=tool;},on(){}});}
  };
  native.createAgentSession=async options=>{
    const result=await create(options),dispose=result.session.dispose;
    result.session.prompt=async()=>{
      const usage=JSON.parse(await readFile(join(directory,'current-usage.json'),'utf8'));
      assert.equal(usage.context.events[0].data.actor_elapsed,undefined);
      assert.equal(usage.observations[0].durationSeconds,1.69);
      await assert.rejects(submit.execute('wrong',{changes:[{...change,response:'actor-change',scope:'implementation'}]}),/Harness failures/);
      await submit.execute('correct',{changes:[change]});
    };
    result.session.dispose=async()=>{disposed=true;await dispose();};return result;
  };
  let validated=0;
  const published=await generateCandidate({directory,input:{context:task,history:[task]},native,
    validate:()=>validated++});
  assert.equal(published[0].response,'report-blocker');
  assert.equal(published[0].scope,'preflight');
  assert.deepEqual(published[0].evidenceRefs,['failed-bsa:1']);
  assert.deepEqual(JSON.parse(await readFile(join(directory,'validated-changes.json'),'utf8')),published);
  assert.equal(validated,1);assert.ok(disposed);
});

test('retrieval records actual search duration independently of the actor allocation clock',async()=>{
  const events=[];
  const flow=new Workflow({event:async(type,data)=>events.push({type,data}),search:async()=>{
    await new Promise(resolve=>setTimeout(resolve,5));return {content:['result'],evidence:['reference']};
  }});
  assert.deepEqual(await flow.query('tail','negative',{}),['result']);
  assert.equal(events[0].type,'knowledge-retrieved');
  assert.ok(events[0].data.durationSeconds>=0.004);
  assert.ok(events[0].data.durationSeconds<1);
  assert.deepEqual(events[0].data.evidence,['reference']);
});

test('ordinary actor selection never adopts pending or rejected candidate versions',()=>{
  for(const evaluationStatus of ['deferred','running','completed'])
    for(const pending of [null,{candidate:'unapproved'}])
      assert.equal(candidateVersion({candidate:'unapproved',candidateBase:'accepted',evaluationStatus,pending},'accepted'),'accepted');
});
