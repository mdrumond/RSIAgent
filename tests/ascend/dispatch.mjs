import test from 'node:test';
import assert from 'node:assert/strict';
import {selectAdmission,memoryAdmission} from '../../controller/dispatch.mjs';
const fresh=id=>({id,kind:'original'}),retry=id=>({id,kind:'retry'});
test('single worker admits three fresh projects then one recovery without starving either queue',()=>{
  let freshSinceRecovery=0;const admissions=[];
  for(let i=0;i<8;i++){
    const choice=selectAdmission([retry('r'),fresh('f')],{freshSinceRecovery});
    admissions.push(choice.kind);freshSinceRecovery=choice.kind==='original'?freshSinceRecovery+1:0;
  }
  assert.deepEqual(admissions,['original','original','original','retry','original','original','original','retry']);
});
test('parallel workers reserve one recovery slot while retaining fresh capacity',()=>{
  const jobs=[retry('r'),fresh('f')],running=[];
  for(let i=0;i<4;i++)running.push(selectAdmission(jobs,{running,concurrency:4}));
  assert.equal(running.filter(job=>job.kind==='retry').length,1);
  assert.equal(running.filter(job=>job.kind==='original').length,3);
  assert.equal(selectAdmission(jobs,{running,concurrency:4}),null);
  assert.equal(selectAdmission(jobs,{running:[retry('active')],concurrency:4}).kind,'original');
});
test('empty queues do not waste admission capacity or bypass recovery share',()=>{
  assert.equal(selectAdmission([]),null);
  assert.equal(selectAdmission([retry('r')]).kind,'retry');
  assert.equal(selectAdmission([fresh('f')]).kind,'original');
  assert.equal(selectAdmission([retry('r')],{running:[retry('active')],concurrency:3}),null);
});
test('unknown worker peaks permit learning serially, measured peaks bound parallel growth',()=>{
  const cap={maxBytes:16*1024**3,currentBytes:2*1024**3,requested:4};
  assert.deepEqual(memoryAdmission(cap),{admit:true,learning:true});
  assert.equal(memoryAdmission({...cap,activeActors:1}).admit,false);
  assert.equal(memoryAdmission({...cap,activeActors:1,workerPeakBytes:2*1024**3}).admit,true);
  assert.equal(memoryAdmission({...cap,currentBytes:14*1024**3,workerPeakBytes:2*1024**3}).reason,'memory-capacity');
  assert.equal(memoryAdmission({...cap,currentBytes:16*1024**3}).admit,false);
  assert.equal(memoryAdmission({...cap,activeActors:4,workerPeakBytes:2*1024**3}).reason,'actor-capacity');
});

test('recovery retains its bounded share when actual capacity is below requested concurrency',()=>{
  const jobs=[fresh('next'),retry('r')];
  assert.equal(selectAdmission(jobs,{concurrency:8,freshSinceRecovery:3}).kind,'retry');
  assert.equal(selectAdmission(jobs,{concurrency:8,freshSinceRecovery:2}).kind,'original');
});
