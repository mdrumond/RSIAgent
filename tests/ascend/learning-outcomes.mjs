import test from 'node:test';
import assert from 'node:assert/strict';
import {State} from '../../controller/state.mjs';
import {RecoveryQueue} from '../../controller/recovery.mjs';
import {enqueueLearningRequest,claimLearningRequest,recordLearningOutcome} from '../../controller/skill-review.mjs';

function fixture(t){
 const state=new State(':memory:');t.after(()=>state.close());
 const queue=new RecoveryQueue(state,{mode:'phase1'}),draft='learning-draft';
 const id=enqueueLearningRequest(state,{target:'cpu',question:'Which boundary is correct?',requiredEvidence:'Pass the original independent verifier'});
 const request=claimLearningRequest(state,{skills:{review:{enabled:true}}},{draft,target:'cpu',topic:'boundary'});
 const project={id:'learning-project',goal:'Resolve the frozen boundary question',actor_seconds:60,generation:{draft,learningRequest:request}};
 state.add(project);
 const learning=()=>{const row=state.db.prepare('SELECT status,data FROM skill_learning_requests WHERE id=?').get(id);return {...row,data:JSON.parse(row.data)};};
 return {state,queue,project,learning};
}
const findings={findings:'Independent probe observed the behavior',evidence:[{path:'evidence.json',sha256:'a'.repeat(64)}]};

test('passing a smaller investigation preserves unresolved learning until the original retry succeeds',t=>{
 const {state,queue,project,learning}=fixture(t);
 const original=queue.claim(project.id,0).attempt;state.finish(original.id,'failed');
 recordLearningOutcome(state,project,state.attempt(original.id));const failed=learning();
 queue.enqueue(original.id,0);
 const investigation=queue.claim(project.id,1);state.finish(investigation.attempt.id,'passed');
 queue.complete(investigation.attempt.id,{...findings,passed:true,useful:true},1);
 recordLearningOutcome(state,project,state.attempt(investigation.attempt.id));
 assert.equal(state.project(project.id).status,'failed');assert.deepEqual(learning(),failed);
 assert.equal(queue.lessons()[0].useful,true);
 for(const lesson of queue.lessons())queue.acknowledgeLesson(lesson.id);
 assert.equal(queue.ready(2)[0].kind,'retry');
 const retry=queue.claim(project.id,2);queue.complete(retry.attempt.id,{...findings,passed:true},2);
 recordLearningOutcome(state,project,state.attempt(retry.attempt.id));
 assert.equal(state.project(project.id).status,'passed');assert.equal(learning().status,'verified');
 assert.equal(learning().data.findings.attemptId,retry.attempt.id);
});

test('a passed attempt without acceptance of the parent project cannot verify its learning request',t=>{
 const {state,queue,project,learning}=fixture(t),attempt=queue.claim(project.id,0).attempt;
 state.db.prepare("UPDATE attempts SET outcome='passed' WHERE id=?").run(attempt.id);
 recordLearningOutcome(state,project,state.attempt(attempt.id));
 assert.equal(state.project(project.id).status,'running');assert.equal(learning().status,'claimed');
 assert.equal(learning().data.findings,undefined);
});

test('legacy investigation admission metadata does not overwrite the original learning outcome',t=>{
 const {state,queue,project,learning}=fixture(t),attempt=queue.claim(project.id,0).attempt;
 state.finish(attempt.id,'failed');recordLearningOutcome(state,project,state.attempt(attempt.id));const failed=learning();
 state.db.prepare("UPDATE projects SET status='queued' WHERE id=?").run(project.id);
 const legacy=state.admit(project.id,{});state.event(legacy.id,'job-admitted',{kind:'investigation'});
 state.finish(legacy.id,'passed');recordLearningOutcome(state,project,state.attempt(legacy.id));
 assert.deepEqual(learning(),failed);
});
