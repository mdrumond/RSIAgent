import test from 'node:test';
import assert from 'node:assert/strict';
import {mkdtemp,mkdir,writeFile,readFile,rm} from 'node:fs/promises';
import {tmpdir} from 'node:os';
import {join} from 'node:path';
import {createServer} from 'node:http';
import {once} from 'node:events';
import {nativeReview} from '../../cli/skill-review-native.mjs';
import {State} from '../../controller/state.mjs';
import {processIdentity} from '../../controller/lifecycle.mjs';
import {scriptedReview} from './helpers/scripted-review.mjs';

test('native proposal uses selectors after dynamic KDB retrieval and restores its source catalogue',async t=>{
 const root=await mkdtemp(join(tmpdir(),'proposal-selectors-'));t.after(()=>rm(root,{recursive:true,force:true}));new State(join(root,'state.sqlite')).close();
 const directory=join(root,'proposal');await mkdir(directory);
 await writeFile(join(directory,'retrievals.jsonl'),JSON.stringify({entries:[{collection:'memory',chunk_id:'old',text:'old source\nsecond old line'}]})+'\n');
 const mcp=createServer(async(req,res)=>{
  if(req.url==='/health'){res.writeHead(200);res.end('ok');return;}
  if(req.method!=='POST'){res.writeHead(405);res.end();return;}
  let raw='';for await(const part of req)raw+=part;const message=JSON.parse(raw);
  if(message.id===undefined){res.writeHead(202);res.end();return;}
  let result={};
  if(message.method==='initialize')result={protocolVersion:message.params.protocolVersion,capabilities:{tools:{}},serverInfo:{name:'fixture',version:'1'}};
  if(message.method==='tools/list')result={tools:['search','read_entry'].map(name=>({name,description:name,inputSchema:{type:'object',properties:{query:{type:'string'},collection:{type:'string'},entry_id:{type:'string'}}}}))};
  if(message.method==='tools/call'){const entry={collection:'memory',chunk_id:'new',text:message.params.name==='search'?'preview':'memory source\nIndependent oracle'};result={content:[{type:'text',text:entry.text}],structuredContent:message.params.name==='search'?{result:[entry]}:entry};}
  res.writeHead(200,{'Content-Type':'application/json'});res.end(JSON.stringify({jsonrpc:'2.0',id:message.id,result}));
 });mcp.listen(0,'127.0.0.1');await once(mcp,'listening');t.after(()=>{mcp.closeAllConnections();mcp.close();});
 await writeFile(join(root,'knowledge-service.json'),JSON.stringify({url:`http://127.0.0.1:${mcp.address().port}/mcp`,owner:await processIdentity()}));
 const proposal={action:'update',reason:'Observed failure',groupIds:['g'],memoryRefs:['memory:new'],changes:[{action:'upsert',id:'oracle',scope:'implementation',text:'Use an independent oracle.',critical:false,negative:false,evidenceRefs:['event:1'],response:'actor-change'}],claims:[{path:'reason',kind:'observation',proofs:[{reference:'event:1',startLine:1,endLine:1}]},{path:'changes.0.text',kind:'hypothesis',proofs:[{reference:'memory:new',startLine:2,endLine:2}]}]};
 const stale=structuredClone(proposal);stale.claims[1].text='An earlier draft';
 const badQuote=structuredClone(proposal);badQuote.claims[1].proofs[0].quote='Invented copied quotation';
 const observed=[];const provider=await scriptedReview(t,root,async(body,n)=>{
  if(n===0)return {name:'read',args:{path:'input.json'}};
  if(n===1)return {name:'mcp__kdb__search',args:{query:'oracle'}};
  if(n===2){const c=JSON.parse(await readFile(join(directory,'proof-selectors.json'),'utf8'));assert.equal(c.find(r=>r.reference==='memory:new').segments[0].lines,1);observed.push(c);return {name:'mcp__kdb__read_entry',args:{collection:'memory',entry_id:'new'}};}
  if(n===3)return {name:'read',args:{path:'proof-selectors.json'}};
  if(n===4){const c=JSON.parse(body.messages.filter(m=>m.role==='tool').at(-1).content);assert.equal(c.find(r=>r.reference==='memory:new').segments[0].lines,2);observed.push(c);return {name:'submit_review',args:{result:stale}};}
  if(n===5)return {name:'submit_review',args:{result:badQuote}};
  if(n===6)return {name:'submit_review',args:{result:proposal}};
  return null;
 });
 const input={directory,config:{root,actor:provider.profile},groups:[{id:'g',occurrences:[{reference:'event:1',data:{message:'Observed failure'},artifacts:[]}]}],entries:[],seconds:20,validate:x=>x};
 const out=await nativeReview(input);assert.equal(out.value.changes[0].text,proposal.changes[0].text);assert.equal(out.value.claims[1].text,proposal.changes[0].text);assert.equal(out.value.claims[1].proofs[0].quote,'Independent oracle');assert.equal(observed.length,2);
 const rejections=(await readFile(join(directory,'submission-rejections.jsonl'),'utf8')).trim().split('\n').map(JSON.parse);assert.equal(rejections.length,2);assert.equal(stale.claims[1].text,'An earlier draft');assert.equal(badQuote.claims[1].proofs[0].quote,'Invented copied quotation');
 const count=provider.requests();await rm(join(directory,'proof-selectors.json'));assert.equal((await nativeReview(input)).recovered,true);assert.equal(provider.requests(),count);
 const restored=JSON.parse(await readFile(join(directory,'proof-selectors.json'),'utf8'));assert.equal(restored.find(r=>r.reference==='memory:new').segments[0].lines,2);assert.equal(restored.find(r=>r.reference==='memory:old').segments[0].lines,2);
});
