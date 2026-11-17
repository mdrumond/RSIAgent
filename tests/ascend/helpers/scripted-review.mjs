import {join} from 'node:path';
import {writeFile} from 'node:fs/promises';
/** Transparent fake provider: responses are serialized unchanged. */
export async function scriptedReview(t,root,plan){
 const {createServer}=await import('node:http'),{once}=await import('node:events');let requests=0;
 const server=createServer(async(req,res)=>{
  let raw='';for await(const chunk of req)raw+=chunk;const body=JSON.parse(raw),tool=await plan(body,requests++);
  res.writeHead(200,{'Content-Type':'text/event-stream'});
  const delta={role:'assistant',...(tool?{tool_calls:[{index:0,id:'review-'+requests,type:'function',function:{name:tool.name,arguments:JSON.stringify(tool.args)}}]}:{content:'Done'})};
  const frame=value=>res.write('data: '+JSON.stringify(value)+'\n\n');
  frame({id:'review-pages',object:'chat.completion.chunk',created:1,model:'scripted',choices:[{index:0,delta,finish_reason:null}]});
  frame({id:'review-pages',object:'chat.completion.chunk',created:1,model:'scripted',choices:[{index:0,delta:{},finish_reason:tool?'tool_calls':'stop'}],usage:{prompt_tokens:10,completion_tokens:5,total_tokens:15}});res.end('data: [DONE]\n\n');
 });server.listen(0,'127.0.0.1');await once(server,'listening');t.after(()=>{server.closeAllConnections();server.close();});
 const credentials_file=join(root,'extra.env');await writeFile(credentials_file,'REVIEW_EXTRA=fixture-only\n');
 return {requests:()=>requests,profile:{provider:'review-extra',model:'scripted',credential_env:'REVIEW_EXTRA',credentials_file,base_url:`http://127.0.0.1:${server.address().port}/v1`,metadata:{id:'scripted',name:'Fixture',api:'openai-completions',reasoning:false,input:['text'],contextWindow:32000,maxTokens:4096,cost:{input:0,output:0,cacheRead:0,cacheWrite:0}}}};
}
