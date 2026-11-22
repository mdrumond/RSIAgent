import {createAssistantMessageEventStream} from '@earendil-works/pi-ai';

// Return the same terminal event as a provider request timeout. Pi retains
// ownership of retry scheduling, request continuation and session history.
export function timeoutReviewStream(stream,shouldTimeout){
 return (model,context,options)=>{
  if(!shouldTimeout({model,context,options}))return stream(model,context,options);
  const timedOut=createAssistantMessageEventStream();
  timedOut.push({type:'error',reason:'error',error:{
   role:'assistant',api:model.api,provider:model.provider,model:model.id,content:[],
   usage:{input:0,output:0,cacheRead:0,cacheWrite:0,totalTokens:0,cost:{input:0,output:0,cacheRead:0,cacheWrite:0,total:0}},
   stopReason:'error',errorMessage:'Request timed out.',timestamp:Date.now()
  }});
  return timedOut;
 };
}
