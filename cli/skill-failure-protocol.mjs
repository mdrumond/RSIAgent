export const failureTiming=Object.freeze({
 baselineReplaySeconds:120,
 actorSeconds:180,
 verificationSeconds:300,
 isolationSeconds:1200,
 experimentSeconds:7200,
});

export function failureStageElapsed(stage){
 const error=new DOMException('Failure-first '+stage+' elapsed allocation exhausted','TimeoutError');
 return Object.assign(error,{failureStage:stage});
}
