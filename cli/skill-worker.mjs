import * as background from './skill-background.mjs';

// The second iteration permanently earmarks one scheduler slot for tuning.
// Physical commands still use the foreground environment's shared device leases.
export function skillWorker({control, config, remaining, event, signal, operations=background, now=Date.now}) {
  let running, aborter, nextPoll=0, capacity=0;
  let resource=control.get('skill-background-resource');
  const enabled=()=>config().skills?.background && (!config().campaign_plan ||
    (control.get('campaign-plan')?.index??0)>=2);
  const settled=()=>operations.backgroundSkillsSettled(config());
  return {
    get resource(){return resource;},
    get reserved(){return Boolean(enabled() || resource);},
    get running(){return running;},
    get capacity(){return capacity;},
    async poll(resources, active, limit) {
      const current=config();
      const available=resources.filter(row=>row.available &&
        (!current.resource_ids || current.resource_ids.includes(row.id)));
      capacity=Math.min(limit,available.length);
      if(!enabled() || signal?.aborted)return;
      if(running)return;
      if(now()<nextPoll)return;
      nextPoll=now()+1000;
      const status=await operations.backgroundSkillStatus(current);
      // Retain the previous identity across unknown execution and sample resume.
      const retained=resource??status.resource;
      const selected=available.find(row=>(!retained || row.id===retained.id) && !control.leased(row.id));
      if(!selected || capacity<1 || active>capacity-1)return;
      resource=selected;control.set('skill-background-resource',resource);
      if(!status.hasWork || remaining()<=0 || !await settled())return;
      aborter=new AbortController();
      running=Promise.resolve().then(()=>operations.runBackgroundSkills(current,{
        resource,remainingSeconds:remaining(),signal:signal?AbortSignal.any([aborter.signal,signal]):aborter.signal,
      })).catch(error=>event('skill-background-error',{message:error.message}))
        .finally(()=>{running=null;}); // The earmark remains even when the pipeline is idle.
    },
    async stop(reason='Campaign stopped') {
      aborter?.abort(new Error(reason));
      if(running)await running;
      if(resource && await settled()) {
        resource=null;control.set('skill-background-resource',null);
      }
    },
  };
}
