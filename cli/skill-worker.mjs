import * as background from './skill-background.mjs';

// This worker borrows one scheduler slot. Physical device admission stays in
// the same environment/lease machinery used by foreground actors.
export function skillWorker({control, config, remaining, event, signal, operations=background}) {
  let running, aborter, nextPoll=0;
  let resource=control.get('skill-background-resource');
  const release=async()=>{
    if(resource && await operations.backgroundSkillsSettled(config())) {
      resource=null;control.set('skill-background-resource',null);
    }
  };
  return {
    get resource(){return resource;},
    get running(){return running;},
    async poll(resources, active, limit) {
      const current=config();
      if(!current.skills?.background)return;
      const available=resources.filter(row=>row.available &&
        (!current.resource_ids || current.resource_ids.includes(row.id)));
      if(running) {
        if(available.length<2 || limit<2)aborter.abort(new Error('Foreground capacity takes priority'));
        return;
      }
      if(Date.now()<nextPoll)return;
      nextPoll=Date.now()+1000;
      await release();
      const status=await operations.backgroundSkillStatus(current);
      if(!status.hasWork || available.length<2 || limit<2 || active>=limit-1 || remaining()<=0)return;
      const reserved=resource??status.resource;
      const selected=available.find(row=>(!reserved || row.id===reserved.id) &&
        !control.leased(row.id));
      if(!selected)return;
      resource=selected;control.set('skill-background-resource',resource);
      aborter=new AbortController();
      running=Promise.resolve().then(()=>operations.runBackgroundSkills(current,{
        resource,remainingSeconds:remaining(),signal:signal?AbortSignal.any([aborter.signal,signal]):aborter.signal,
      })).catch(error=>event('skill-background-error',{message:error.message}))
        .finally(async()=>{
          try{await release();}finally{running=null;}
        });
    },
    async stop(reason='Campaign stopped') {
      aborter?.abort(new Error(reason));
      if(running)await running;
      await release();
    },
  };
}
