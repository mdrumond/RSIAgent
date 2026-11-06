import {ActorClock} from './budget.mjs';

// Read-only projection: a retained running clock keeps charging on the same boot.
export function campaignElapsed(prior,{boot,now=()=>Number(process.hrtime.bigint())/1e6}) {
  if(!prior)return 0;
  if(!Number.isFinite(prior.elapsed)||prior.elapsed<0)throw new Error('Invalid retained campaign budget');
  let elapsed=prior.elapsed;
  if(prior.running&&prior.boot===boot) {
    const current=now();
    if(!Number.isFinite(prior.monotonic)||current<prior.monotonic)throw new Error('Campaign monotonic clock moved backwards');
    elapsed+=(current-prior.monotonic)/1000;
  }
  return elapsed;
}

// The controller supplies the same cycle identity after automatic restart or
// Resume. Only an explicit new campaign creates a new cycle. Project clocks are
// independent and are never reset here.
export function campaignBudget(state,{seconds,cycle,boot,now=()=>Number(process.hrtime.bigint())/1e6}) {
  if(!Number.isFinite(seconds)||seconds<=0||seconds>Number.MAX_SAFE_INTEGER/1000)
    throw new Error('Positive campaign elapsed allocation required');
  if(typeof cycle!=='string'||!cycle||typeof boot!=='string'||!boot)throw new Error('Campaign cycle and host boot identity required');
  state.db.exec('CREATE TABLE IF NOT EXISTS control(key TEXT PRIMARY KEY,value TEXT NOT NULL)');
  const row=state.db.prepare("SELECT value FROM control WHERE key='campaign-budget'").get();
  const prior=row?JSON.parse(row.value):null;
  const started=now();
  let elapsed=0;
  if(prior?.cycle===cycle) {
    if(!Number.isFinite(prior.elapsed)||prior.elapsed<0||!Number.isFinite(prior.monotonic))throw new Error('Invalid retained campaign budget');
    elapsed=campaignElapsed(prior,{boot,now:()=>started});
  }
  const clock=new ActorClock(seconds,elapsed,now);
  let stopped=false,snapshot;
  const persist=running=>{
    if(stopped)return snapshot;
    snapshot={cycle,boot,seconds,elapsed:clock.elapsed,monotonic:now(),running};
    state.db.prepare("INSERT OR REPLACE INTO control(key,value) VALUES ('campaign-budget',?)").run(JSON.stringify(snapshot));
    return snapshot;
  };
  persist(true);
  return {
    get elapsed(){return stopped?snapshot.elapsed:clock.elapsed;},
    get remaining(){return Math.max(0,seconds-this.elapsed);},
    checkpoint:()=>persist(true),
    stop:()=>{const value=persist(false);stopped=true;return value;},
  };
}
