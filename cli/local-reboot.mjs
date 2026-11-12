import {join} from 'node:path';
import {State} from '../controller/state.mjs';
import {localHostIdentity,attestLocalReboot} from '../controller/local-reboot.mjs';
import {withPackCapture} from './storage.mjs';

export async function reconcileLocalReboot(config,input) {
 if(config.binding_required||config.environment?.options?.mode!=='local')
  throw Error('A bound local campaign is required for reboot reconciliation');
 if(!input)return {currentHost:await localHostIdentity(),kind:'operator-attested-same-host-reboot'};
 if(input.operator?.target!==(config.environment.options.target??config.target))
  throw Error('Attested target differs from the local campaign');
 return withPackCapture(config,async()=>{
  const result=await attestLocalReboot(config.root,input);
  if(result.applied) {
   const state=new State(join(config.root,'state.sqlite'));
   try{state.event(null,'local-reboot-reconciled',{previousBoot:input.previousBoot,currentHost:result.currentHost,
    operator:input.operator,receipts:result.closures});}finally{state.close();}
  }
  return result;
 });
}
