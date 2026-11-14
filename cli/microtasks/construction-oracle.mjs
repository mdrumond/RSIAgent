/** CPU-only oracle over the real generation schema/static checks and native controls. */
import {readFile} from 'node:fs/promises';
import {join,resolve} from 'node:path';
import {parseArgs} from 'node:util';
import {Value} from 'typebox/value';
import {generationProposalSchema,compileGenerationProposal} from '../project-generation.mjs';
import {entrypointCommand} from '../../controller/project-commands.mjs';
import {createEnvironment} from '../../integrations/environments/index.mjs';
export async function constructionOracle({root,proposal,controls=false,seconds=30,signal}){
 const begun=performance.now(),checks=[];let stage='schema';root=resolve(root);
 try{
  if(controls!==false&&controls!==true&&!Array.isArray(controls))throw Error('controls must be false, true, or explicit control names');
  Value.Assert(generationProposalSchema,proposal);stage='static-admission';
  const {artifacts,verifier,baseline,construction}=await compileGenerationProposal(root,proposal);
  if(controls){
   const available=['reference','negative','starter',...(baseline?['baseline']:[])];
   if(Array.isArray(controls)&&(!controls.length||controls.some(kind=>!available.includes(kind))))throw Error('Requested CPU control must be available: '+JSON.stringify(available));
   stage='cpu-controls';const env=await createEnvironment({mode:'local'},{instancePath:join(root,'.oracle-executions'),signal,event:()=>{},exclude:(_label,fn)=>fn()});
   const command=entrypointCommand(construction.entrypoints.command);
   for(const c of [{kind:'reference',cwd:artifacts.reference,expected:0},...artifacts.negatives.map(cwd=>({kind:'negative',cwd,expected:1})),
    {kind:'starter',cwd:artifacts.source,expected:0},...(baseline?[{kind:'baseline',cwd:artifacts.source,expected:0,command:entrypointCommand(baseline.entrypoints.command)}]:[])]){
    if(Array.isArray(controls)&&!controls.includes(c.kind))continue;
    const remaining=seconds-(performance.now()-begun)/1000;if(remaining<=0)throw Error('Oracle elapsed allocation exhausted');
    const result=await env.execute(c.command??command,c.cwd,{signal,timeout:remaining,onData:()=>{}});checks.push({...c,...result});
    if(!result.settled||result.exitCode!==c.expected)throw Error('CPU construction control rejected');
   }
  }
  if(Array.isArray(controls)&&controls.some(kind=>!checks.some(check=>check.kind===kind)))throw Error('Requested CPU control did not execute');
  return {admitted:true,stage,entrypoints:verifier.entrypoints,checks,claim:'Production host admission only; not target/device qualification'};
 }catch(error){return {admitted:false,stage,error:error.message,checks,claim:'Observed production admission rejection; no repair inferred'};}
}
export async function oracleMain(argv=process.argv.slice(2)){
 const {values}=parseArgs({args:argv,options:{root:{type:'string'},proposal:{type:'string'},controls:{type:'boolean'},control:{type:'string',multiple:true},seconds:{type:'string'}}});
 if(values.controls&&values.control?.length)throw Error('Choose --controls or --control, not both');
 const result=await constructionOracle({root:values.root??'.',proposal:JSON.parse(await readFile(values.proposal,'utf8')),controls:values.controls??values.control??false,seconds:Number(values.seconds??30)});
 console.log(JSON.stringify(result));return result;
}
if(import.meta.main)await oracleMain();
