import {relative,resolve,isAbsolute,sep} from 'node:path';

const fields=['command','baseline_command','performance_command'];
const quote=value=>"'"+value.replaceAll("'","'\\''")+"'";

// Omit the source cwd for CPU construction, which runs from several fixture
// directories and must keep the same frozen entrypoint identity in each one.
export function entrypointCommand({entrypoint,args},source) {
  return ['python3',source===undefined?entrypoint:relative(source,entrypoint),...args].map(quote).join(' ');
}

// Generated verifier commands are Python entrypoints with literal arguments.
// Store the file identity as well as the compatible shell command so imports
// can regenerate paths without interpreting or repairing shell programs.
export function compileVerifierEntries(verifier,{root,source,files,required=false}) {
  if(!verifier)return verifier;
  const frozen=new Set((files??verifier.files??[]).map(file=>resolve(file))),entries={},result={...verifier};
  for(const field of fields) {
    const submitted=verifier.entrypoints?.[field]??verifier[field];
    if(submitted===undefined)continue;
    if(typeof submitted==='string') {
      if(required)throw new Error(`Generated verifier.${field} requires an entrypoint and literal args`);
      continue;
    }
    if(!submitted||typeof submitted.entrypoint!=='string'||!submitted.entrypoint.trim()||
        !Array.isArray(submitted.args)||submitted.args.some(arg=>typeof arg!=='string'||arg.includes('\0')))
      throw new Error(`Invalid verifier.${field} entrypoint or args`);
    if(required&&submitted.args.some(arg=>isAbsolute(arg)||
        (arg.startsWith('-')&&arg.includes('=')&&isAbsolute(arg.slice(arg.indexOf('=')+1)))||arg.includes(resolve(root)+sep)))
      throw new Error('Generated verifier args must use source-relative paths, not absolute generation paths');
    const entrypoint=resolve(root,submitted.entrypoint);
    if(!frozen.has(entrypoint))throw new Error(`Verifier entrypoint must be a declared frozen file: ${submitted.entrypoint}`);
    entries[field]={entrypoint,args:[...submitted.args]};
    result[field]=entrypointCommand(entries[field],source);
  }
  if(Object.keys(entries).length)result.entrypoints=entries;
  return result;
}
