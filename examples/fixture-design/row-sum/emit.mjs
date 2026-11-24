import {mkdir,writeFile} from 'node:fs/promises';
import {join} from 'node:path';
import {rowSumPackets} from './packets.mjs';

const directory=process.argv[2];
if(!directory)throw Error('Usage: node emit.mjs OUTPUT_DIRECTORY');
const packets=await rowSumPackets();
for(const [key,value] of Object.entries(packets)){
 const dir=join(directory,key==='input'?'public':'evaluator');
 await mkdir(dir,{recursive:true});
 await writeFile(join(dir,key+'.json'),JSON.stringify(value,null,2)+'\n');
}
