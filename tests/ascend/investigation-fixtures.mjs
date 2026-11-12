import {mkdir,writeFile} from 'node:fs/promises';
import {join} from 'node:path';

export const controls={positive:'positive',negative:['negative']};
export async function numericControls(root,{callable=false}={}) {
  for(const [name,value] of [['positive',42],['negative',0]]) {
    await mkdir(join(root,name),{recursive:true});
    await writeFile(join(root,name,'answer.py'),callable?`def answer(): return ${value}\n`:`answer=${value}\n`);
  }
}
