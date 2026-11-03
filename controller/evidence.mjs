import { createHash } from 'node:crypto';
import { readdir, readFile, lstat } from 'node:fs/promises';
import { resolve, relative, join, sep, isAbsolute } from 'node:path';

export async function fingerprint(root) {
  const hash=createHash('sha256');
  async function visit(directory) {
    for (const name of (await readdir(directory)).sort()) {
      if (['.git','node_modules','__pycache__','.pytest_cache'].includes(name)) continue;
      const path=join(directory,name), stat=await lstat(path);
      if (stat.isSymbolicLink()) throw new Error('Project symlinks need explicit materialization');
      if (stat.isDirectory()) await visit(path);
      else if (stat.isFile()) { hash.update(relative(root,path)); hash.update('\0'); hash.update(await readFile(path)); }
    }
  }
  await visit(resolve(root));
  return hash.digest('hex');
}

export async function freezeFiles(files,source,expected) {
  const frozen=[];
  for (const path of files) {
    const absolute=resolve(path), rel=relative(resolve(source),absolute);
    if (rel === '' || (!rel.startsWith('..'+sep) && !isAbsolute(rel))) throw new Error('Verifier must be outside editable source');
    const actual=createHash('sha256').update(await readFile(absolute)).digest('hex');
    if(expected && expected[absolute] !== actual)throw new Error('Frozen verifier changed');
    frozen.push([absolute,actual]);
  }
  const check=async()=>{
    for (const [path,expected] of frozen) {
      if (createHash('sha256').update(await readFile(path)).digest('hex') !== expected)
        throw new Error('Frozen verifier changed');
    }
  };
  check.manifest=Object.fromEntries(frozen);
  return check;
}
