import { spawn } from 'node:child_process';
import { pipeline } from 'node:stream/promises';
import { mkdir, mkdtemp, rm, access } from 'node:fs/promises';
import { resolve, join } from 'node:path';
import { tmpdir } from 'node:os';
import { createHash } from 'node:crypto';
import { isDeepStrictEqual } from 'node:util';

const encode = value => JSON.stringify(value) + '\n';
const digest = value => createHash('sha256').update(value).digest('hex');
const id = value => {
  if (typeof value !== 'string' || !/^[a-zA-Z0-9_-]+$/.test(value)) throw new Error('Invalid id');
  return value;
};

/** Small campaign-local Git history; caller owns cross-process writer exclusion.
 * indexRevision(revision, snapshot) returns a generation identity.
 * activate({revision,generation}) must tolerate retries after a process crash.
 * Only structured lessons and skill text belong here; keep traces/databases elsewhere.
 */
export class KnowledgeHistory {
  constructor(directory) {
    this.directory = resolve(directory);
    this.pending = Promise.resolve();
  }

  git(args, { input = '', env = {}, raw = false } = {}) {
    return new Promise((ok, fail) => {
      const child = spawn('git', ['-C', this.directory, ...args], {
        env: { ...process.env, GIT_AUTHOR_NAME: 'Knowledge history', GIT_AUTHOR_EMAIL: 'knowledge@localhost',
          GIT_COMMITTER_NAME: 'Knowledge history', GIT_COMMITTER_EMAIL: 'knowledge@localhost', ...env },
        stdio: ['pipe', 'pipe', 'pipe'],
      });
      let stdout = '', stderr = '';
      child.stdout.setEncoding('utf8');child.stderr.setEncoding('utf8');
      child.stdout.on('data', chunk => { stdout += chunk; });
      child.stderr.on('data', chunk => { stderr += chunk; });
      child.on('error', fail);
      child.stdin.on('error', error => { if (error.code !== 'EPIPE') fail(error); });
      child.on('close', code => code === 0 ? ok(raw ? stdout : stdout.trimEnd()) :
        fail(Object.assign(new Error(`git ${args[0]}: ${stderr}`), { code })));
      child.stdin.end(input);
    });
  }

  write(fn) {
    const next = this.pending.then(fn);
    this.pending = next.catch(() => {});
    return next;
  }

  async ref(ref) {
    try { return await this.git(['show-ref', '--verify', '--hash', ref]); }
    catch (error) { if (error.code === 1 || error.code === 128) return null; throw error; }
  }

  // Older checkpoints used a bare repository at the history root. A previous
  // initializer could create .git inside it, hiding its objects from newer writes.
  async importLegacyBare() {
    const legacy=['--git-dir='+this.directory];
    if(await this.git([...legacy,'remote']))throw new Error('Knowledge history must be remoteless');
    const refs=(await this.git([...legacy,'for-each-ref','--format=%(refname) %(objectname)']))
      .split('\n').filter(Boolean).map(line=>line.split(' '));
    const missing=[];
    for(const [ref,revision] of refs) {
      const alias='refs/knowledge/legacy-bare/'+ref.slice('refs/'.length);
      const existing=await this.ref(alias);
      if(existing&&existing!==revision)throw new Error('Legacy knowledge history reference changed: '+ref);
      if(!existing)missing.push([alias,revision]);
    }
    if(!missing.length)return;
    // Stream binary pack bytes directly between native Git processes. Never
    // buffer/decode object data in JS or depend on an external object directory.
    const pack=spawn('git',[...legacy,'-c','pack.threads=1','pack-objects','--window=0','--all','--stdout'],{stdio:['ignore','pipe','pipe']});
    const index=spawn('git',['-C',this.directory,'index-pack','--stdin'],{stdio:['pipe','ignore','pipe']});
    const exited=child=>new Promise((ok,fail)=>{
      let stderr='';child.stderr.on('data',chunk=>{stderr=(stderr+chunk).slice(-65536);});
      child.on('error',fail);
      child.on('close',code=>code===0?ok():fail(new Error('Legacy knowledge object import failed: '+stderr)));
    });
    const children=[exited(pack),exited(index)];
    try {await Promise.all([...children,pipeline(pack.stdout,index.stdin)]);}
    catch(error) {pack.kill();index.kill();await Promise.allSettled(children);throw error;}
    // Keep all imported objects reachable without replacing current activation,
    // memory, skill or delivery refs. The original bare refs remain untouched.
    await this.git(['update-ref','--stdin'],{input:'start\n'+missing.map(([ref,revision])=>
      `create ${ref} ${revision}\n`).join('')+'prepare\ncommit\n'});
  }

  async init() {
    await mkdir(this.directory, { recursive: true });
    let ownedRoot=false,nested=false;
    try {
      await access(join(this.directory,'HEAD'));
      // --is-bare-repository reflects worktree discovery/configuration, not
      // ownership of this retained root object store.
      ownedRoot=resolve(await this.git(['--git-dir='+this.directory,'rev-parse','--absolute-git-dir']))===this.directory;
    }catch(error){if(error.code!=='ENOENT')throw error;}
    try {await access(join(this.directory,'.git'));nested=true;}
    catch(error){if(error.code!=='ENOENT')throw error;}
    if(nested) {
      if(resolve(await this.git(['rev-parse','--show-toplevel']))!==this.directory)
        throw new Error('Knowledge history is not its own repository');
    }else if(!ownedRoot)await this.git(['init', '--initial-branch=memory']);
    if (await this.git(['remote'])) throw new Error('Knowledge history must be remoteless');
    if(ownedRoot&&nested)await this.importLegacyBare();
    if (!(await this.ref('refs/heads/memory'))) {
      if (await this.git(['rev-list', '--all'])) throw new Error('Not an empty knowledge repository');
      const revision = await this.commit(null, { 'memory/.gitkeep': '', 'skills/.gitkeep': '' }, { type: 'initialize' });
      await this.git(['update-ref', 'refs/heads/memory', revision]);
    }
    await mkdir(join(this.directory, 'memory'), { recursive: true });
    await mkdir(join(this.directory, 'skills'), { recursive: true });
    return this;
  }

  async commit(parent, files, event) {
    const directory = await mkdtemp(join(tmpdir(), 'knowledge-index-'));
    const env = { GIT_INDEX_FILE: join(directory, 'index') };
    try {
      await this.git(parent ? ['read-tree', `${parent}^{tree}`] : ['read-tree', '--empty'], { env });
      for (const [path, input] of Object.entries(files)) {
        const blob = await this.git(['hash-object', '-w', '--stdin'], { input });
        await this.git(['update-index', '--add', '--cacheinfo', '100644', blob, path], { env });
      }
      const tree = await this.git(['write-tree'], { env });
      return await this.git(['-c', 'commit.gpgsign=false', 'commit-tree', tree,
        ...(parent ? ['-p', parent] : []), '-m', encode(event)]);
    } finally { await rm(directory, { recursive: true, force: true }); }
  }

  async state(kind) {
    if (!['memory', 'skills'].includes(kind)) throw new Error('Unknown activation kind');
    const revision = await this.ref(`refs/knowledge/active-${kind}`);
    return revision ? JSON.parse(await this.git(['cat-file', 'blob', revision])) : null;
  }

  async setState(kind, state) {
    const blob = await this.git(['hash-object', '-w', '--stdin'], { input: encode(state) });
    await this.git(['update-ref', `refs/knowledge/active-${kind}`, blob]);
  }

  async readRevision(revision) {
    const commit = await this.git(['rev-parse', '--verify', `${revision}^{commit}`]);
    const paths = (await this.git(['ls-tree', '-r', '--name-only', commit])).split('\n');
    const memory = [], skills = {};
    for (const path of paths.filter(path => path && !path.endsWith('/.gitkeep'))) {
      const content = await this.git(['show', `${commit}:${path}`],{raw:true});
      if (path.startsWith('memory/')) memory.push(JSON.parse(content));
      if (path.startsWith('skills/')) skills[path.slice(7)] = content;
    }
    return { revision: commit, memory, skills,
      parents: (await this.git(['show', '-s', '--format=%P', commit])).split(' ').filter(Boolean),
      event: JSON.parse(await this.git(['show', '-s', '--format=%B', commit])) };
  }

  commitMemory(records, { indexRevision, activate }) {
    return this.write(async () => {
      if (!records.length || typeof indexRevision !== 'function' || typeof activate !== 'function')
        throw new Error('Records, indexRevision and activate are required');
      const files = {}, deliveries = [];
      for (const record of records) {
        const { delivery_id, lesson, evidence_refs } = record;
        if (typeof delivery_id !== 'string' || !delivery_id || typeof lesson !== 'string' || !lesson ||
            !Array.isArray(evidence_refs) || !evidence_refs.length ||
            evidence_refs.some(ref => typeof ref !== 'string' || !ref)) throw new Error('Invalid lesson record');
        const path = `memory/${id(record.id)}.json`;
        if (files[path]) throw new Error('Duplicate record id');
        // Preserve small structured target/kind metadata for the indexing callback.
        const content = encode(record);
        if (Buffer.byteLength(content) > 65536) throw new Error('Lesson exceeds 64 KiB');
        files[path] = content;
        deliveries.push(`refs/knowledge/deliveries/${digest(delivery_id)}`);
      }
      if (new Set(deliveries).size !== deliveries.length) throw new Error('Duplicate delivery id');
      const previous = await Promise.all(deliveries.map(ref => this.ref(ref)));
      // A baseline export or recovered outbox may regroup earlier deliveries.
      // Validate each immutable receipt, not the incidental original batch size.
      for (let i=0;i<records.length;i++) if(previous[i]) {
        const recorded=JSON.parse(await this.git(['show',`${previous[i]}:memory/${records[i].id}.json`]));
        if(!isDeepStrictEqual(records[i],recorded))throw new Error('Delivery id reused with different records');
      }
      const unseen=records.map((record,index)=>({record,index})).filter(({index})=>!previous[index]);
      const parent=await this.ref('refs/heads/memory');
      let revision=parent;
      if(unseen.length) {
        revision = await this.commit(parent,Object.fromEntries(unseen.map(({record})=>
          [`memory/${record.id}.json`,files[`memory/${record.id}.json`]])),
        { type:'memory',deliveries:unseen.map(({record})=>record.delivery_id) });
        // Atomic receipt + branch publication closes the commit/crash/retry window.
        await this.git(['update-ref', '--stdin'], { input:
          `start\nupdate refs/heads/memory ${revision} ${parent}\n` +
          unseen.map(({index}) => `create ${deliveries[index]} ${revision}\n`).join('') + 'prepare\ncommit\n' });
      }
      const active = await this.state('memory');
      if (active?.revision === revision) return active;
      if (active && await this.git(['merge-base', active.revision, revision]) === revision) return active;
      const generation = await indexRevision(revision, await this.readRevision(revision));
      if (generation === undefined || generation === null) throw new Error('Indexer must return generation identity');
      const result = { revision, generation };
      encode(result); // Reject non-persistable identities before the external side effect.
      await activate(result);
      await this.setState('memory', result);
      return result;
    });
  }

  proposeSkill({ id: proposal, skills, metadata = {} }) {
    return this.write(async () => {
      const branch = `refs/heads/skills/${id(proposal)}`;
      if (await this.ref(branch)) throw new Error('Proposal already exists');
      const active = await this.state('skills');
      const parent = active?.revision ?? await this.git(['rev-list', '--max-parents=0', 'refs/heads/memory']);
      const files = {};
      for (const [key, text] of Object.entries(skills)) {
        if (typeof text !== 'string' || !text || Buffer.byteLength(text) > 65536) throw new Error('Invalid skill text');
        files[`skills/${id(key)}.md`] = text;
      }
      if (!Object.keys(files).length) throw new Error('Skills required');
      const revision = await this.commit(parent, files, { type: 'skill-proposal', id: proposal, metadata });
      await this.git(['update-ref', branch, revision]);
      return { revision, branch };
    });
  }

  decideSkill(proposal, { promote, metadata = {}, activate = async () => {} }) {
    return this.write(async () => {
      const branch = `refs/heads/skills/${id(proposal)}`;
      const candidate = await this.ref(branch);
      if (!candidate) throw new Error('Unknown proposal');
      const snapshot = await this.readRevision(candidate);
      const type = promote ? 'skill-promotion' : 'skill-rejection';
      if (snapshot.event.type === 'skill-rejection') {
        if (promote) throw new Error('Proposal was rejected');
        return { revision: candidate };
      }
      const active = await this.state('skills');
      const base = active?.revision ?? await this.git(['rev-list', '--max-parents=0', 'refs/heads/memory']);
      if (promote && snapshot.event.type === 'skill-proposal' && snapshot.parents[0] !== base)
        throw new Error('Proposal is stale; propose against current skills');
      if (snapshot.event.type === 'skill-promotion' && !promote) throw new Error('Proposal was promoted');
      if (snapshot.event.type === 'skill-promotion' && active && active.revision !== candidate &&
          snapshot.event.previous !== active.revision) throw new Error('Promotion was superseded');
      const revision = snapshot.event.type === type ? candidate :
        await this.commit(candidate, {}, { type, id: proposal, candidate, previous: active?.revision ?? null, metadata });
      await this.git(['update-ref', branch, revision]);
      if (promote && active?.revision !== revision) {
        await activate({ revision });
        await this.setState('skills', { revision });
      }
      return { revision };
    });
  }

  promoteSkill(proposal, options = {}) { return this.decideSkill(proposal, { ...options, promote: true }); }
  rejectSkill(proposal, options = {}) { return this.decideSkill(proposal, { ...options, promote: false }); }
}
