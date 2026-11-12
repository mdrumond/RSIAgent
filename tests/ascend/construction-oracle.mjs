import test from 'node:test';
import assert from 'node:assert/strict';
import {mkdtemp,mkdir,writeFile,rm} from 'node:fs/promises';
import {join,dirname} from 'node:path';
import {tmpdir} from 'node:os';
import {constructionOracle} from '../../cli/microtasks/construction-oracle.mjs';
import {constructionFixture} from '../../cli/microtasks/hypothesis-fixtures.mjs';

test('production construction oracle accepts top-level baseline CPU commands and rejects nested-only placement',async t=>{
 const root=await mkdtemp(join(tmpdir(),'construction-oracle-'));t.after(()=>rm(root,{recursive:true,force:true}));
 const {files,proposal}=constructionFixture();for(const [name,text]of Object.entries(files)){await mkdir(dirname(join(root,name)),{recursive:true});await writeFile(join(root,name),text);}
 assert.equal((await constructionOracle({root,proposal})).admitted,true);
 const nested=structuredClone(proposal);nested.verifier.baseline_construction_command=nested.baseline_construction_command;delete nested.baseline_construction_command;
 assert.equal((await constructionOracle({root,proposal:nested})).admitted,false);
 const file=structuredClone(proposal);file.source+='\/implementation.py';assert.equal((await constructionOracle({root,proposal:file})).admitted,false);
 const ownership=structuredClone(proposal);ownership.verifier.files.push('reference/implementation.py');assert.equal((await constructionOracle({root,proposal:ownership})).admitted,false);
 const wrong=structuredClone(proposal);wrong.baseline_construction_command.entrypoint='quick.py';assert.equal((await constructionOracle({root,proposal:wrong})).admitted,false);
 const accepted=await constructionOracle({root,proposal,controls:true});assert.equal(accepted.admitted,true);assert.deepEqual(accepted.checks.map(({kind,exitCode})=>({kind,exitCode})),[{kind:'reference',exitCode:0},{kind:'negative',exitCode:1},{kind:'starter',exitCode:0},{kind:'baseline',exitCode:0}]);
 const badMode=structuredClone(proposal);badMode.baseline_construction_command.args=['--mode','little'];assert.equal((await constructionOracle({root,proposal:badMode,controls:true})).admitted,false);
});
