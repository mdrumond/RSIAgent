// Run under the shared validation flock with the required Node/SDK installation.
import {spawn,execFile} from 'node:child_process';
import {promisify} from 'node:util';
import {mkdir,writeFile,rm,readFile} from 'node:fs/promises';
import {join} from 'node:path';
const run=promisify(execFile),id='rsivalidation'+process.pid+Date.now(),slice=id+'.slice';
const directory=join(process.env.XDG_RUNTIME_DIR,'systemd/user'),file=join(directory,slice),cap=4*1024**3;
const ctl=args=>run('systemctl',['--user',...args],{timeout:5000,maxBuffer:65536});
let timer,child;
try {
  await mkdir(directory,{recursive:true});
  await writeFile(file,'[Slice]\nMemoryAccounting=yes\nMemoryMax='+cap+'\nMemorySwapMax=0\n');
  await ctl(['daemon-reload']);await ctl(['start',slice]);
  const suites=process.argv.slice(2);
  const env={...process.env,RSI_TEST_SLICE:slice,RSI_MANAGED_CLI_TEST:'1',RSI_MANAGED_SYSTEMD_TEST:'1'};
  const args=['--user','--quiet','--wait','--pipe','--collect','--service-type=exec','--expand-environment=no',
    '--unit='+id+'.service','--slice='+slice,'--property=MemoryMax='+cap,'--property=MemorySwapMax=0',
    '--property=OOMPolicy=kill','--property=KillMode=control-group','--working-directory='+process.cwd(),
    ...Object.entries(env).map(([key,value])=>'--setenv='+key+'='+value),
    '--',process.execPath,'--test','--test-concurrency=1',...(suites.length?suites:[
      'tests/ascend/campaign-budget.mjs','tests/ascend/managed-adoption.mjs','tests/ascend/managed-runtime.mjs',
      'tests/ascend/managed-cli.mjs','tests/ascend/project-failure-containment.mjs','tests/ascend/host-memory-configure.mjs'])];
  child=spawn('systemd-run',args,{stdio:'inherit'});
  timer=setTimeout(()=>{void ctl(['stop',slice]);},180000);
  process.exitCode=await new Promise((resolve,reject)=>{child.once('error',reject);child.once('close',code=>resolve(code??1));});
  if(process.env.RSI_MANAGED_PROOF) {
    const group=(await ctl(['show',slice,'--property=ControlGroup','--value'])).stdout.trim();
    const base='/sys/fs/cgroup'+group;
    await writeFile(process.env.RSI_MANAGED_PROOF+'.envelope.json',JSON.stringify({slice,group,exit_code:process.exitCode,
      memory_max:Number((await readFile(base+'/memory.max','utf8')).trim()),
      memory_peak:Number((await readFile(base+'/memory.peak','utf8')).trim()),
      events:(await readFile(base+'/memory.events','utf8')).trim()},null,2));
  }
} finally {
  clearTimeout(timer);await ctl(['stop',slice]);await rm(file,{force:true});await ctl(['daemon-reload']);
}
