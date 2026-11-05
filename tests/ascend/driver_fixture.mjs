/** Frozen OpenAI chat-completions SSE fixture; production Pi owns all execution. */
import { createServer } from 'node:http';
import { once } from 'node:events';
const tool = (name, args = {}) => ({name, args});
const steps = ['preflight','task-construction','design','implementation','test','debug',
  'optimisation','validation','acceptance'];
function workflow({pass, interrupt}) {
  const calls = [];
  for (const stage of steps) {
    calls.push(tool('rsi_skill'));
    if (['design','optimisation'].includes(stage)) {
      for (const kind of ['good','negative']) calls.push(tool('rsi_knowledge',{query:'arithmetic answer',kind}));
    }
    if (stage === 'implementation') {
      if (pass) calls.push(tool('write',{path:'answer.py',content:'def answer(): return 6 * 7\n'}));
      calls.push(tool('rsi_target',{command:interrupt ?
        'python -c "import time; print(\'holding execution\',flush=True); time.sleep(30)"' :
        'python -c "import answer; print(answer.answer())"'}));
    }
    if (stage === 'validation') {
      calls.push(tool('rsi_verify'));
      if (!pass) break;
    }
    calls.push(tool('rsi_advance',{reflection:'Use independent arithmetic oracle; reject the observed zero baseline.'}));
  }
  return calls;
}
export async function provider(scenario, {nativeSession} = {}) {
  const requests = [], failures = [], plans = new Map();
  let originals = 0, writers = 0;
  const server = createServer(async (req,res) => {
    try {
      if (req.url !== '/v1/chat/completions') throw new Error(`Unexpected provider route ${req.url}`);
      let raw = ''; for await (const chunk of req) raw += chunk;
      const body = JSON.parse(raw); requests.push(body);
      const user = body.messages.find(message => message.role === 'user');
      const prompt = typeof user?.content === 'string' ? user.content :
        (user?.content ?? []).map(block => block.text ?? '').join('');
      const count = body.messages.reduce((n,message) => n + (message.tool_calls?.length ?? 0),0);
      const key = prompt.startsWith('Goal:') ? prompt.split('\n')[0] : prompt;
      if (!count) {
        let plan;
        if (prompt.startsWith('Read the failed')) {
          const verifier = 'import importlib.util\nfrom pathlib import Path\n' +
            'p=Path(__file__).parent/"source"/"answer.py"\n' +
            's=importlib.util.spec_from_file_location("candidate",p)\nm=importlib.util.module_from_spec(s)\n' +
            's.loader.exec_module(m)\nactual=m.answer(); expected=sum(6 for _ in range(7))\n' +
            'print({"actual":actual,"expected":expected},flush=True)\nassert actual==expected\n';
          plan = [tool('write',{path:'verifier.py',content:verifier}),
            tool('investigation_knowledge',{query:'independent numeric oracle',kind:'good'}),
            tool('investigation_knowledge',{query:'numeric mismatch recovery',kind:'negative'}),
            tool('submit_investigation',{goal:'SMALL arithmetic observation',
              question:'Does six times seven satisfy the independent oracle?',verifier:'verifier.py'})];
        } else if (prompt.startsWith('Inspect the native')) {
          const context = JSON.parse(prompt.slice(prompt.indexOf('{'),prompt.lastIndexOf('}')+1));
          const events = context.events;
          const verification = events.findLast(event => event.type === 'verification' && event.data.evidence);
          const path = verification?.data.evidence?.correctness ?? context.session ?? await nativeSession?.();
          if (!path) throw new Error('Summary has neither verifier output nor an actual native session file');
          const passed = verification?.data.passed ?? prompt.includes('passed: true');
          plan = [tool('read',{path}),tool('record_findings',{findings:verification ? (passed ?
            'Independent arithmetic oracle passed; attached verifier output records the result.' :
            'Independent arithmetic oracle failed; attached verifier output records the actual mismatch.') :
            'Attached native session records observed execution and tool results; no separate numeric claim is made.',
            useful:scenario !== 'freeze' && prompt.includes('this investigation') && passed,paths:[path]})];
        } else if (prompt.startsWith('Read current-guidance.json')) {
          plan=[tool('read',{path:'current-guidance.json'}),tool('read',{path:'current-usage.json'}),
            tool('mcp__kdb__search',{query:'independent numeric oracle',kind:'good',limit:1}),
            tool('submit_changes',{changes:writers++===0?[{action:'upsert',id:'numeric-oracle',scope:'debug',
              text:'Use the independent numeric oracle before accepting a candidate.',critical:false,negative:false}]:[]})];
        } else if (prompt.startsWith('Goal:')) {
          const unrelated = prompt.includes('Goal: unrelated');
          const small = prompt.includes('Goal: SMALL');
          const pass = unrelated || scenario === 'phase2' || scenario === 'interrupt' || (scenario !== 'freeze' && (small || originals++ > 0));
          plan = workflow({pass,interrupt:scenario === 'interrupt'&&!unrelated});
        } else throw new Error(`Unknown scripted conversation: ${prompt.slice(0,100)}`);
        plans.set(key,plan);
      }
      const plan = plans.get(key);
      if (!plan) throw new Error('Conversation has no frozen script');
      const call = plan[count];
      if (call && !body.tools?.some(t => t.function?.name === call.name)) throw new Error(`Missing native tool ${call.name}`);
      res.writeHead(200,{'Content-Type':'text/event-stream','Cache-Control':'no-cache'});
      const frame = (delta,finish_reason=null) => res.write('data: '+JSON.stringify({id:'chatcmpl-fixture',
        object:'chat.completion.chunk',created:1,model:'scripted',choices:[{index:0,delta,finish_reason}]})+'\n\n');
      frame({role:'assistant',...(call ? {tool_calls:[{index:0,id:`fixture-${count}`,type:'function',
        function:{name:call.name,arguments:JSON.stringify(call.args)}}]} : {content:'Observed results recorded.'})});
      frame({},call ? 'tool_calls' : 'stop');res.end('data: [DONE]\n\n');
    } catch(error) {failures.push(error.message);res.writeHead(500);res.end(JSON.stringify({error:{message:error.message}}));}
  });
  server.listen(0,'127.0.0.1');await once(server,'listening');
  return {url:`http://127.0.0.1:${server.address().port}/v1`,requests,failures,
    close:async()=>{server.closeAllConnections();await new Promise(resolve=>server.close(resolve));}};
}
