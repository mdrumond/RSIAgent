"""Campaign home reads CLI execution data without dispatching work."""
from html.parser import HTMLParser
import json
from pathlib import Path
import shutil
import subprocess
from types import SimpleNamespace

import pytest

from dashboard.app import create_app


class Page(HTMLParser):
    def __init__(self, body):
        super().__init__()
        self.elements = []
        self.feed(body)
    def handle_starttag(self, tag, attrs):
        self.elements.append((tag, dict(attrs)))


@pytest.mark.parametrize('bound', [True, False])
def test_instance_navigation_and_home_are_conditional_without_changing_legacy_routes(tmp_path, bound):
    config = tmp_path / 'dashboard.json'
    config.write_text(json.dumps({'rsi_instance': str(tmp_path / 'instance')} if bound else {}))
    calls = []
    def runner(argv, **kwargs):
        calls.append(argv[5:])
        return SimpleNamespace(returncode=0, stdout=json.dumps([] if argv[5:] == ['job', 'list'] else {}), stderr='')
    app = create_app(config, tmp_path / 'cache.sqlite', rsi_runner=runner)
    try:
        client = app.test_client()
        response = client.get('/')
        assert response.status_code == 200
        page = Page(response.get_data(as_text=True))
        links = [attrs.get('href') for tag, attrs in page.elements if tag == 'a']
        assert ('/rsi/execution' in links) is bound
        ids = [attrs.get('id') for _, attrs in page.elements]
        assert ('instance-campaigns' in ids) is bound
        scripts = [attrs.get('src') for tag, attrs in page.elements if tag == 'script']
        assert ('/static/instance_campaigns.js' in scripts) is bound
        assert '/memory' in links and '/' in links
        # Rendering the page never starts work or invokes the job subprocess.
        assert calls == []
        assert client.get('/api/rsi/jobs').status_code == (200 if bound else 404)
        assert all(call == ['job', 'list'] or call == ['capabilities'] for call in calls)
    finally:
        app.extensions['dashboard_controls'].close()


def test_home_browser_refreshes_current_cli_jobs_preserves_failures_and_uses_text_nodes():
    node = shutil.which('node')
    if not node:
        pytest.skip('Node is needed for the browser behavior test')
    script = r"""
const assert = require('node:assert/strict'), vm = require('node:vm');
const source = require('node:fs').readFileSync(process.argv[1], 'utf8');
class Element {
  constructor(tag) { this.tag=tag; this.children=[]; this.dataset={}; this.value=''; this._text=''; }
  set textContent(v) { this._text=String(v); this.children=[]; }
  get textContent() { return this._text + this.children.map(x=>x.textContent).join(' '); }
  append(...nodes) { this.children.push(...nodes); }
  replaceChildren(...nodes) { this._text=''; this.children=nodes; }
  addEventListener(event, fn) { this[event]=fn; }
}
const ids = Object.fromEntries(['instance-campaigns','instance-campaigns-list',
  'instance-campaigns-status','instance-campaigns-refresh'].map(id=>[id,new Element('div')]));
ids['instance-campaigns'].dataset.executionUrl='/rsi/execution';
let nextTimer, sequence=[], requests=[];
const context={document:{getElementById:id=>ids[id],createElement:tag=>new Element(tag)},
  setTimeout:(fn,ms)=>{assert.equal(ms,10000); nextTimer=fn; return 1;},clearTimeout:()=>{},
  fetch:async(url,opts)=>{
    requests.push(url);assert.equal(opts.cache,'no-store');
    const item=sequence.shift();assert.ok(item,'unexpected extra request');
    if(item instanceof Error) throw item;
    return {ok:item.ok!==false,json:async()=>item.body};
  }};
const job={job_id:'current / job',cycle_id:'cycle-one',mode:'phase1',state:'running',previous_job:'old'};
const project={project_id:'<img src=x onerror=bad()>',status:'running',stage:'implementation',
  actor_seconds:25,actor_allocation:900,remaining_actor_seconds:875,wave_actor_ceiling:6300,
  device:'4',pending:{handle:'remote:target:job:retained'}};
const detail={job,inspection:{available:true,tasks:[],waves:[{wave_id:'wave-one',status:'executing',projects:[
  project,{project_id:'waiting',status:'reserved'},
  {project_id:'unbounded',status:'completed',stage:'test',actor_allocation:null,
   remaining_actor_seconds:null,actor_seconds:7},
  {project_id:'denied',status:'observation_required',admission:{admitted:false,failure:{type:'probe_unknown'}}}
]}]}};
(async()=>{
  sequence=[{body:{jobs:[{job_id:'old',cycle_id:'cycle-one',state:'completed'},job]}},{body:detail}];
  await vm.runInNewContext(source,context);
  assert.deepEqual(requests,['/api/rsi/jobs','/api/rsi/jobs/current%20%2F%20job']);
  const list=ids['instance-campaigns-list'];
  assert.equal(list.children.length,1);
  const visible=list.textContent;
  for(const value of [project.project_id,'implementation','25.0 s / 900.0 s / 875.0 s',
    'Wave ceiling: 6300.0 s','unknown / unknown / unknown','7.0 s / unlimited / unlimited',
    'observation_required','probe_unknown','remote:target:job:retained']) assert.ok(visible.includes(value), value);
  function walk(x) { return [x,...x.children.flatMap(walk)]; }
  assert.equal(walk(list).filter(x=>x.tag==='img').length,0);
  assert.ok(walk(list).some(x=>x.tag==='a'&&x.href==='/rsi/execution'));
  assert.equal(ids['instance-campaigns-refresh'].disabled,false);

  // A list failure retains the previous view and still schedules a retry.
  sequence=[{ok:false,body:{error:'CLI temporarily unavailable'}}];
  await nextTimer();
  assert.equal(list.children.length,1);
  assert.ok(ids['instance-campaigns-status'].textContent.includes('CLI temporarily unavailable'));

  // A project evidence failure remains visible next to its authoritative job state.
  sequence=[{body:{jobs:[job]}},{ok:false,body:{error:'Retained evidence unavailable'}}];
  await ids['instance-campaigns-refresh'].click();
  assert.ok(list.textContent.includes('Retained evidence unavailable'));
  assert.ok(list.textContent.includes('running'));

  // A genuinely empty successful response clears obsolete job cards.
  sequence=[{body:{jobs:[]}}];
  await nextTimer();
  assert.equal(list.children.length,0);
  assert.equal(sequence.length,0);

  let fetched=false;
  vm.runInNewContext(source,{document:{getElementById:()=>null},fetch:()=>fetched=true});
  assert.equal(fetched,false);
})().catch(error=>{console.error(error);process.exitCode=1;});
"""
    result = subprocess.run([node, '-e', script, str(Path(__file__).parents[1] / 'static/instance_campaigns.js')],
                            capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stdout + result.stderr
