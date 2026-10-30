"""Input editor integration uses the actual installed-module CLI boundary."""
import json
import time

import pytest

from dashboard.app import create_app


@pytest.fixture
def editor(tmp_path):
    instance = tmp_path / 'instance with spaces'
    (instance / 'config').mkdir(parents=True)
    (instance / 'source folder').mkdir()
    (instance / 'instance.json').write_text('{}')
    (instance / 'config/instance.json').write_text(json.dumps({
        'kdb': 'baseline/kdb', 'skills': 'baseline/skills', 'skill_map': 'baseline/map'}))
    config = tmp_path / 'dashboard.json'
    config.write_text(json.dumps({'rsi_instance': str(instance), 'controls': {'enabled': True}}))
    app = create_app(config, tmp_path / 'cache.sqlite')
    yield app, app.test_client(), instance
    app.extensions['dashboard_controls'].close()


def settle(app, response):
    assert response.status_code == 202, response.data
    for _ in range(500):
        operation = app.extensions['dashboard_controls'].operation(response.json['operation_id'])
        if operation['status'] in ('succeeded', 'failed'):
            return operation
        time.sleep(.01)
    pytest.fail('save did not finish')


def test_actual_cli_edit_revision_and_folder_selection(editor):
    app, client, instance = editor
    assert client.get('/rsi').status_code == 200
    snapshot = client.get('/api/rsi/config').json
    listing = client.get('/api/rsi/folders').json
    selected = next(row['path'] for row in listing['folders'] if row['name'] == 'source folder')
    snapshot['payload']['phase2_projects'] = [dict(project_id='sum', goal='Stable sum',
        command=['python', 'main.py'], starting_folder=selected, performance_goal='under 1s')]
    snapshot['payload']['future'] = {'retained': [1, 2]}
    snapshot['payload']['budgets']['actor_seconds'] = None
    headers = {'X-CSRF-Token': client.get('/api/csrf').json['token'], 'Idempotency-Key': 'save'}
    response = client.post('/api/rsi/config', json=snapshot, headers=headers)
    operation = settle(app, response)
    assert operation['status'] == 'succeeded', operation
    assert client.post('/api/rsi/config', json=snapshot, headers=headers).json == response.json
    current = client.get('/api/rsi/config').json
    assert current['payload']['future'] == {'retained': [1, 2]}
    assert current['payload']['budgets']['actor_seconds'] is None
    assert current['payload']['phase2_projects'][0]['starting_folder'] == selected
    stale = settle(app, client.post('/api/rsi/config', json=snapshot,
        headers={**headers, 'Idempotency-Key': 'stale'}))
    assert stale['status'] == 'failed'
    assert 'revision conflict' in stale['error']
    assert client.get('/api/rsi/folders', query_string={'path': str(instance.parent)}).status_code == 400


def test_guards_and_invalid_write(editor):
    app, client, _ = editor
    snapshot = client.get('/api/rsi/config').json
    assert client.post('/api/rsi/config', json=snapshot).status_code == 403
    token = client.get('/api/csrf').json['token']
    headers = {'X-CSRF-Token': token, 'Idempotency-Key': 'invalid'}
    assert client.post('/api/rsi/config', json={}, headers=headers).status_code == 409
    assert client.post('/api/rsi/config', json=snapshot, headers=headers,
                       environ_base={'REMOTE_ADDR': '192.0.2.1'}).status_code == 403
    app.extensions['dashboard_controls'].enabled = False
    assert client.get('/rsi').status_code == 200
    assert client.get('/api/rsi/config').status_code == 200
    assert client.post('/api/rsi/config', json=snapshot, headers=headers).status_code == 403


def test_registration_is_explicit(tmp_path):
    config = tmp_path / 'dashboard.json'
    config.write_text('{}')
    app = create_app(config, tmp_path / 'cache.sqlite')
    try:
        assert app.test_client().get('/rsi').status_code == 404
        assert app.test_client().get('/api/rsi/config').status_code == 404
    finally:
        app.extensions['dashboard_controls'].close()


def test_generic_controls_cannot_bypass_loopback(editor):
    _, client, _ = editor
    snapshot = client.get('/api/rsi/config').json
    response = client.post('/api/control/rsi-config-set', json=snapshot,
        headers={'X-CSRF-Token': client.get('/api/csrf').json['token'], 'Idempotency-Key': 'remote'},
        environ_base={'REMOTE_ADDR': '192.0.2.1'})
    assert response.status_code == 403


def test_cli_runner_receives_argv_without_shell(tmp_path):
    from types import SimpleNamespace
    calls = []
    def runner(argv, **options):
        calls.append((argv, options))
        return SimpleNamespace(returncode=0, stdout='{}', stderr='')
    config = tmp_path / 'dashboard.json'
    instance = tmp_path / 'space and ; text'
    config.write_text(json.dumps({'rsi_instance': str(instance)}))
    app = create_app(config, tmp_path / 'cache.sqlite', rsi_runner=runner)
    try:
        assert app.test_client().get('/api/rsi/folders', query_string={'path': '/safe/space ; text'}).status_code == 200
        argv, options = calls[-1]
        assert argv[1:] == ['-m', 'rsi', '--instance', str(instance), 'folders', 'list', '/safe/space ; text']
        assert options['shell'] is False
        assert options['timeout'] == 30
    finally:
        app.extensions['dashboard_controls'].close()


def test_browser_forms_preserve_advanced_fields_and_edit_projects():
    """Exercise the shipped controller using Node's built-in VM, without a browser dependency."""
    from pathlib import Path
    import shutil
    import subprocess
    node = shutil.which('node')
    if not node:
        pytest.skip('Node is unavailable')
    script = r'''
const assert = require('node:assert/strict'), vm = require('node:vm');
const nodes = new Map();
class Element {
  constructor(tag) { this.tagName=tag; this.children=[]; this.listeners={}; this.dataset={}; this.value=''; }
  append(...children) { this.children.push(...children); }
  replaceChildren(...children) { this.children=children; }
  setAttribute(key,value) { this[key]=value; }
  setCustomValidity(value) { this.invalid=value; }
  addEventListener(kind,fn) { this.listeners[kind]=fn; }
  dispatchEvent(event) { this.listeners[event.type]?.(event); }
  showModal() {} close() {}
}
const get=id=>{if(!nodes.has(id))nodes.set(id,new Element('div'));return nodes.get(id);};
const add=new Element('button');add.dataset.add='phase2_projects';
const addHeldout=new Element('button');addHeldout.dataset.add='heldout_cases';
get('rsi-inputs').dataset.enabled='true';get('rsi-inputs').querySelectorAll=()=>[add,addHeldout];
const baseline={prompts:{curriculum:'learn',actor:'work',waves:'wave',skillwriter:'write'},
 kdb:'seed',skills:'skills',skill_map:'map',provider:{profile_id:'deepseek',model:'model'},
 roles:{},budgets:{design_iterations:3,implementation_iterations:5,execution_tries:20,actor_seconds:1800,wall_seconds:null},
 recovery:{enabled:true,future:'keep'},tuning:{every_projects:5,heldout_projects:5,runs_per_arm:2,at_wave_end:true},
 phase1_projects:[],phase2_projects:[],project_configs:{unlisted:{unknown:'keep'}},future:{opaque:42}};
let posted;
const context={document:{getElementById:get,createElement:tag=>new Element(tag),body:{dataset:{csrf:'token'}}},
 console,Event:class{constructor(type){this.type=type;}},crypto:{randomUUID:()=> 'key'},
 setTimeout:fn=>fn(),fetch:async(url,options)=>{
 let body;
 if(url==='/api/rsi/config' && options) {posted=JSON.parse(options.body);body={operation_id:'saved'};}
 else if(url.startsWith('/api/operations/'))body={status:'succeeded',result:{revision:'new',payload:posted.payload}};
 else if(url.startsWith('/api/rsi/folders'))body={path:'/root/selected',roots:['/root'],folders:[]};
 else body={revision:'old',payload:baseline};
 return {ok:true,json:async()=>body};
}};
vm.runInNewContext(require('node:fs').readFileSync(process.argv[1],'utf8'),context);
const tick=()=>new Promise(resolve=>setImmediate(resolve));
const flatten=node=>[node,...node.children.filter(x=>typeof x==='object').flatMap(flatten)];
const field=(id,title)=>flatten(get(id)).find(n=>n['aria-label']===title);
function change(input,value) {input.value=value;input.dispatchEvent({type:'input'});assert.ok(!input.invalid);}
(async()=>{
 await tick();
 add.onclick();
 change(field('phase2-projects','Project ID'),'stable-sum');
 change(field('phase2-projects','Goal'),'Accurate sum');
 await get('inputs-form').onsubmit({preventDefault(){}});
 assert.equal(posted,undefined); // Missing host execution contract cannot publish silently.
 flatten(get('phase2-projects')).find(n=>n.tagName==='button'&&n.textContent==='Use CPU Python template commands').listeners.click();
 change(field('phase2-projects','Test arguments (JSON array)'),'["python","verify.py"]');
 // Actual folder response populates the source input through the picker.
 const sourceLabel=flatten(get('phase2-projects')).find(n=>n.children.includes(field('phase2-projects','Starting folder')));
 sourceLabel.children.find(n=>n.tagName==='button').listeners.click();await tick();get('select-folder').onclick();
 change(field('phase2-projects','Frozen verifier file path'),'/root/selected/verify.py');
 change(field('phase2-projects','Project ID'),'renamed-sum');
 change(field('phase2-projects','Performance threshold seconds, optional'),'0.05');
 await get('inputs-form').onsubmit({preventDefault(){}});
 assert.equal(posted,undefined); // Performance threshold needs a real timing argv.
 change(field('phase2-projects','Timing arguments, optional (JSON array)'),'["python","benchmark.py"]');
 // Configure an actual independent held-out project and execution contract.
 addHeldout.onclick();
 change(field('heldout-projects','Project ID'),'heldout-numeric');
 change(field('heldout-projects','Goal'),'Independent cancellation cases');
 change(field('heldout-projects','Starting folder'),'/root/heldout');
 flatten(get('heldout-projects')).find(n=>n.tagName==='button'&&n.textContent==='Use CPU Python template commands').listeners.click();
 change(field('policy-fields','Held-out case count'),'1');
 // Disable an actor cap and ensure null reaches the backend.
 const budgetInput=field('budget-fields','Actor seconds');
 const budgetRow=get('budget-fields').children.find(row=>flatten(row).includes(budgetInput));
 const checkbox=flatten(budgetRow).find(n=>n.type==='checkbox');checkbox.checked=true;checkbox.listeners.change();
 await get('inputs-form').onsubmit({preventDefault(){}});
 assert.equal(posted.revision,'old');assert.equal(posted.payload.future.opaque,42);
 assert.equal(posted.payload.recovery.future,'keep');assert.equal(posted.payload.budgets.actor_seconds,null);
 assert.equal(posted.payload.phase2_projects[0].starting_folder,'/root/selected');
 assert.deepEqual(posted.payload.phase2_projects[0].command,['python','verify.py']);
 assert.equal(posted.payload.project_configs['renamed-sum'].source,'/root/selected');
 assert.equal(posted.payload.project_configs['renamed-sum'].verifier,'/root/selected/verify.py');
 assert.equal(posted.payload.project_configs['renamed-sum'].performance_seconds,0.05);
 assert.deepEqual(posted.payload.project_configs['renamed-sum'].commands.timing,['python','benchmark.py']);
 assert.deepEqual(posted.payload.project_configs['renamed-sum'].commands.test,posted.payload.phase2_projects[0].command);
 assert.equal(posted.payload.project_configs['stable-sum'],undefined);
 assert.equal(posted.payload.project_configs.unlisted.unknown,'keep');
 assert.equal(posted.payload.tuning.heldout_cases[0].project.project_id,'heldout-numeric');
 assert.equal(posted.payload.tuning.heldout_cases[0].execution.verifier,'/root/heldout/verify.py');
 assert.equal(posted.payload.project_configs['heldout-numeric'],undefined);
 // Removing the project persists the deletion with the newly returned revision.
 flatten(get('phase2-projects')).find(n=>n.tagName==='button'&&n.textContent==='Remove project').listeners.click();
 await get('inputs-form').onsubmit({preventDefault(){}});
 assert.equal(posted.revision,'new');assert.equal(posted.payload.phase2_projects.length,0);
 assert.equal(posted.payload.project_configs['renamed-sum'],undefined);
 assert.equal(posted.payload.project_configs.unlisted.unknown,'keep');
 assert.equal(posted.payload.tuning.heldout_cases.length,1);
})().catch(error=>{console.error(error);process.exitCode=1;});
'''
    subprocess.run([node, '-e', script, str(Path(__file__).parents[1] / 'static/rsi_inputs.js')],
                   check=True, capture_output=True, text=True, timeout=20)
