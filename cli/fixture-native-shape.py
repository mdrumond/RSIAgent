"""Identify definite host-only controls; compiled route proof remains dynamic."""
import ast
import json
import sys

v, witnesses = json.load(sys.stdin)
def modules(tree):
    names=set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(n.name for n in node.names)
        if isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
    return names
checker = ast.parse(v['checker'])
process_aliases = {}
for node in ast.walk(checker):
    if isinstance(node, ast.Import):
        for alias in node.names:
            process_aliases[alias.asname or alias.name] = alias.name
    if isinstance(node, ast.ImportFrom) and node.module:
        for alias in node.names:
            process_aliases[alias.asname or alias.name] = node.module + '.' + alias.name
launchers = {'subprocess.run', 'subprocess.call', 'subprocess.check_call', 'subprocess.check_output',
             'subprocess.Popen', 'multiprocessing.Process', 'os.system', 'os.popen'}
for call in (n for n in ast.walk(checker) if isinstance(n, ast.Call)):
    fn = call.func
    if isinstance(fn, ast.Name):
        name = process_aliases.get(fn.id)
    elif isinstance(fn, ast.Attribute) and isinstance(fn.value, ast.Name):
        name = process_aliases.get(fn.value.id, fn.value.id) + '.' + fn.attr
    else:
        continue
    if name in launchers:
        sys.exit('checker: execute the submitted implementation in this Python process so its compile and launch are measured. Load sys.argv[1] via runpy.run_path or importlib, invoke its callable, and independently compare returned values.')
approved=set()
for source in witnesses:
    try: approved.update(modules(ast.parse(source)))
    except SyntaxError: pass
runtime={'torch','torch_npu','sys','importlib','pathlib','math','typing','numpy','os','functools','operator','itertools','collections','builtins','dataclasses'}
for role in ('starter','good','bad'):
    try:
        tree = ast.parse(v[role])
    except SyntaxError as exc:
        expected = f"{type(exc).__name__}: {exc.msg}"
        if role == 'starter' and v[role] in witnesses and v['archivedCause'] == expected:
            # Its failure precedes import. Device qualification still executes the unchanged
            # starter and requires witnessed launches from both numerical controls.
            continue
        sys.exit(f"{role}: {expected}. Only the exact archived starter may reproduce its recorded parse failure.")

    calls=[n for n in ast.walk(tree) if isinstance(n,ast.Call)]
    names=modules(tree)
    for n in ast.walk(tree):
        if role=='starter' and isinstance(n,ast.Raise) and isinstance(n.exc,ast.Call):
            if any(isinstance(s,ast.Constant) and isinstance(s.value,str) and v['diagnostic'] in s.value for s in ast.walk(n.exc)):
                sys.exit('starter: manually raising the observed diagnostic simulates the failure. Keep the archived native operation so the compiler emits it.')
    aliases={}
    for n in ast.walk(tree):
        if isinstance(n,ast.Import):
            for a in n.names: aliases[a.asname or a.name.split('.')[0]]=a.name
        if isinstance(n,ast.ImportFrom) and n.module:
            for a in n.names: aliases[a.asname or a.name]=n.module+'.'+a.name
    def full(n):
        if isinstance(n,ast.Name):return aliases.get(n.id,n.id)
        if isinstance(n,ast.Attribute):return full(n.value)+'.'+n.attr
        return ''
    native=any(full(c.func).startswith('catlass.') and full(c.func).endswith('.compile') for c in calls)
    helpers={m for m in names&approved if m.split('.')[0] not in runtime|{'catlass'}}
    helper=any(full(c.func)==m or full(c.func).startswith(m+'.') for m in helpers for c in calls)
    # Also permit an archived helper under catlass, or a wrapper invoked by the
    # checker through a callback/kernel export. Execution still has to hit the spy.
    exports=any(isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) and any('catlass.' in full(d.func if isinstance(d,ast.Call) else d) for d in n.decorator_list) for n in ast.walk(tree))
    sdk=any(full(c.func).startswith('catlass.') and any(full(c.func).startswith(m+'.') for m in approved if m.startswith('catlass.')) for c in calls)
    if not(native or helper or exports or sdk):
        sys.exit(role+': no native compilation path, traced Catlass kernel export, or invocation of an archived native helper was found. Supply the archived device implementation and its actual calling convention; Torch/CPU-only arithmetic cannot pass the route check.')
