"""Resumable curriculum proposals in the existing executable optimizer format.

This module validates data only. Reference source executes in the optimizer's
bounded worker, never in the curriculum process.
"""
from __future__ import annotations

import base64
import json
from pathlib import Path
import re

from optimizer.cases import validate_submission
from optimizer.jobs import atomic, canonical
from optimizer.memory import verify_snapshot
from benchmarks.a5kernels.artifacts import ContentAddressedArtifacts
from .agent_presentation import memory_presentation, compact_observation, artifact_page_presentation


SYSTEM = '''You are the Phase 2 Curriculum Agent. Propose one bounded, useful new
kernel task, informed by reviewed Phase 1 learning and KDB code when available.
Use executable Torch reference source, not a ProjectSpec or operation grammar.
Return one JSON object: {"task_id":"lowercase-slug","submission":{"source":
"import torch\\ndef forward(x): return x + 1\\n","entrypoint":"forward",
"signatures":[{"args":[{"shape":[64],"dtype":"float32"}]}]},
"rationale":"why useful and different","citations":[]}.
source defines a named function or torch.nn.Module class. Optional init_args
(list) and init_kwargs (object) configure a Module. Signatures contain args (list)
and kwargs (object); each argument is {"shape":[positive dimensions],"dtype":
"float32"} or {"scalar":JSON primitive}. Tensor hints optionally specify
distribution (normal/uniform/zeros/ones/alternating), layout (contiguous/transposed),
positive nonoverlapping strides, and constraints (min_shape/max_shape/multiple_of).
Support at most 24 representative signatures and 16M elements per tensor.
Outputs must be tensors, None, or nested tensor lists/tuples/string-keyed dicts.
Keep computation, memory, and runtime bounded. Use consistent dimensions across
matmul inputs/residuals. Generated shape variants may be rejected; every supplied
representative must execute. Do not claim correctness, speed, Cube use, or fusion
from a reference: the host runs and freezes it, then independently checks kernels.
For kind=practice, target the supplied diagnosed weakness and retry plan; do not
replace the fixed final target. For kind=goal, choose a genuinely new composition,
not merely a renamed Phase 1 task. Never name held-out benchmarks.
You may return {"query":"specific kernel code search"} or
{"read_artifact":{"sha256":"hash","offset":0,"limit":8192}} before proposing.
At most six tool reads, including three KDB queries, per proposal. Query KDB before
proposing when enabled. Memory is a bounded index; full evidence is hash-readable.
Return JSON only. No private chain of thought.'''


def _save(path, value):
    atomic(path, canonical(value))


def reject_optimizer_task(root, diagnostic):
    """Feed a host reference rejection back without discarding proposal history."""
    path = Path(root) / 'selection.json'
    state = json.loads(path.read_bytes())
    if state['stage'] != 'selected':
        raise ValueError('only a selected task can receive reference feedback')
    state['history'].append(dict(candidate=state['selected'], diagnostic=str(diagnostic)))
    state.update(stage='selection', selected=None, messages=None, pending=None,
                 diagnostic=str(diagnostic))
    _save(path, state)


def select_optimizer_task(*, complete, snapshot, objective, root, kind='goal',
                          context=None, query_knowledge=None, max_proposals=3):
    """Return task_id/submission/rationale/citations; retain provider/tool turns.

    ``complete(messages)`` returns JSON text. ``query_knowledge(query)`` returns
    the existing gated observation with results[].citation. Provider/transport
    exceptions escape and preserve pending work; invalid proposals consume the
    bounded proposal budget. Use reject_optimizer_task after worker rejection.
    """
    verify_snapshot(snapshot)
    if kind not in {'goal', 'practice'} or not 1 <= max_proposals <= 3:
        raise ValueError('kind must be goal/practice; proposal budget must be 1..3')
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    path = root / 'selection.json'
    identity = dict(snapshot_id=snapshot['snapshot_id'], objective=objective,
                    kind=kind, context=context or {}, kdb=query_knowledge is not None,
                    max_proposals=max_proposals)
    state = (json.loads(path.read_bytes()) if path.exists() else dict(
        identity=identity, stage='selection', proposal_count=0, selected=None,
        history=[], diagnostic=None, messages=None, pending=None))
    if state['identity'] != identity:
        raise ValueError('curriculum selection identity changed; use a fresh directory')
    store = ContentAddressedArtifacts(root / 'artifacts')
    for record in snapshot['files'].values():
        store.put_bytes(base64.b64decode(record['base64'], validate=True))
    seed = canonical(dict(schema='ascend-phase2-curriculum-seed-v1',
                          snapshot_id=snapshot['snapshot_id'],
                          phase1_memory=snapshot['memory_entries']))
    memory = memory_presentation(canonical(dict(initial_memory_base64=
        base64.b64encode(seed).decode(), project_memory_base64=[])), store)
    _save(path, state)
    while state['stage'] != 'selected' and state['proposal_count'] < max_proposals:
        if state['messages'] is None:
            state.update(messages=[dict(role='system', content=SYSTEM), dict(role='user',
                content=json.dumps(dict(objective=objective, kind=kind, context=context or {},
                    memory=memory, kdb_available=query_knowledge is not None,
                    feedback=state['diagnostic'], history=state['history']), sort_keys=True))],
                reads=0, queries=0, citations=[])
            _save(path, state)
        if state['pending'] is None:
            state['pending'] = complete(state['messages'])
            _save(path, state)
        response = state['pending']
        try:
            value = json.loads(response)
            if isinstance(value, dict) and set(value) in ({'query'}, {'read_artifact'}):
                if state['reads'] >= 6:
                    raise ValueError('six tool reads exhausted; return a submission')
                if 'query' in value:
                    if query_knowledge is None or state['queries'] >= 3:
                        raise ValueError('KDB unavailable or three queries exhausted')
                    if not isinstance(value['query'], str) or not value['query'].strip():
                        raise ValueError('knowledge query must be nonempty text')
                    # Transport exceptions intentionally escape without consuming a proposal.
                    observation = query_knowledge(value['query'])
                    state['citations'].extend(row['citation'] for row in observation.get('results', []))
                    state['queries'] += 1
                    observation = compact_observation(observation, store)
                else:
                    request = value['read_artifact']
                    if (not isinstance(request, dict) or type(request.get('offset', 0)) is not int
                            or request.get('offset', 0) < 0 or type(request.get('limit', 8192)) is not int
                            or not 1 <= request.get('limit', 8192) <= 8192):
                        raise ValueError('artifact page requires nonnegative offset and limit 1..8192')
                    try:
                        observation = artifact_page_presentation(request, store)
                    except FileNotFoundError as error:
                        raise ValueError('requested artifact is not retained') from error
                state['reads'] += 1
                state['messages'].extend([dict(role='assistant', content=response),
                    dict(role='user', content=json.dumps(observation))])
                state['pending'] = None
                _save(path, state)
                continue
            if not isinstance(value, dict) or set(value) != {'task_id', 'submission', 'rationale', 'citations'}:
                raise ValueError('expected task_id, submission, rationale and citations')
            if not isinstance(value['task_id'], str) or not re.fullmatch(r'[a-z0-9]+(?:-[a-z0-9]+)*', value['task_id']):
                raise ValueError('task_id must be a lowercase slug')
            if not isinstance(value['rationale'], str) or not value['rationale'].strip():
                raise ValueError('task requires a rationale')
            if not isinstance(value['citations'], list):
                raise ValueError('citations must be a list')
            if query_knowledge is not None and state['queries'] == 0:
                raise ValueError('KDB-enabled curriculum must query kernel code first')
            from benchmarks.a5kernels.phase1_curriculum_selector import _HELD_OUT_WORDS
            if _HELD_OUT_WORDS.search(json.dumps(value)):
                raise ValueError('task names a held-out benchmark')
            value['submission'] = validate_submission(value['submission'])
            if len(value['submission']['signatures']) > 24:
                raise ValueError('representatives exceed 24-case budget')
            # Citations are host-observed, not model-invented.
            value['citations'] = state['citations']
        except (ValueError, KeyError, TypeError) as error:
            state['history'].append(dict(response=response, diagnostic=str(error)))
            state.update(proposal_count=state['proposal_count'] + 1, messages=None,
                         pending=None, diagnostic=str(error))
            _save(path, state)
            continue
        state.update(stage='selected', selected=value, pending=None,
                     proposal_count=state['proposal_count'] + 1, diagnostic=None)
        _save(path, state)
    if state['stage'] != 'selected':
        state['stage'] = 'failed'
        _save(path, state)
        raise ValueError('optimizer curriculum proposals exhausted: ' + str(state['diagnostic']))
    return state['selected']
