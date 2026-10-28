"""Evidence-preserving proposals; publication still belongs to memory admission."""
from copy import deepcopy
import hashlib
import json
import math
import re
import time
from pathlib import Path

from optimizer.jobs import canonical, atomic
from .subject_memory import digest

FIELDS = ('summary', 'applicability', 'preconditions', 'expectations', 'limitations')
RELATIONS = {'equivalent', 'variants', 'conflict', 'unrelated'}


def _words(row):
    return set(re.findall(r'\w+', ' '.join(str(row['proposal'][f]) for f in FIELDS).casefold()))


def discover(anchor, candidates):
    """Coarse lexical retrieval only; semantic compatibility requires review."""
    rows = {row['id']: row for row in [anchor, *candidates]}
    tokens = {key: _words(row) for key, row in rows.items()}
    counts = {word: sum(word in words for words in tokens.values())
              for word in set().union(*tokens.values())}
    # Discount shared boilerplate rather than treating "Ascend memory" as a match.
    weight = {word: math.log(1 + len(rows) / count) for word, count in counts.items()
              if len(rows) < 5 or count < .8 * len(rows)}
    words = tokens[anchor['id']] & weight.keys()
    scored = []
    for key, row in rows.items():
        if key == anchor['id']:
            continue
        other = tokens[key] & weight.keys()
        overlap = sum(weight[word] for word in words & other)
        total = sum(weight[word] for word in words | other)
        if overlap:
            scored.append((overlap / total, row))
    return [row for score, row in sorted(scored, key=lambda item: (-item[0], item[1]['id']))]



def _unique_json(items):
    """Preserve first-seen JSON values, including structured KDB citations."""
    seen, result = set(), []
    for item in items:
        identity = canonical(item)
        if identity not in seen:
            seen.add(identity)
            result.append(deepcopy(item))
    return result


def _artifacts(row):
    proposal = row['proposal']
    return set(proposal['evidence']) | {proposal[role]['sha256'] for role in ('before', 'after')
                                       if proposal.get(role)}


class MemoryConsolidator:
    """A provider judges relations; assembly cannot introduce new lesson wording.

    Draft rows are {id, proposal, provenance}. Accepted rows additionally carry
    base={memory_id, version}. No draft is published or removed by this class.
    """
    def __init__(self, complete, *, clock=time.monotonic, max_peers=3, cache_root=None):
        if type(max_peers) is not int or max_peers < 1:
            raise ValueError("max_peers must be positive")
        self.complete, self.clock, self.last_trace = complete, clock, []
        self.max_peers = max_peers
        self.cache_root = Path(cache_root) if cache_root is not None else None
        if self.cache_root is not None:
            self.cache_root.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _preview(row):
        # Producer manifests can contain thousands of unrelated artifacts. Only
        # actual lesson/source evidence is admitted to this relation review.
        return {key: row[key] for key in ('id', 'proposal', 'base') if key in row}

    def compare(self, left, right, read_artifact, *, remaining):
        identity = digest(dict(schema="memory-relation-v1", left=left, right=right))
        cache = self.cache_root / (identity + ".json") if self.cache_root is not None else None
        if cache is not None and cache.exists():
            return json.loads(cache.read_bytes())
        known = _artifacts(left) | _artifacts(right)
        read, artifact_pages = set(), {}
        messages = [dict(role='system', content=(
            'Compare two concrete memory lessons. Read each supplied evidence and source artifact '
            'before deciding. Classify equivalent (same lesson and conditions), variants (same '
            'lesson with compatible explicit applicability variants), conflict (contradictory '
            'advice in overlapping conditions), or unrelated. Preserve type, runtime, hardware, '
            'shape and performance limits. Similar vocabulary is not evidence of equivalence. '
            'Return {"read_artifact":{"sha256":"...","offset":0,"limit":8192}} or '
            '{"relation":"equivalent|variants|conflict|unrelated","reason":"...",'
            '"evidence":["read hashes"]}. No rewritten lesson or generalization.')),
            dict(role='user', content=json.dumps(dict(left=self._preview(left), right=self._preview(right),
                 required_artifacts=sorted(known)), sort_keys=True))]
        while True:
            raw = self.complete(tuple(messages), remaining_seconds=remaining())
            remaining()
            self.last_trace.append(dict(response=raw))
            messages.append(dict(role='assistant', content=raw))
            try:
                value = json.loads(raw)
                if set(value) == {'read_artifact'}:
                    request = value['read_artifact']
                    sha = request['sha256']
                    if sha not in known:
                        raise ValueError('artifact is outside the compared lessons')
                    data = read_artifact(sha, remaining_seconds=remaining())
                    if not isinstance(data, bytes) or hashlib.sha256(data).hexdigest() != sha:
                        raise ValueError('artifact bytes do not match identity')
                    offset, limit = request.get('offset', 0), request.get('limit', 8192)
                    if type(offset) is not int or not 0 <= offset <= len(data) or type(limit) is not int or not 1 <= limit <= 8192:
                        raise ValueError('invalid artifact page')
                    # Every source/evidence page must be inspected, not just its first byte.
                    pages = artifact_pages.setdefault(sha, [])
                    pages.append((offset, min(len(data), offset + limit)))
                    end = 0
                    for start, stop in sorted(pages):
                        if start > end:
                            break
                        end = max(end, stop)
                    if end == len(data):
                        read.add(sha)
                    observation = dict(sha256=sha, size=len(data), offset=offset,
                        text=data[offset:offset+limit].decode('utf-8', errors='replace'),
                        next_offset=offset+limit if offset+limit < len(data) else None)
                else:
                    if set(value) != {'relation', 'reason', 'evidence'} or value['relation'] not in RELATIONS:
                        raise ValueError('invalid relation decision')
                    if not isinstance(value['reason'], str) or not value['reason'].strip():
                        raise ValueError('relation requires a reason')
                    if known - read:
                        raise ValueError('read all source/evidence pages before deciding')
                    if not isinstance(value['evidence'], list) or not value['evidence'] or any(sha not in read for sha in value['evidence']):
                        raise ValueError('cite inspected evidence')
                    if cache is not None:
                        atomic(cache, canonical(value))
                    return value
            except (ValueError, KeyError, TypeError) as error:
                observation = dict(error=str(error))
            messages.append(dict(role='user', content=json.dumps(observation, sort_keys=True)))

    def prepare(self, anchor, candidates, *, memory, read_artifact, max_seconds,
                check_remaining=None):
        deadline = self.clock() + max_seconds
        self.last_trace = []
        def remaining():
            seconds = deadline - self.clock()
            if check_remaining is not None:
                seconds = min(seconds, check_remaining())
            if seconds <= 0:
                raise TimeoutError('memory consolidation elapsed allocation exhausted')
            return seconds
        remaining()
        accepted = [dict(id=record['memory_id'], proposal=record['proposal'],
                    provenance=record['provenance'], base=dict(memory_id=record['memory_id'],
                    version=digest(record))) for record in memory.records()]
        rows = {row['id']: row for row in candidates}
        # Current accepted identity takes precedence over an older draft copy.
        rows.update({row['id']: row for row in accepted})
        anchor = rows.get(anchor['id'], anchor)
        selected, comparisons, incomplete = [anchor], [], False
        retrieved = discover(anchor, rows.values())
        for peer in retrieved[:self.max_peers]:
            compatible = True
            for member in selected:
                try:
                    remaining()
                    result = self.compare(member, peer, read_artifact, remaining=remaining)
                except TimeoutError:
                    incomplete, compatible = True, False
                    break
                comparisons.append(dict(left=member['id'], right=peer['id'], **result))
                if result['relation'] not in {'equivalent', 'variants'}:
                    compatible = False
                    break
            if compatible:
                selected.append(peer)
            if incomplete:
                break
        discovery = dict(candidates=[row["id"] for row in retrieved], reviewed_limit=self.max_peers,
                         incomplete=incomplete)
        if len(selected) == 1:
            return dict(proposal=None, members=[anchor['id']], comparisons=comparisons, discovery=discovery)
        proposal = deepcopy(anchor['proposal'])
        bases = {}
        for row in selected:
            # Accepted proposals retain historical consumed bases as lineage.
            # Replacement targets the current accepted version exclusively.
            replacements = [row['base']] if 'base' in row else row['proposal']['base_versions']
            for base in replacements:
                if base['memory_id'] in bases and bases[base['memory_id']] != base:
                    raise ValueError('inconsistent replacement bases')
                bases[base['memory_id']] = base
        proposal.update(operation='merge' if bases else 'create', base_versions=list(bases.values()))
        for field in FIELDS:
            proposal[field] = '\n\n'.join(f"[{row['id']}]\n{row['proposal'][field]}" for row in selected)
        for field in ('evidence', 'citations'):
            proposal[field] = _unique_json(item for row in selected for item in row['proposal'][field])
        proposal['evidence'] = _unique_json([*proposal['evidence'],
            *(sha for row in selected for sha in sorted(_artifacts(row)))])
        # Before/after remain the primary runnable example. All other examples,
        # their exact claims and source identities remain durable provenance.
        lineage = [deepcopy(row) for row in selected]
        provenance = deepcopy(anchor['provenance'])
        provenance['consolidation'] = dict(schema='memory-consolidation-v1', members=lineage,
            comparisons=comparisons, claims=[dict(member=row['id'], fields={f: row['proposal'][f]
                for f in FIELDS}, artifacts=sorted(_artifacts(row))) for row in selected])
        return dict(proposal=proposal, provenance=provenance,
                    members=[row['id'] for row in selected], comparisons=comparisons, discovery=discovery)


def submit_consolidation(admission, prepared, request, *, remaining_seconds):
    """Stage accepted-member consolidation for normal admission.

    Draft groups are useful review proposals, but each member must first pass
    admission independently. A primary example cannot validate other variants.
    """
    if prepared['proposal'] is None:
        return None
    members = prepared['provenance']['consolidation']['members']
    current = {digest(record): record for record in admission.memory.records()}
    for member in members:
        base = member.get('base', {})
        record = current.get(base.get('version'))
        if (record is None or record['memory_id'] != base.get('memory_id')
                or record['proposal'] != member['proposal']
                or record['provenance'] != member['provenance']):
            raise ValueError('admit every member independently before submitting consolidation')
    identity = 'consolidation-' + digest(dict(proposal=prepared['proposal'], members=prepared['members']))
    admission.submit(identity, prepared['proposal'], prepared['provenance'], request,
                     remaining_seconds=remaining_seconds)
    return identity
