"""Host validation for evidence-driven memory judgments and repairs."""
from __future__ import annotations

from .subject_memory import digest


def content_identity(proposal):
    values = {key:proposal.get(key) for key in ('summary','applicability','preconditions','expectations','limitations','before_unavailable_reason')}
    values.update({key:None if proposal.get(key) is None else {field:proposal[key][field] for field in ('sha256','start_line','end_line')} for key in ('before','after')})
    return digest(values)


def validate_memory_review(value, request, read, evidence, fully_read):
    """Memory claims require explicit judgments, exact versions and read evidence."""
    fields = ('memory_judgments', 'memory_resolutions', 'retirement_checks')
    if any(not isinstance(value.get(key), list) for key in fields):
        raise ValueError('memory_judgments, memory_resolutions and retirement_checks must be arrays')
    exposures = {row['exposure_id']: row for row in request.memory_exposures}
    judgments = value['memory_judgments']
    if {row.get('exposure_id') for row in judgments} != set(exposures) or len(judgments) != len(exposures):
        raise ValueError('judge every accessed actor/project/memory-version exposure exactly once')
    memory_bound = {sha for group in (request.memory_exposures, request.memory_repairs, request.retired_memories)
                    for row in group for sha in [row['document_sha256'], *row.get('artifact_sha256', [])]}
    project_evidence = set(request.artifacts.values()) - memory_bound
    for row in judgments:
        if set(row) != {'exposure_id', 'usefulness', 'correctness', 'reason', 'evidence', 'investigate'}:
            raise ValueError('invalid memory judgment fields')
        if row['usefulness'] not in {'useful', 'not-useful', 'unknown'} or row['correctness'] not in {'supported', 'contradicted', 'unknown'}:
            raise ValueError('invalid memory usefulness/correctness judgment')
        if not isinstance(row['reason'], str) or not row['reason'].strip():
            raise ValueError('memory judgment needs a reason')
        if row['evidence']:
            evidence(row['evidence'], read)
        elif row['usefulness'] != 'unknown' or row['correctness'] != 'unknown':
            raise ValueError('decided memory judgments require evidence')
        if type(row['investigate']) is not bool or row['investigate'] and (row['correctness'] != 'unknown' or not row['evidence']):
            raise ValueError('investigation requires an explicit uncertain concern with evidence')
        memory = exposures[row['exposure_id']]
        if row['correctness'] != 'unknown' and memory['document_sha256'] not in fully_read:
            raise ValueError('read the complete exact accessed memory before deciding correctness')
        if (row['correctness'] != 'unknown' or row['usefulness'] != 'unknown') and not project_evidence.intersection(row['evidence']):
            raise ValueError('decided memory judgment requires independent project/source evidence')
    repairs = {row['repair_id']: row for row in request.memory_repairs}
    resolutions = value['memory_resolutions']
    if len({row.get('repair_id') for row in resolutions}) != len(resolutions):
        raise ValueError('memory repairs must be resolved once')
    if any(row.get('repair_id') not in repairs for row in resolutions):
        raise ValueError('memory resolution names an unassigned repair')
    for row in resolutions:
        if set(row) != {'repair_id', 'action', 'reason', 'evidence'} or row['action'] not in {'replace', 'restore', 'retire', 'confirm'}:
            raise ValueError('invalid memory repair resolution')
        evidence(row['evidence'], read)
        repair = repairs[row['repair_id']]
        if not set(repair.get('evidence', [])) <= read:
            raise ValueError('read the recorded counterevidence before resolving a memory repair')
        if repair['document_sha256'] not in fully_read:
            raise ValueError('read the disputed memory before resolving its repair')
        if not row['reason'].strip() or not project_evidence.intersection(row['evidence']):
            raise ValueError('memory resolution requires independent evidence and a reason')
        base = {'memory_id': repair['memory_id'], 'version': repair['version']}
        if row['action'] in {'confirm', 'restore'} and value['memory'] is not None and base in value['memory']['base_versions']:
            raise ValueError('confirm/restore cannot also supersede the repaired version; use replace')
        if row['action'] == 'replace':
            memory = value['memory']
            base = {'memory_id': repair['memory_id'], 'version': repair['version']}
            if memory is None or base not in memory['base_versions']:
                raise ValueError('a replacement must explicitly replace the repaired memory version')
            if repair.get('content_identity') == content_identity(memory):
                raise ValueError('a replacement must change consumable memory content, not merely reset counters')
        if row['action'] == 'confirm' and (repair.get('kind') != 'investigate' or repair.get('status') != 'active'):
            raise ValueError('only an active investigation can be confirmed without changing memory')
    if value['decision'] == 'accept' and {row['repair_id'] for row in resolutions} != set(repairs):
        raise ValueError('a passing project must explicitly resolve every assigned memory repair')
    if value['decision'] == 'repair' and resolutions:
        raise ValueError('an incomplete project cannot resolve memory repairs')
    retired = {row['tombstone_id']: row for row in request.retired_memories}
    checks = value['retirement_checks']
    if len({row.get('tombstone_id') for row in checks}) != len(checks):
        raise ValueError('retirement checks must identify distinct records')
    if value['memory'] is not None and {row.get('tombstone_id') for row in checks} != set(retired):
        raise ValueError('compare proposed memory with every supplied retirement record')
    def claim(proposal):
        return tuple(' '.join(str(proposal.get(key, '')).casefold().split()) for key in
                     ('summary', 'applicability', 'preconditions', 'expectations', 'limitations'))
    for row in checks:
        if set(row) != {'tombstone_id', 'relation', 'reason', 'evidence'} or row['relation'] not in {'unrelated', 'duplicate', 'reconciles'}:
            raise ValueError('invalid retirement check')
        old = retired.get(row['tombstone_id'])
        if old is None or not row['reason'].strip():
            raise ValueError('retirement check needs a supplied record and reason')
        if old['document_sha256'] not in fully_read:
            raise ValueError('read retirement evidence before comparing the proposed claim')
        if value['memory'] is not None and claim(value['memory']) == tuple(old.get('claim_identity', ())):
            if row['relation'] != 'reconciles':
                raise ValueError('exact retired claim requires explicit resolving evidence')
        if row['relation'] == 'duplicate' and value['memory'] is not None:
            raise ValueError('retired claims cannot be republished without new resolving evidence')
        if row['evidence']:
            evidence(row['evidence'], read)
        if row['relation'] == 'reconciles':
            if not set(old['evidence']) <= read:
                raise ValueError('read the recorded retirement evidence before restoring its claim')
            if not (set(row['evidence']) & project_evidence) - set(old['evidence']):
                raise ValueError('restoring a retired claim requires fresh resolving project evidence')
