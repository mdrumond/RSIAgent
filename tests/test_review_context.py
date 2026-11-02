"""Retained review input fixture; native SDK review tests own context behavior."""
import hashlib

def fixture():
    digests = [hashlib.sha256(str(i).encode()).hexdigest() for i in range(12)]
    details = {f'case_{i}': {'expectation': 'Observed é correctness for case ' + str(i),
               'before': 'short source line ' * 12, 'limitations': 'shape-specific evidence ' * 8}
               for i in range(35)}
    controls = dict(host_checks_passed=True, knowledge_queried=True,
        allowed_evidence=digests[:8], read_artifacts=digests[:8],
        missing_required_artifacts=[{'sha256': h, 'roles': [f'host-result:{i}']}
                                    for i, h in enumerate(digests[8:])], citations=[])
    return dict(host=dict(project=dict(project=dict(project_id='retained-copy',
        source_sha256=digests[0], host_verified=True, artifacts={'source': digests[0]}),
        rewrite_feedback={'prior_draft': details, 'task': 'Audit and retain only supported lessons',
            'known_counterevidence': {'input': '2^-40', 'epsilon': '2^-32', 'formula_result': '1/256'}}, tools={'knowledge_query': True}),
        review_state=controls, artifact_pages=[], validation_feedback=None,
        tool_observations={'knowledge_query': {'hits': [], 'citations': []}}),
        remaining_seconds=60, source_sha256=None)

