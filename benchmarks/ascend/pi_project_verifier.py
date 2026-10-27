"""Typed Pi tools for project review; the verifier host still admits decisions."""
import time

from coding.pi.schema import STRING, INTEGER, obj
from coding.pi.session import PiCodingSession
from .project_verifier import REVIEW_INSTRUCTIONS, EVIDENCE_INSTRUCTIONS

SYSTEM = REVIEW_INSTRUCTIONS + '''Use the declared typed tools, never print JSON action envelopes.
Choose one host action per turn. The queued receipt is not execution: the next
host snapshot contains the authoritative observation. Use submit_review for a
terminal accept/repair proposal; the host validates it before admission. Its
fields follow the evidence instructions below. Use review_state.missing_required_artifacts
for the exact artifact roles and SHA256 values still required before acceptance.
Use read_context to page context records if previews are insufficient.
''' + EVIDENCE_INSTRUCTIONS

def nullable(schema):
    return {'anyOf': [schema, {'type': 'null'}]}


SNIPPET = obj({'sha256': STRING, 'start_line': INTEGER, 'end_line': INTEGER, 'text': STRING},
              ['sha256', 'start_line', 'end_line'])
MEMORY = obj({
    'operation': {'type': 'string', 'enum': ['create', 'append', 'merge']},
    'base_versions': {'type': 'array', 'items': obj({'memory_id': STRING, 'version': STRING})},
    **{key: STRING for key in ('summary', 'applicability', 'preconditions', 'expectations', 'limitations')},
    'before': nullable(SNIPPET), 'before_unavailable_reason': nullable(STRING),
    'after': SNIPPET, 'evidence': {'type': 'array', 'items': STRING},
    'citations': {'type': 'array', 'items': {'type': 'object'}},
})

SCHEMAS = {
    'list_artifacts': obj({'offset': INTEGER, 'limit': INTEGER, 'prefix': STRING}, []),
    'read_artifact': obj({'sha256': STRING, 'offset': INTEGER, 'limit': INTEGER}, ['sha256']),
    'knowledge_query': obj({'query': STRING}),
    'request_check': obj({'hypothesis': STRING, 'check': {'type': 'object'}}),
    # Nested memory snippets and nullable repair/memory fields are validated by
    # the existing host contract, after original Pi arguments are validated.
    'submit_review': obj({'decision': {'type': 'string', 'enum': ['accept', 'repair']},
        'reason': STRING, 'evidence': {'type': 'array', 'items': STRING},
        'repair': nullable(obj({'instructions': STRING, 'expected_evidence': STRING})),
        'memory': nullable(MEMORY)}),
}


def session_factory(settings, provider, complete):
    """Real production providers use Pi; injected/scripted transports stay injectable."""
    from benchmarks.a5kernels.phase1_provider import openai_compatible_transport
    engine = settings.get('verifier_engine')
    if engine is None:
        engine = ('pi' if getattr(provider, '_transport', None) is openai_compatible_transport
                  else 'completion')
    if engine not in {'pi', 'completion'}:
        raise ValueError('verifier_engine must be pi or completion')
    if engine == 'completion':
        return None

    def create(*, max_seconds):
        return PiCodingSession(model=complete.model, schemas=SCHEMAS,
            deadline=time.monotonic() + max_seconds, system_prompt=SYSTEM)
    return create
