"""Typed Pi tools for project review; the verifier host still admits decisions."""
from contextlib import contextmanager
import time

from coding.pi.schema import STRING, INTEGER, obj
from .native_review import PiReviewSession
from .project_verifier import REVIEW_INSTRUCTIONS, EVIDENCE_INSTRUCTIONS
from .review_context import ReviewContextRecords

SYSTEM = REVIEW_INSTRUCTIONS + '''Use the declared typed native tools; never print JSON action envelopes.
Read/list/query tools return real host observations immediately and may be called
together. submit_review and request_check must each be the only tool in its
assistant message. Only a host-validated terminal decision ends the review.
Use review_context.host.review_state.missing_required_artifacts for exact roles
and hashes still required; ready_for_decision is guidance, not automatic acceptance.
Evidence arrays contain only bare hashes from allowed_evidence. Citations must
be exact KDB objects from review_state.citations (or [] when none). Snippet lines
start at 1. Put interpretation in reason, never in evidence references.
Read pages contain at most 2048 bytes; use the exact returned next_offset.
Large observations and older complete exchanges have lossless read_context
record pointers; a pointer is not an empty result. Current host controls and
known counterevidence remain authoritative. Legitimate rereads remain available.
''' + EVIDENCE_INSTRUCTIONS

def nullable(schema):
    return {'anyOf': [schema, {'type': 'null'}]}


SHA256 = {'type': 'string', 'pattern': '^[0-9a-f]{64}$',
          'description': 'Exact bare SHA256 artifact identity returned by the host. No annotations or prose.'}
LINE = {'type': 'integer', 'minimum': 1, 'description': '1-based source line number.'}
EVIDENCE = {'type': 'array', 'minItems': 1, 'items': SHA256,
            'description': 'Bare SHA256 identities of artifacts actually read; copy review_state.allowed_evidence.'}
SNIPPET = obj({'sha256': SHA256, 'start_line': LINE, 'end_line': LINE, 'text': STRING},
              ['sha256', 'start_line', 'end_line'])
MEMORY = obj({
    'operation': {'type': 'string', 'enum': ['create', 'append', 'merge']},
    'base_versions': {'type': 'array', 'items': obj({'memory_id': STRING, 'version': STRING})},
    **{key: STRING for key in ('summary', 'applicability', 'preconditions', 'expectations', 'limitations')},
    'before': nullable(SNIPPET), 'before_unavailable_reason': nullable(STRING),
    'after': SNIPPET, 'evidence': EVIDENCE,
    'citations': {'type': 'array', 'items': {'type': 'object'},
                  'description': 'Exact KDB citation objects from review_state.citations, or [] if none were returned.'},
})

SCHEMAS = {
    'list_artifacts': obj({'offset': INTEGER, 'limit': INTEGER, 'prefix': STRING}, []),
    'read_artifact': obj({'sha256': SHA256, 'offset': INTEGER, 'limit': INTEGER}, ['sha256']),
    'knowledge_query': obj({'query': STRING}),
    'request_check': obj({'hypothesis': STRING, 'check': {'type': 'object'}}),
    # Nested memory snippets and nullable repair/memory fields are validated by
    # the existing host contract, after original Pi arguments are validated.
    'submit_review': obj({'decision': {'type': 'string', 'enum': ['accept', 'repair']},
        'reason': STRING, 'evidence': EVIDENCE,
        'repair': nullable(obj({'instructions': STRING, 'expected_evidence': STRING})),
        'memory': nullable(MEMORY),
        'memory_judgments': {'type': 'array', 'items': obj({'exposure_id': STRING,
            'usefulness': {'type':'string','enum':['useful','not-useful','unknown']},
            'correctness': {'type':'string','enum':['supported','contradicted','unknown']},
            'investigate': {'type':'boolean'}, 'reason': STRING, 'evidence': {'type':'array','items':SHA256}})},
        'memory_resolutions': {'type': 'array', 'items': obj({'repair_id': STRING,
            'action': {'type':'string','enum':['replace','restore','retire','confirm']},
            'reason':STRING, 'evidence':EVIDENCE})},
        'retirement_checks': {'type':'array','items':obj({'tombstone_id':STRING,
            'relation':{'type':'string','enum':['unrelated','duplicate','reconciles']},
            'reason':STRING, 'evidence':{'type':'array','items':SHA256}})}}),
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

    @contextmanager
    def create(*, max_seconds):
        from benchmarks.a5kernels.phase1_live import InfrastructureFailure
        from coding.pi.context import ContextLimitError
        from coding.pi.rpc import PiCancelled, PiDeadlineError, PiError
        try:
            # CLI/env-file credentials remain private to the provider and host
            # gateway. Never replace them with a possibly different ambient key.
            environment = provider.coding_environment() if hasattr(provider, 'coding_environment') else None
            with PiReviewSession(model=complete.model, schemas=SCHEMAS, environ=environment,
                    deadline=time.monotonic() + max_seconds, system_prompt=SYSTEM,
                    context_factory=ReviewContextRecords) as session:
                yield session
        except (PiCancelled, PiDeadlineError) as error:
            raise TimeoutError('Pi project review interrupted: ' + type(error).__name__) from None
        except (PiError, ContextLimitError) as error:
            raise InfrastructureFailure('Pi project review infrastructure failure: ' + str(error)) from None
    return create
