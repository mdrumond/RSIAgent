"""Typed Pi tools for project review; the verifier host still admits decisions."""
from contextlib import contextmanager
import time

from coding.pi.schema import STRING, INTEGER, obj
from coding.pi.session import PiCodingSession
from .project_verifier import REVIEW_INSTRUCTIONS, EVIDENCE_INSTRUCTIONS

SYSTEM = REVIEW_INSTRUCTIONS + '''Use the declared typed tools, never print JSON action envelopes.
Choose one host action per turn. If multiple tools are emitted, only the first
queued action is selected; extras are rejected without execution.
The queued receipt is not execution: the next
host snapshot contains the authoritative observation. Use submit_review for a
terminal accept/repair proposal; the host validates it before admission. Its
fields follow the evidence instructions below. Use review_state.missing_required_artifacts
for the exact artifact roles and SHA256 values still required before acceptance.
Evidence arrays must contain only bare SHA256 strings copied from
review_state.allowed_evidence. Put interpretation in reason, never in evidence.
Memory citations must be exact objects from review_state.citations; use [] when
that list is empty. Do not invent source-line citations. Snippet line numbers
are 1-based; omit snippet text to let the host extract the exact observed lines.
Use read_context to page context records if previews are insufficient.
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

    @contextmanager
    def create(*, max_seconds):
        from benchmarks.a5kernels.phase1_live import InfrastructureFailure
        from coding.pi.context import ContextLimitError
        from coding.pi.rpc import PiCancelled, PiDeadlineError, PiError
        try:
            # CLI/env-file credentials remain private to the provider and host
            # gateway. Never replace them with a possibly different ambient key.
            environment = provider.coding_environment() if hasattr(provider, 'coding_environment') else None
            with PiCodingSession(model=complete.model, schemas=SCHEMAS, environ=environment,
                    deadline=time.monotonic() + max_seconds, system_prompt=SYSTEM,
                    first_action_only=True) as session:
                yield session
        except (PiCancelled, PiDeadlineError) as error:
            raise TimeoutError('Pi project review interrupted: ' + type(error).__name__) from None
        except (PiError, ContextLimitError) as error:
            raise InfrastructureFailure('Pi project review infrastructure failure: ' + str(error)) from None
    return create
