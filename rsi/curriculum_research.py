"""Typed host research contracts; accepted originals remain immutable."""
from dataclasses import dataclass


@dataclass(frozen=True)
class DiagnosticProposal:
    name: str
    goal: str
    acceptance: str
    template_id: str
    parameters: dict
    parent_project_id: str | None = None


@dataclass(frozen=True)
class ResearchResult:
    proposal: DiagnosticProposal | None
    evidence_refs: tuple[str, ...]
    actor_seconds: float
    receipt_ref: str


@dataclass(frozen=True)
class ResearchQuery:
    query: str
    purpose: str


@dataclass(frozen=True)
class QueryPlan:
    rationale: str
    evidence_ids: tuple[str, ...]
    queries: tuple[ResearchQuery, ...]


@dataclass(frozen=True)
class NegativeDecision:
    query_ref: str
    index: int
    action: str
    reason: str
    redirection: str | None = None


@dataclass(frozen=True)
class DiagnosticDecision:
    rationale: str
    evidence_ids: tuple[str, ...]
    proposal: DiagnosticProposal
    negative_decisions: tuple[NegativeDecision, ...]


def parse_proposal(role, data, admitted):
    """Validate typed NOOA JSON without importing its optional framework."""
    from core.artifacts import canonical_bytes
    keys = {'rationale', 'evidence_ids', 'queries'} if role == 'research_queries' else {
        'rationale', 'evidence_ids', 'proposal', 'negative_decisions'}
    if type(data) is not dict or set(data) != keys:
        raise ValueError('invalid research proposal fields')
    def text(value):
        if not isinstance(value, str) or not value.strip():
            raise ValueError('research fields must be nonempty strings')
        return value
    rationale = text(data['rationale'])
    ids = data['evidence_ids']
    if type(ids) is not list or any(type(ref) is not str or ref not in admitted for ref in ids):
        raise ValueError('research cites unadmitted evidence')
    if role == 'research_queries':
        rows = data['queries']
        if type(rows) is not list or not rows:
            raise ValueError('documentation research requires actual queries')
        queries = []
        for row in rows:
            if type(row) is not dict or set(row) != {'query', 'purpose'}:
                raise ValueError('invalid research query')
            queries.append(ResearchQuery(text(row['query']), text(row['purpose'])))
        return QueryPlan(rationale, tuple(ids), tuple(queries))
    row = data['proposal']
    required = {'name', 'goal', 'acceptance', 'template_id', 'parameters'}
    if type(row) is not dict or not required <= row.keys() or row.keys() - required - {'parent_project_id'}:
        raise ValueError('invalid diagnostic project')
    for key in required - {'parameters'}:
        text(row[key])
    if type(row['parameters']) is not dict:
        raise ValueError('diagnostic parameters must be a JSON object')
    canonical_bytes(row['parameters'])
    if row.get('parent_project_id') is not None:
        text(row['parent_project_id'])
    proposal = DiagnosticProposal(**row)
    decisions = []
    if type(data['negative_decisions']) is not list:
        raise ValueError('negative decisions must be an array')
    for index, row in enumerate(data['negative_decisions']):
        required = {'query_ref', 'index', 'action', 'reason'}
        if (type(row) is not dict or not required <= row.keys() or row.keys() - required - {'redirection'}
                or type(row['index']) is not int or row['index'] < 0
                or row['action'] not in ('used', 'rejected')):
            raise ValueError('invalid negative decision')
        text(row['query_ref']); text(row['reason'])
        if row['action'] == 'used' and (not isinstance(row.get('redirection'), str)
                                      or not row['redirection'].strip()):
            raise ValueError(f'negative_decisions[{index}].redirection must be nonempty when action is used')
        decisions.append(NegativeDecision(**row))
    return DiagnosticDecision(rationale, tuple(ids), proposal, tuple(decisions))


class ResearchPending(RuntimeError):
    def __init__(self, receipt_ref, actor_seconds=None):
        super().__init__('research requires observation of the retained operation')
        self.receipt_ref, self.actor_seconds = receipt_ref, actor_seconds


class ResearchRejected(ResearchPending):
    """Known closed research; a fresh decision may use this retained feedback."""


