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
