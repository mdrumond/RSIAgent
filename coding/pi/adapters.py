"""Mechanical adapters from typed Pi tools to existing host action contracts."""
import json

from .schema import STRING, INTEGER, obj
from .session import PiCodingSession, ENGINE_IDENTITY, ActionError

ARRAY = {"type": "array", "items": STRING}
A3_SCHEMAS = {
    "write_source": obj({"source": STRING}), "compile": obj(), "run": obj(),
    "query": obj({"query": STRING, "limit": {"type": "integer", "minimum": 1, "maximum": 5}}),
    "profile": obj({"metric": STRING}),
    "submit": obj({"interpretation": STRING, "supports": ARRAY}),
}
A5_SCHEMAS = {
    **{name: obj() for name in ("compile", "run", "submit", "request-profile", "request-timing", "request-debug")},
    "write": obj({"slot": {"type": "string", "enum": ["kernel"]}, "content": STRING}),
    "propose": obj({"text": STRING}),
    "query-knowledge": obj({"query": STRING, "limit": {"type": "integer", "minimum": 1, "maximum": 20}}, ["query"]),
    "read-artifact": obj({"sha256": STRING, "offset": INTEGER,
                          "limit": {"type": "integer", "minimum": 1, "maximum": 8192}}, ["sha256"]),
    "request-sanitizer": obj({"tool": {"type": "string", "enum": ["memcheck", "racecheck", "initcheck", "synccheck"]}}),
}
SOURCE_SCHEMAS = {
    "candidate_source": obj({"source": STRING}),
    "compile_probe": obj({"source": STRING}),
    "knowledge_query": A5_SCHEMAS["query-knowledge"],
}
OPTIMIZER_SCHEMAS = {
    **{name: obj({"source": STRING, "block_count": {"type": "integer", "minimum": 1, "maximum": 65535},
                 "summary": STRING}) for name in ("candidate", "compile")},
    "cases": obj({"signatures": {"type": "array", "items": {}}, "summary": STRING}),
    "query-knowledge": obj({"query": STRING}),
    "read-memory": obj({"index": INTEGER}),
    "read-artifact": obj({"sha256": STRING, "offset": INTEGER}, ["sha256"]),
    "finish": obj({"summary": STRING}),
}


class A3Actor:
    def __init__(self, model, environ, *, session_factory=PiCodingSession):
        self.session = session_factory(model=model, schemas=A3_SCHEMAS, environ=environ)
        self.record = lambda *_: None

    def bind_host(self, record, budgets):
        self.record = record
        self.session.gateway.max_calls = budgets.max_turns + 4
        self.session.gateway.completion_limit = budgets.max_tokens

    def __call__(self, profile, context):
        from benchmarks.a3_model_profiles import A3Completion
        state = json.loads(context)
        before = self.session.completion_tokens
        source = state.get("current_candidate_source")
        allowed = list(state["allowed_actions"])
        try:
            name, args = self.session.next_action(state, allowed, source=source, record=self.record,
                write_action=(lambda s: ("write_source", {"source": s})) if "write_source" in allowed else None)
            text = json.dumps({"action": name, **args})
        except ActionError:
            text = "{}"  # Existing host records/charges invalid actions.
        return A3Completion(text, self.session.completion_tokens-before,
                            {"profile_sha256": profile.fingerprint, **ENGINE_IDENTITY})

    def close(self):
        self.session.close()
