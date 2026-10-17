"""Explicit, content-addressed admission of the new profiling treatment."""

from dataclasses import asdict, dataclass
import hashlib
from pathlib import Path
import re


@dataclass(frozen=True)
class ProfilingGuidanceIdentity:
    schema: str
    guidance_sha256: str
    cpl_skills_revision: str

    def __post_init__(self):
        if self.schema != "a5-new-profiling-guidance-v1":
            raise ValueError("unsupported profiling guidance schema")
        if re.fullmatch(r"[0-9a-f]{64}", self.guidance_sha256) is None:
            raise ValueError("profiling guidance requires a SHA-256 digest")
        if re.fullmatch(r"[0-9a-f]{40}", self.cpl_skills_revision) is None:
            raise ValueError("profiling guidance requires a full CPL revision")

    def as_dict(self):
        return asdict(self)


def verify_profiling_guidance(identity: ProfilingGuidanceIdentity, text: str) -> None:
    if not isinstance(identity, ProfilingGuidanceIdentity) or not isinstance(text, str):
        raise ValueError("new profiling requires admitted identity and guidance text")
    if not text.strip() or hashlib.sha256(text.encode("utf-8")).hexdigest() != identity.guidance_sha256:
        raise ValueError("profiling guidance content does not match its identity")


def admit_profiling_guidance(skill_root: str | Path, *, cpl_skills_revision: str):
    """Freeze the installed skill and required references, without local paths."""
    parts = [
        "RSI action mapping: first write, compile, and run the candidate to obtain "
        'HOST_VERIFIED current source. Use {"request-timing":{}} for an '
        'ordinary timing sweep and {"request-profile":{}} for the project\'s '
        "profile_metric. These are harness actions, not shell tools. The harness "
        "captures BasicInfo first and discovers the exact kernel selector. Choose "
        "timing_lengths/profile_metric in the project specification. The ONLY "
        "supported profile_metric values are pipe-utilization (PipeUtilization), "
        "memory-throughput (Memory), and timeline (PipeTimeline). Other metric "
        "groups and shell commands in this skill are explanatory, not available "
        "Actor actions or ProjectSpec metrics. Keep the "
        "returned raw evidence identities in lessons. Only the checked collector "
        "can execute profiler commands; the shell examples below explain semantics."
    ]
    for name in ("SKILL.md", "references/a5-metric-semantics.md", "references/a5-timeline-semantics.md"):
        text = (Path(skill_root) / name).read_text(encoding="utf-8")
        if not text.strip():
            raise ValueError("profiling guidance source must not be empty")
        parts.append(f"## {name}\n{text}")
    text = "\n\n".join(parts)
    identity = ProfilingGuidanceIdentity(
        "a5-new-profiling-guidance-v1", hashlib.sha256(text.encode("utf-8")).hexdigest(),
        cpl_skills_revision,
    )
    return identity, text
