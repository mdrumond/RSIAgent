"""Shared paths, prompt rules, and reproducibility metadata."""

from pathlib import Path
import hashlib
import subprocess
import shutil
import json

from .tasks import task_description

ROOT = Path(__file__).resolve().parents[2]
LOCAL = ROOT / ".codex/local/pi-investigation"
RULES = """Work only on kernel.cpp. You have at most three
candidate revisions. Treat host compilation and verification as authoritative.
Stop after a pass, or after an infrastructure-unavailable result. Never claim an
NPU pass without host validation. Use only the supplied task context and host feedback.
"""
TASK = task_description("vector-add") + "\n" + RULES
EFFICIENT_RULES = RULES + """\nSubmit exactly one complete implementation through
submit_candidate. It writes and validates the source automatically. The current
source is already in the prompt. Emit the tool call directly, without progress
commentary or a closing explanation. Keep source comments brief. The host owns
the revision loop and provides the next snapshot if a correction is needed.
"""


DEFAULT_KNOWLEDGE = Path(__file__).parent / "knowledge" / "ascendc.md"
PAIR_LIMITS = {"revisions": 12, "requests": 30, "seconds": 3600, "max_output_tokens": 16384}


def source_metadata():
    """Fingerprint implementation even when the checkout contains uncommitted work."""
    files = sorted(Path(__file__).parent.glob("*.py"))
    files += sorted((ROOT / "integrations/pi").glob("*.ts"))
    files += [ROOT / "integrations/pi/package-lock.json"]
    files += sorted((Path(__file__).parent / "fixtures").glob("*"))
    digest = hashlib.sha256()
    for path in files:
        digest.update(str(path.relative_to(ROOT)).encode() + b"\0" + path.read_bytes())
    revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True)
    return {"git_revision": revision.stdout.strip() if revision.returncode == 0 else None,
            "implementation_sha256": digest.hexdigest()}


def compose_request(request, knowledge):
    if not knowledge:
        return request
    return request + "\n\nStarting knowledge (guidance; the task contract remains authoritative):\n" + knowledge


def runtime_preflight():
    """Fail before a paid pair if the optional, pinned runtime is unavailable."""
    executable = ROOT / "integrations/pi/node_modules/.bin/pi"
    manifest = ROOT / "integrations/pi/node_modules/@earendil-works/pi-coding-agent/package.json"
    if not shutil.which("node") or not executable.is_file() or not manifest.is_file():
        raise ValueError("install Node 22+ and run npm ci --prefix integrations/pi before a paired run")
    if json.loads(manifest.read_text()).get("version") != "1.3.0":
        raise ValueError("Pi must match the pinned 1.3.0 runtime; run npm ci --prefix integrations/pi")
    version = subprocess.run(["node", "--version"], capture_output=True, text=True, check=True)
    if int(version.stdout.strip().lstrip("v").split(".")[0]) < 22:
        raise ValueError("Pi requires Node 22 or newer")
