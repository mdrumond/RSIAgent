"""Offline adaptive trace capture and conservative annex export.

Historical provider summaries are evidence, never reconstructed conversations.
"""
from pathlib import Path
import hashlib
import json
import tempfile

from benchmarks.a5kernels.sft_dataset import (
    TraceRecorder, _canonical, _example, _read_trace, build_dataset,
)
from benchmarks.ascend.training_annex import inventory_campaign


def recorder_factory(*, phase, round_id, model, source_manifest_sha256,
                     operator_family="adaptive", tools=()):
    """Return a recorder bound to each request's isolated branch root."""
    def create(request):
        identity = hashlib.sha256(_canonical({
            "project": request.spec.as_dict(), "ordinal": request.authored_ordinal,
            "memory_sha256": hashlib.sha256(request.pre_wave_memory).hexdigest(),
            "phase": phase, "round": round_id,
        })).hexdigest()
        trace_id = "adaptive-" + identity
        return TraceRecorder(
            Path(request.branch_root) / "sft-traces" / f"{trace_id}.jsonl",
            trace_id=trace_id, phase=phase, round_id=round_id,
            operator_family=operator_family, model=model,
            source_manifest_sha256=source_manifest_sha256, tools=tools,
        )
    return create


def export_annex(root, output, *, artifact_store=None):
    """Export only sealed exact conversations bound to retained source/evidence.

    Failed attempts stay in the annex and are reported as rejected labels.
    Existing provider ledgers without a sealed trace are explicitly reported.
    The campaign is read-only; export destinations must be outside it.
    """
    root, output = Path(root).resolve(strict=True), Path(output).resolve()
    if output.is_relative_to(root):
        raise ValueError("export destination must be outside campaign")
    inventory = inventory_campaign(root, artifact_store=artifact_store)
    retained = {entry["sha256"] for entry in inventory["files"]}
    rejected, accepted, traced_projects = [], [], set()
    with tempfile.TemporaryDirectory(prefix="ascend-sft-") as staging:
        for path in sorted(root.rglob("*.jsonl")):
            if path.parent.name not in {"sft-traces", "traces"}:
                continue
            relative = path.relative_to(root).as_posix()
            try:
                rows = _read_trace(path)
                traced_projects.add(path.parent.parent)
                candidate = _example(rows)
                if candidate is None:
                    raise ValueError("no terminal verified source")
                metadata = candidate[1]
                if metadata["candidate_sha256"] not in retained:
                    raise ValueError("exact candidate bytes absent from annex")
                if metadata["terminal_evidence_sha256"] not in retained:
                    raise ValueError("exact terminal evidence absent from annex")
                data = path.read_bytes()
                trace_digest = hashlib.sha256(data).hexdigest()
                if trace_digest not in retained:
                    raise ValueError("trace changed since inventory")
                (Path(staging) / f"{trace_digest}.jsonl").write_bytes(data)
                accepted.append({"path": relative, "sha256": trace_digest,
                                 "round_id": metadata["round_id"],
                                 "operator_family": metadata["operator_family"]})
            except (ValueError, KeyError, TypeError) as exc:
                rejected.append({"path": relative, "reason": str(exc)})
        for ledger in sorted(root.rglob("evidence.jsonl")):
            if ledger.parent not in traced_projects:
                rejected.append({"path": ledger.relative_to(root).as_posix(),
                                 "reason": "no complete sealed conversation/source/evidence binding"})
        if inventory_campaign(root, artifact_store=artifact_store) != inventory:
            raise ValueError("campaign changed during export")
        dataset = build_dataset(Path(staging), output)
    manifest = {"schema": "ascend-annex-sft-export-v1",
                "inventory_sha256": inventory["inventory_sha256"],
                "annex_issues": inventory["issues"], "accepted": accepted,
                "rejected": rejected, "dataset": dataset}
    (output / "annex-export.json").write_bytes(_canonical(manifest) + b"\n")
    return manifest
