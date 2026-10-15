"""Archive public, ranked CANNBench source bundles without authentication.

Sources are reference-exposed, not held-out evaluation material. No downloaded
code is executed, and published leaderboard measurements are not our validation.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import re
import threading
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urljoin
from urllib.request import Request, urlopen
import zipfile

BASE = "https://cannbench.com"
CODE = {".py", ".cpp", ".cc", ".c", ".h", ".hpp", ".cuh", ".asc"}
TEXT = CODE | {".md", ".txt", ".json", ".yaml", ".yml", ".sh", ".cmake"}


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def save_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


class PublicArchive:
    """URL-keyed snapshots permit interrupted acquisition to resume."""

    def __init__(self, root: Path, timeout: int = 30):
        self.root = root
        self.timeout = timeout
        self.locks: dict[str, threading.Lock] = {}

    def get(self, url: str) -> tuple[bytes | None, dict]:
        with self.locks.setdefault(url, threading.Lock()):
            return self._get(url)

    def _get(self, url: str) -> tuple[bytes | None, dict]:
        key = digest(url.encode())
        meta = self.root / f"{key}.json"
        body = self.root / f"{key}.body"
        if meta.exists():
            record = json.loads(meta.read_text())
            if record["status"] == 200 and body.exists():
                data = body.read_bytes()
                if digest(data) == record["sha256"]:
                    return data, record
            if record["status"] in (401, 403, 404):
                return None, record
        record = {"url": url, "retrieved_at": datetime.now(timezone.utc).isoformat()}
        try:
            request = Request(url, headers={"User-Agent": "RSI-public-source-archive/1"})
            with urlopen(request, timeout=self.timeout) as response:
                data = response.read()
                record.update(status=response.status, sha256=digest(data),
                              content_type=response.headers.get("Content-Type", ""))
        except HTTPError as error:
            data = None
            record.update(status=error.code, reason=f"HTTP {error.code}")
        except (URLError, TimeoutError, OSError) as error:
            data = None
            record.update(status=0, reason=str(error))
        self.root.mkdir(parents=True, exist_ok=True)
        if data is not None:
            temporary = body.with_suffix(".tmp")
            temporary.write_bytes(data)
            temporary.replace(body)
        save_json(meta, record)
        return data, record

    def json(self, url: str) -> dict:
        data, record = self.get(url)
        if data is None:
            raise RuntimeError(f"Public metadata unavailable: {url}: {record}")
        return json.loads(data)


def candidate_urls(job: dict) -> list[str]:
    """Only follow published pointers; never guess private storage URLs."""
    result = []
    for owner in (job, job.get("submission") or {}):
        for field in ("submission_download_url", "download_url", "source_url",
                      "source_archive_url", "repository_url"):
            value = owner.get(field)
            if isinstance(value, str) and value.startswith(("/", "https://", "http://")):
                result.append(urljoin(BASE, value))
    return list(dict.fromkeys(result))


def unpack(data: bytes, operator: dict, job: dict) -> tuple[dict[str, bytes], str]:
    """Retain text wrappers and licenses; identify this operator's code."""
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        files = {}
        for info in archive.infolist():
            path = PurePosixPath(info.filename)
            if info.is_dir() or path.is_absolute() or ".." in path.parts:
                continue
            if any(part in (".git", "__pycache__", "node_modules") for part in path.parts):
                continue
            if path.suffix.lower() not in TEXT and not path.name.upper().startswith(("LICENSE", "COPYING")):
                continue
            value = archive.read(info)
            try:
                value.decode("utf-8")
            except UnicodeDecodeError:
                continue
            files[str(path)] = value
    code = {name: value for name, value in files.items() if Path(name).suffix.lower() in CODE}
    normalize = lambda value: re.sub(r"[^a-z0-9]", "", value.lower())
    key = normalize(operator["function_name"])
    own = {name: value for name, value in code.items()
           if any(key == normalize(part.split(".")[0]) for part in PurePosixPath(name).parts)}
    selected = job.get("selected_operators") or job.get("operator_names") or []
    if not own and len(selected) == 1 and normalize(selected[0]) == key:
        own = code
    if not own:
        raise ValueError("No code identifiable as the ranked operator")
    # Prefer explicit kernel code over launch/test wrappers for duplicate checks.
    kernels = {name: value for name, value in own.items()
               if any("kernel" in part.lower() for part in PurePosixPath(name).parts)}
    identity = digest(b"\0".join(sorted(set((kernels or own).values()))))
    return files, identity


def acquire_operator(operator: dict, archive: PublicArchive, root: Path,
                     version: str, wanted: int) -> dict:
    name = operator["function_name"]
    offset, candidates, accepted, skipped, identities = 0, [], [], [], set()
    retryable_failures = 0
    while True:
        query = urlencode(dict(benchmark_slug="official-tasks", operator=name,
                               hardware="910c", version=version, limit=100, offset=offset))
        page = archive.json(f"{BASE}/api/leaderboard?{query}")
        rows = page["entries"]
        if not rows and offset < page["total"]:
            raise RuntimeError("Leaderboard ended before its published total")
        for row in rows:
            if row.get("hardware", "910c") != "910c" or row.get("benchmark_version", version) != version:
                raise RuntimeError("Leaderboard response has a different hardware/version")
        candidates.extend(rows)
        offset += len(rows)
        if offset >= page["total"] or not rows:
            break
    for rank, entry in enumerate(candidates, 1):
        if len(accepted) == wanted:
            break
        job_id = entry.get("best_job_id")
        record = {"rank": rank, "entry": entry, "job_id": job_id, "attempts": []}
        if not job_id:
            record["reason"] = "No public job pointer"
            skipped.append(record)
            continue
        try:
            job = archive.json(f"{BASE}/api/jobs/{job_id}")["job"]
        except (RuntimeError, ValueError, KeyError) as error:
            record["reason"] = str(error)
            retryable_failures += 1
            skipped.append(record)
            continue
        for url in candidate_urls(job):
            data, response = archive.get(url)
            record["attempts"].append(response)
            if data is None:
                if response["status"] == 0 or response["status"] >= 500:
                    retryable_failures += 1
                continue
            try:
                files, identity = unpack(data, operator, job)
            except (zipfile.BadZipFile, ValueError, KeyError) as error:
                response["source_error"] = str(error)
                continue
            if identity in identities:
                record["reason"] = "Duplicate operator implementation"
                break
            identities.add(identity)
            destination = root / "sources" / name / f"rank-{rank}-{job_id}"
            file_records = []
            for relative, content in sorted(files.items()):
                path = destination / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(content)
                file_records.append({"path": relative, "sha256": digest(content)})
            source = {"name": f"cannbench-{name}-{job_id}", "repository": url,
                      "revision": response["sha256"], "root": str(destination.resolve()),
                      "files": file_records, "operator": name, "rank": rank,
                      "score": entry.get("score"), "job_id": job_id,
                      "submission_id": job.get("submission_id"),
                      "implementation_sha256": identity, "reference_exposed": True,
                      "license_files": [f["path"] for f in file_records
                                        if Path(f["path"]).name.upper().startswith(("LICENSE", "COPYING"))]}
            save_json(destination / "source-manifest.json", source)
            accepted.append(source)
            break
        else:
            record["reason"] = "No publicly downloadable identifiable source archive"
        if len(accepted) == 0 or accepted[-1].get("rank") != rank:
            skipped.append(record)
    return {"operator": name, "ranked_candidates": len(candidates),
            "examined_candidates": len(accepted) + len(skipped), "sources": accepted,
            "skipped": skipped, "shortfall": max(0, wanted - len(accepted)),
            "retryable_failures": retryable_failures,
            "exhausted": len(accepted) < wanted and retryable_failures == 0}


def acquire(root: Path, *, version: str = "1.1.2", wanted: int = 3,
            workers: int = 6, operators: list[str] | None = None) -> dict:
    archive = PublicArchive(root / "responses")
    catalog = archive.json(f"{BASE}/api/public/catalog?context=eval_page")
    benchmark = next(item for item in catalog["benchmarks"] if item["slug"] == "official-tasks")
    if version not in {item["version"] for item in benchmark["versions"]}:
        raise ValueError("Requested version not in public catalog")
    selected = benchmark["multi_operators"]
    if len(selected) != benchmark["operator_count"]:
        raise ValueError("Public catalog operator list does not match its count")
    if operators:
        selected = [item for item in selected if item["function_name"] in operators]
        missing = set(operators) - {item["function_name"] for item in selected}
        if missing:
            raise ValueError(f"Unknown operators: {sorted(missing)}")
    def run(operator):
        try:
            result = acquire_operator(operator, archive, root, version, wanted)
        except (RuntimeError, ValueError, KeyError) as error:
            result = {"operator": operator["function_name"], "sources": [], "skipped": [],
                      "examined_candidates": 0, "shortfall": wanted, "exhausted": False,
                      "error": str(error)}
        save_json(root / "operators" / f"{operator['function_name']}.json", result)
        print(f"{operator['function_name']}: {len(result['sources'])}/{wanted}; "
              f"examined {result['examined_candidates']}", flush=True)
        return result
    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(run, selected))
    summary = {"schema_version": 1, "benchmark": "official-tasks", "version": version,
               "hardware": "910c", "reference_exposed": True, "public_only": True,
               "catalog_operator_count": benchmark["operator_count"],
               "requested_per_operator": wanted, "operators": results,
               "sources": [source for result in results for source in result["sources"]],
               "shortfall": sum(result["shortfall"] for result in results),
               "retrieved_at": datetime.now(timezone.utc).isoformat()}
    save_json(root / "inventory.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--version", default="1.1.2")
    parser.add_argument("--wanted", type=int, default=3)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--operators", nargs="+")
    args = parser.parse_args()
    if args.wanted < 1 or args.workers < 1:
        parser.error("--wanted and --workers must be positive")
    summary = acquire(args.root, version=args.version, wanted=args.wanted,
                      workers=args.workers, operators=args.operators)
    print(json.dumps({"sources": len(summary["sources"]), "shortfall": summary["shortfall"]}))


if __name__ == "__main__":
    main()
