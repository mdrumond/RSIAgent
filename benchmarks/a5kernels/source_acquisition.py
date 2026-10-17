"""Download immutable reference trees and inventory their usable source text.

These are reference artifacts, not build or execution inputs. The inventory
retains the upstream URL and commit even when a verified local clone is reused.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile


TEXT_SUFFIXES = {
    ".asc", ".c", ".cc", ".cpp", ".cxx", ".h", ".hpp", ".hh", ".hxx",
    ".py", ".pyi", ".md", ".rst", ".txt", ".cmake", ".sh", ".yaml",
    ".yml", ".json", ".toml", ".ini", ".cfg", ".proto", ".inc", ".td",
    ".mlir", ".ll", ".csv", ".tpp", ".in", ".html", ".xml", ".cu", ".cuh", ".tla",
}
EXCLUDED_PARTS = {".git", "build", "dist", "__pycache__", "node_modules", ".venv", "venv"}


def git(*args: str, cwd: Path | None = None, timeout: int = 300) -> str:
    result = subprocess.run(
        ["git", *args], cwd=cwd, capture_output=True, check=True, timeout=timeout,
        env={**os.environ, "GIT_TERMINAL_PROMPT": "0"},
    )
    return result.stdout.decode("utf-8").strip()


def selected(path: str) -> bool:
    pure = Path(path)
    return not (set(pure.parts) & EXCLUDED_PARTS) and (
        pure.suffix.lower() in TEXT_SUFFIXES
        or pure.name.lower().startswith(("license", "copying", "notice", "readme"))
        or pure.name.lower().endswith("notice")
        or pure.name in {"CMakeLists.txt", "Makefile", "Dockerfile"}
    )


def tracked_blobs(root: Path, revision: str):
    """Stream raw Git objects through one process, independent of checkout filters."""
    entries = git("ls-tree", "-rz", revision, cwd=root).split("\0")
    with subprocess.Popen(["git", "cat-file", "--batch"], cwd=root,
                          stdin=subprocess.PIPE, stdout=subprocess.PIPE) as process:
        try:
            for entry in sorted(filter(None, entries)):
                metadata, relative = entry.split("\t", 1)
                mode, kind, oid = metadata.split()
                if not selected(relative) or kind != "blob" or mode == "120000":
                    yield relative, None
                    continue
                process.stdin.write((oid + "\n").encode("ascii"))
                process.stdin.flush()
                header = process.stdout.readline().split()
                if len(header) != 3 or header[1] != b"blob":
                    raise ValueError("Git blob batch returned an invalid object")
                data = process.stdout.read(int(header[2]))
                if len(data) != int(header[2]) or process.stdout.read(1) != b"\n":
                    raise ValueError("Git blob batch ended unexpectedly")
                yield relative, data
        finally:
            process.stdin.close()
        if process.wait() != 0:
            raise ValueError("Git blob batch failed")


def inventory_tree(root: Path, revision: str) -> tuple[list[dict], list[dict]]:
    """Hash canonical Git UTF-8 text; classify every omitted path explicitly."""
    files, omitted = [], []
    for relative, data in tracked_blobs(root, revision):
        reason = None
        if not selected(relative):
            reason = "excluded-generated-or-non-source"
        elif data is None:
            reason = "symlink-or-submodule"
        else:
            try:
                data.decode("utf-8")
            except UnicodeDecodeError:
                reason = "not-utf8"
            else:
                if b"\0" in data:
                    reason = "binary"
                else:
                    files.append({"path": relative, "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)})
        if reason:
            omitted.append({"path": relative, "reason": reason})
    return sorted(files, key=lambda item: item["path"]), sorted(omitted, key=lambda item: item["path"])


def canonical_root(root: Path, revision: str, files: list[dict]) -> Path:
    """Keep identical checkouts; otherwise export raw blobs beside the checkout."""
    def matches(directory):
        return all((directory / item["path"]).is_file() and
                   hashlib.sha256((directory / item["path"]).read_bytes()).hexdigest() == item["sha256"]
                   for item in files)

    if matches(root):
        return root
    destination = root.with_name(revision + "-raw")
    if destination.exists():
        if not matches(destination):
            raise ValueError("existing canonical export does not match pinned Git blobs")
        return destination
    allowed = {item["path"] for item in files}
    with tempfile.TemporaryDirectory(prefix=".raw-", dir=root.parent) as staging:
        sources = Path(staging) / "sources"
        sources.mkdir()
        for relative, data in tracked_blobs(root, revision):
            if relative in allowed:
                target = sources / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(data)
        os.replace(sources, destination)
    return destination


def acquire_source(spec: dict, destination: Path, reuse: Path | None = None) -> dict:
    name, repository = spec["name"], spec["repository"]
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]*", name):
        raise ValueError("source name must be a safe directory component")
    if not repository.startswith("https://"):
        raise ValueError("public upstream repository must use HTTPS")
    requested = spec.get("revision")
    if requested and not re.fullmatch(r"[0-9a-f]{40}", requested):
        raise ValueError("revision must be a full immutable commit")
    base = destination / name
    base.mkdir(parents=True, exist_ok=True)
    # A pinned existing snapshot is resumable without a network request.
    if requested and (base / requested).is_dir():
        root = base / requested
        if git("rev-parse", "HEAD", cwd=root) != requested:
            raise ValueError("existing snapshot revision mismatch")
    else:
        with tempfile.TemporaryDirectory(prefix=".fetch-", dir=base) as staging:
            root = Path(staging) / "tree"
            if reuse is not None:
                if git("remote", "get-url", "origin", cwd=reuse).removesuffix(".git") != repository.removesuffix(".git"):
                    raise ValueError("local reuse origin does not match upstream")
                git("clone", "--quiet", "--no-hardlinks", str(reuse), str(root))
                git("remote", "set-url", "origin", repository, cwd=root)
            else:
                git("clone", "--quiet", "--depth=1", repository, str(root))
            if requested and git("rev-parse", "HEAD", cwd=root) != requested:
                git("fetch", "--quiet", "--depth=1", "origin", requested, cwd=root)
                git("checkout", "--quiet", "--detach", requested, cwd=root)
            revision = git("rev-parse", "HEAD", cwd=root)
            if reuse is not None:
                # A matching origin URL does not establish that local HEAD was
                # published. Fetch upstream refs into a separate namespace so
                # cloned local branches cannot count as provenance evidence.
                git("fetch", "--quiet", "--no-tags", "origin",
                    "+refs/heads/*:refs/upstream-verification/heads/*",
                    "+refs/tags/*:refs/upstream-verification/tags/*", cwd=root)
                if not git("for-each-ref", "--format=%(refname)",
                           "--contains=" + revision, "refs/upstream-verification/", cwd=root):
                    raise ValueError("reused commit is not reachable from upstream branches or tags")
            target = base / revision
            if not target.exists():
                os.replace(root, target)
            root = target
    revision = git("rev-parse", "HEAD", cwd=root)
    if git("status", "--porcelain", cwd=root):
        raise ValueError("source snapshot has local modifications")
    files, omitted = inventory_tree(root, revision)
    root = canonical_root(root, revision, files)
    return {
        "name": name, "repository": repository, "revision": revision,
        "root": str(root.resolve()), "status": "complete", "files": files,
        "kernel_languages": spec.get("kernel_languages", ["common"]),
        "document_language": spec.get("document_language", "mixed"),
        "reference_exposed": False,
        "licenses": [f["path"] for f in files if Path(f["path"]).name.lower().startswith(("license", "copying", "notice")) or Path(f["path"]).name.lower().endswith("notice")],
        "omitted": omitted,
    }


def acquire_sources(specs: list[dict], destination: Path, *, reuse: dict[str, Path] | None = None, workers: int = 3) -> dict:
    """Retain partial success and explicit errors; reruns can reuse pinned specs."""
    reuse = reuse or {}
    names = [spec["name"] for spec in specs]
    if len(names) != len(set(names)):
        raise ValueError("duplicate source names")

    def one(spec):
        try:
            return acquire_source(spec, destination, reuse.get(spec["name"]))
        except (OSError, ValueError, subprocess.SubprocessError) as exc:
            # Do not include transport stdout/stderr, which may contain credentials.
            return {"name": spec["name"], "repository": spec["repository"], "status": "failed", "error": type(exc).__name__}

    with ThreadPoolExecutor(max_workers=workers) as pool:
        sources = list(pool.map(one, specs))
    return {"schema_version": 1, "retrieved_at": datetime.now(timezone.utc).isoformat(), "sources": sources}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--reuse", action="append", default=[], metavar="NAME=CLONE")
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args(argv)
    if args.workers < 1:
        parser.error("--workers must be positive")
    reuse = dict(item.split("=", 1) for item in args.reuse)
    result = acquire_sources(json.loads(args.config.read_text())["sources"], args.destination, reuse={k: Path(v) for k, v in reuse.items()}, workers=args.workers)
    args.inventory.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", dir=args.inventory.parent, delete=False) as stream:
        json.dump(result, stream, indent=2)
        temporary = stream.name
    os.replace(temporary, args.inventory)
    print(json.dumps({"inventory": str(args.inventory), "sources": [{"name": s["name"], "status": s["status"], "files": len(s.get("files", []))} for s in result["sources"]]}))
    return int(any(s["status"] != "complete" for s in result["sources"]))


if __name__ == "__main__":
    raise SystemExit(main())
