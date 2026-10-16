"""Independent, persistent collection of closed campaign profiling reports."""
from __future__ import annotations

import argparse
from collections import Counter
import fcntl
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import time

from benchmarks.ascend.training_annex import report_digest


_SOURCE_DIGEST = '''#!/usr/bin/env python3
import hashlib, json
from pathlib import Path
root = Path(SOURCE_PATH)
paths = sorted(root.rglob("*"))
if any(path.is_symlink() for path in paths):
    raise RuntimeError("report contains symlink")
files = [path for path in paths if path.is_file()]
if not files:
    raise RuntimeError("empty raw report")
digest = hashlib.sha256()
for path in files:
    digest.update(path.relative_to(root).as_posix().encode() + b"\\0")
    digest.update(hashlib.sha256(path.read_bytes()).digest())
print(json.dumps({"source_sha256": digest.hexdigest()}))
'''


class ArtifactCollector:
    """A single transfer worker; state and staging must be outside the campaign.

    Uncertain submissions without a recoverable native handle remain held for
    operator inspection. They are never automatically submitted a second time.
    """

    def __init__(self, campaign_root, artifact_store, state_path, *,
                 cpl_remote=None, free_floor_bytes=5 * 1024**3):
        self.root = Path(campaign_root).resolve()
        self.store = Path(artifact_store).resolve()
        self.path = Path(state_path).resolve()
        for location in (self.store, self.path):
            if location == self.root or self.root in location.parents:
                raise ValueError("collector state and artifacts must be outside campaign")
        self.command = str(cpl_remote or Path.home() /
                           ".agents/skills/remote-access/scripts/cpl-remote")
        self.floor = free_floor_bytes
        if self.floor < 0:
            raise ValueError("free disk floor must be nonnegative")
        self.store.mkdir(parents=True, exist_ok=True)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def _save(self, state):
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        with temporary.open("w") as stream:
            json.dump(state, stream, sort_keys=True)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(self.path)

    def _retry(self, entry, now, error):
        entry["attempts"] = entry.get("attempts", 0) + 1
        deferred = entry["attempts"] >= 3
        entry.update(status="retry", next_retry=now + (900 if deferred else 120),
                     error=error, handle=None, operation=None)
        if deferred:
            entry["attempts"] = 0

    def _complete(self, entry):
        if entry.get("digest_origin") == "collection-time":
            receipt = self.store / (entry["id"] + ".receipt.json")
            temporary = receipt.with_suffix(".tmp")
            temporary.write_text(json.dumps({"descriptor_id": entry["id"],
                                            "source_sha256": entry["expected_sha256"],
                                            "downloaded_sha256": entry["expected_sha256"],
                                            "digest_origin": "collection-time"}))
            temporary.replace(receipt)
        entry.update(status="complete", handle=None, error=None)

    @staticmethod
    def _descriptor_complete(entry):
        return all(isinstance(entry.get(key), str) and entry[key].strip()
                   for key in ("target", "remote_path"))

    def _result(self, state, entry, response, now):
        handle = response.get("handle")
        if handle:
            entry["handle"] = handle
        status = response.get("state")
        operation = entry["operation"]
        entry["operation"] = None
        if operation.startswith("hash-"):
            if handle:
                entry["source_hash_handle"] = handle
            if status in {"failed", "cancelled"}:
                self._retry(entry, now, "source digest job " + status)
            elif status == "completed" and operation == "hash-logs":
                try:
                    digest = json.loads(response["content"])["source_sha256"]
                    if not isinstance(digest, str) or not re.fullmatch("[0-9a-f]{64}", digest):
                        raise ValueError("invalid source digest")
                except (KeyError, TypeError, ValueError) as error:
                    retained = entry.get("handle")
                    self._retry(entry, now, str(error))
                    if retained:
                        entry.update(status="hash-logs", handle=retained)
                else:
                    entry.update(expected_sha256=digest, digest_origin="collection-time",
                                 status="pending", handle=None, next_retry=now)
            elif status == "completed":
                entry["status"] = "hash-logs" if entry.get("handle") else "uncertain"
            elif entry.get("handle"):
                entry.update(status="hash-observe", next_retry=now + 120)
            else:
                entry.update(status="uncertain", error="source digest job handle unavailable")
            self._save(state)
            return
        if status in {"failed", "cancelled"}:
            self._retry(entry, now, status)
        elif status == "completed":
            if operation == "observe":
                entry["status"] = "fetch"
            else:
                staging = self.store / (".partial-" + entry["id"])
                try:
                    valid = report_digest(staging) == entry["expected_sha256"]
                except (OSError, ValueError):
                    valid = False
                if valid:
                    staging.replace(self.store / entry["id"])
                    self._complete(entry)
                else:
                    # A completed but corrupt tree must not poison later fetches.
                    # Keep it for diagnosis; interrupted native partials stay put.
                    if staging.exists():
                        staging.rename(self.store / (".corrupt-" + entry["id"] +
                                                     "-" + str(time.time_ns())))
                    retained = entry.get("handle")
                    self._retry(entry, now, "digest mismatch")
                    if retained:
                        entry.update(status="fetch", handle=retained)
        elif entry.get("handle"):
            entry["status"] = "observe"
            entry["next_retry"] = now + 120
        else:
            entry.update(status="uncertain", error="no recoverable transfer handle")
        self._save(state)

    def _transfer(self, state, entry, now):
        if not entry.get("backend"):
            try:
                probe = subprocess.run([self.command, "--json", "capabilities", entry["target"]],
                                       capture_output=True, text=True, check=False)
                capability = json.loads(probe.stdout)
                if not capability.get("capabilities", {}).get("download"):
                    raise ValueError("target does not support downloads")
                entry["backend"] = capability["backend"]
            except (OSError, ValueError, KeyError) as error:
                self._retry(entry, now, str(error))
                self._save(state)
                return
        staging = self.store / (".partial-" + entry["id"])
        staging.mkdir(exist_ok=True)
        operation = entry["status"] if entry["status"] in {
            "observe", "fetch", "hash-observe", "hash-logs"} else "download"
        if not entry.get("expected_sha256") and operation == "download":
            operation = "hash-submit"
        action = {"hash-submit": "run", "hash-observe": "observe", "hash-logs": "logs"}.get(operation, operation)
        args = [self.command, "--json", action]
        if operation == "hash-submit":
            script = self.path.parent / (self.path.name + ".source-digest.py")
            script.write_text(_SOURCE_DIGEST.replace("SOURCE_PATH", repr(entry["remote_path"])))
            args += [entry["target"], "--file", str(script)]
        elif operation == "download":
            args += [entry["target"], entry["remote_path"].rstrip("/") + "/",
                     str(staging), "--timeout", "120"]
        elif operation == "fetch":
            args += [entry["handle"], str(staging)]
        else:
            args += [entry["handle"]]
        journal = self.path.parent / (self.path.name + ".transport.json")
        # Persist before submission. A process death cannot cause a resubmit.
        journal.write_text("")
        entry.update(status="inflight", operation=operation)
        self._save(state)
        try:
            with journal.open("w") as output, journal.with_suffix(".stderr").open("w") as errors:
                # The child retains the queue lock even if this worker dies.
                subprocess.run(args, stdout=output, stderr=errors, check=False,
                               pass_fds=(self._lock_fd,))
            response = json.loads(journal.read_text())
            if not isinstance(response, dict):
                raise ValueError("transport response must be an object")
        except (OSError, ValueError) as error:
            if entry.get("handle"):
                entry.update(status=operation, next_retry=now + 120, error=str(error))
            elif entry.get("backend") == "ssh" and operation == "download":
                self._retry(entry, now, str(error))
            else:
                entry.update(status="uncertain", error=str(error))
            self._save(state)
            return
        self._result(state, entry, response, now)

    def collect_once(self, now=None):
        """Discover at most every 300 seconds and advance one queued transfer."""
        from benchmarks.ascend.training_annex import inventory_campaign

        now = time.time() if now is None else now
        with self.path.with_suffix(self.path.suffix + ".lock").open("a") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                return {"counts": {"worker_busy": 1}, "pending_retry": None}
            self._lock_fd = lock.fileno()
            state = json.loads(self.path.read_text()) if self.path.exists() else {"entries": {}}
            entries = state["entries"]
            # Recover a completed subprocess response before discovering more work.
            for entry in entries.values():
                if entry["status"] == "inflight":
                    journal = self.path.parent / (self.path.name + ".transport.json")
                    try:
                        response = json.loads(journal.read_text())
                        if not isinstance(response, dict):
                            raise ValueError("invalid transport response")
                        self._result(state, entry, response, now)
                    except (OSError, ValueError):
                        if entry.get("handle"):
                            entry["status"] = entry["operation"]
                        elif entry.get("backend") == "ssh" and entry["operation"] == "download":
                            self._retry(entry, now, "worker interrupted; resume native partial tree")
                        else:
                            entry["status"] = "uncertain"
            if now >= state.get("next_discovery", 0):
                inventory = inventory_campaign(self.root, artifact_store=self.store)
                for descriptor in inventory.get("remote_reports", []):
                    if descriptor["id"] not in entries:
                        item = dict(descriptor, status="pending", attempts=0)
                        if not item.get("expected_sha256") and item.get("resolved_sha256"):
                            item["expected_sha256"] = item["resolved_sha256"]
                        entries[descriptor["id"]] = item
                    elif (entries[descriptor["id"]]["status"] == "complete"
                          and descriptor.get("status", "available") != "available"):
                        # Cache cleanup or corruption can invalidate prior publication.
                        entries[descriptor["id"]].update(status="pending", attempts=0,
                                                         next_retry=now, handle=None)
                state["next_discovery"] = now + 300
            self._save(state)
            low_disk = shutil.disk_usage(self.store).free < self.floor
            if not low_disk:
                for entry in entries.values():
                    if not self._descriptor_complete(entry):
                        entry.update(status="incomplete-descriptor",
                                     error="report target and remote_path must be nonempty strings")
                        self._save(state)
                        continue
                    if entry["status"] in {"complete", "uncertain"}:
                        continue
                    if entry.get("next_retry", 0) > now:
                        continue
                    destination = self.store / entry["id"]
                    valid = False
                    if destination.is_dir():
                        try:
                            valid = report_digest(destination) == entry["expected_sha256"]
                        except (OSError, ValueError):
                            pass
                    if valid:
                        self._complete(entry)
                        self._save(state)
                    else:
                        if destination.exists() and entry.get("expected_sha256"):
                            destination.rename(self.store / (".corrupt-" + entry["id"] +
                                                             "-" + str(time.time_ns())))
                        self._transfer(state, entry, now)
                    break
            retry_times = [e["next_retry"] for e in entries.values()
                           if e["status"] != "complete" and e.get("next_retry")]
            return {"counts": dict(Counter(e["status"] for e in entries.values())),
                    "pending_retry": min(retry_times, default=None), "paused_low_disk": low_disk}

    def watch(self, stop=None):
        """Run separately from campaign execution (for example in durable Screen)."""
        while stop is None or not stop.is_set():
            try:
                print(json.dumps(self.collect_once()), flush=True)
            except Exception as error:
                # Discovery and disk errors belong solely to this worker.
                print(json.dumps({"worker_error": str(error)}), flush=True)
            if stop is None:
                time.sleep(30)
            elif stop.wait(30):
                return


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("campaign_root")
    parser.add_argument("--artifact-store", required=True)
    parser.add_argument("--state", required=True)
    parser.add_argument("--free-floor-bytes", type=int, default=5 * 1024**3)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    worker = ArtifactCollector(args.campaign_root, args.artifact_store, args.state,
                               free_floor_bytes=args.free_floor_bytes)
    if args.once:
        print(json.dumps(worker.collect_once()))
    else:
        worker.watch()


if __name__ == "__main__":
    main()
