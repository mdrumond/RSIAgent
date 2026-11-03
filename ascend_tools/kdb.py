"""Current-domain knowledge API over the retained indexed baseline format."""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import os
import tempfile

from .knowledge import KnowledgeDB


class KDB:
    def __init__(self, path: Path, embeddings):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = KnowledgeDB(path, embeddings)
        self.db.connection.executescript("""
            CREATE TABLE IF NOT EXISTS knowledge_scopes (
                collection TEXT PRIMARY KEY, kind TEXT NOT NULL, target TEXT
            );
            CREATE TABLE IF NOT EXISTS knowledge_events (
                id INTEGER PRIMARY KEY, timestamp TEXT DEFAULT CURRENT_TIMESTAMP,
                attempt_id TEXT, operation TEXT NOT NULL, data_json TEXT NOT NULL
            );
        """)

    def close(self):
        self.db.close()

    def event(self, operation, attempt_id, **data):
        with self.db.connection:
            self.db.connection.execute(
                'INSERT INTO knowledge_events(attempt_id,operation,data_json) VALUES(?,?,?)',
                (attempt_id, operation, json.dumps(data)),
            )

    def search(self, query: str, kind: str = 'all', target: str | None = None,
               limit: int = 10):
        if kind not in ('good', 'negative', 'all') or not 1 <= limit <= 50:
            raise ValueError('kind must be good/negative/all; limit must be 1..50')
        hits = []
        rows = self.db.connection.execute('''
            SELECT c.name, s.kind, s.target FROM collections c
            LEFT JOIN knowledge_scopes s ON c.name=s.collection
        ''').fetchall()
        for row in rows:
            manifest = self.db.manifest(row['name'])
            scoped_kind = row['kind'] or 'good'
            if kind != 'all' and kind != scoped_kind:
                continue
            scopes = (row['target'],) if row['target'] else (
                manifest.language, *manifest.kernel_languages)
            if target is not None and target not in scopes:
                continue
            for hit in self.db.query(row['name'], query, limit=limit):
                hits.append(dict(asdict(hit), collection=row['name'], kind=scoped_kind,
                                 target=row['target'],
                                 source_sha256=self.db.source_fingerprint(
                                     row['name'], hit.path).sha256))
        return sorted(hits, key=lambda hit: (-hit['score'], hit['collection'],
                                           hit['chunk_id']))[:limit]

    def add_repo(self, path: str, collection: str, language: str,
                 target: str | None = None, kind: str = 'good',
                 paths: list[str] | None = None):
        if kind not in ('good', 'negative'):
            raise ValueError('kind must be good or negative')
        root = Path(path).resolve()
        # Explicit paths are preferred; default excludes VCS and binary artifacts.
        sources = [root / name for name in paths] if paths is not None else [
            p for p in root.rglob('*') if p.is_file() and
            not any(part.startswith('.') for part in p.relative_to(root).parts) and
            p.suffix in ('.py', '.md', '.c', '.h', '.cpp', '.hpp', '.mlir', '.txt')]
        existing = self.db.connection.execute(
            'SELECT kind,target FROM knowledge_scopes WHERE collection=?',
            (collection,)).fetchone()
        if existing and (existing['kind'], existing['target']) != (kind, target):
            raise ValueError('collection scope is immutable; use a new collection')
        manifest = self.db.index(root, sources, collection=collection, language=language)
        with self.db.connection:
            self.db.connection.execute(
                'INSERT OR IGNORE INTO knowledge_scopes VALUES(?,?,?)',
                (collection, kind, target))
        return json.loads(manifest.to_json())

    def record_lesson(self, lesson: str, outcome: str, applicability: str,
                      evidence: list[dict], attempt_id: str):
        if outcome not in ('good', 'negative'):
            raise ValueError('outcome must be good or negative')
        if not attempt_id or not lesson.strip() or len(lesson) > 4000:
            raise ValueError('attempt_id and a compact lesson are required')
        if not applicability.strip() or len(applicability) > 200:
            raise ValueError('applicability must name a bounded target scope')
        if not 1 <= len(evidence) <= 16:
            raise ValueError('provide 1..16 direct evidence references')
        refs = []
        for ref in evidence:
            if set(ref) - {'path', 'sha256', 'entry_id', 'collection'}:
                raise ValueError('evidence accepts only direct references, never raw_refs')
            if 'path' in ref:
                path = Path(ref['path']).resolve()
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
                if ref.get('sha256', digest) != digest:
                    raise ValueError('evidence hash mismatch')
                refs.append({'path': str(path), 'sha256': digest})
            elif ref.get('entry_id') and ref.get('collection'):
                row = self.db.connection.execute(
                    'SELECT 1 FROM chunks WHERE collection=? AND chunk_id=?',
                    (ref['collection'], ref['entry_id'])).fetchone()
                if row is None:
                    raise ValueError('evidence entry is absent')
                refs.append({'entry_id': ref['entry_id'], 'collection': ref['collection']})
            else:
                raise ValueError('evidence requires a file or indexed entry')
        # Outbox replay must identify the same logical delivery after a crash.
        # Evidence order does not change the lesson identity.
        refs.sort(key=lambda ref: json.dumps(ref, sort_keys=True))
        content = dict(lesson=lesson, outcome=outcome, applicability=applicability,
                       evidence=refs, attempt_id=attempt_id)
        identity = json.dumps(content, sort_keys=True, separators=(",", ":"))
        entry_id = hashlib.sha256(identity.encode()).hexdigest()
        record = dict(entry_id=entry_id, **content)
        root = self.db.path.parent / (self.db.path.name + '.lessons')
        root.mkdir(exist_ok=True)
        source = root / (entry_id + '.json')
        if source.exists():
            if json.loads(source.read_text()) != record:
                raise ValueError('persisted lesson does not match its content identity')
        else:
            fd, temporary = tempfile.mkstemp(prefix='.', suffix='.tmp', dir=root)
            try:
                with os.fdopen(fd, 'w') as output:
                    output.write(json.dumps(record, indent=2) + '\n')
                    output.flush()
                    os.fsync(output.fileno())
                os.replace(temporary, source)
                directory = os.open(root, os.O_RDONLY)
                try:
                    os.fsync(directory)
                finally:
                    os.close(directory)
            finally:
                Path(temporary).unlink(missing_ok=True)
        # index() returns an existing immutable collection without embedding again.
        # Keep a staged file after an indexing failure: replay completes this step.
        self.add_repo(str(root), 'lesson-' + entry_id, 'lesson',
                      target=applicability, kind=outcome, paths=[source.name])
        return record
