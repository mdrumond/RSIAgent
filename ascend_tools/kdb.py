"""Current-domain knowledge API over the retained indexed baseline format."""
from __future__ import annotations

from heapq import nsmallest
import hashlib
import json
from pathlib import Path
import os
import tempfile

from .knowledge import KnowledgeDB, RRF_K


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
        selected = []
        rows = self.db.connection.execute('''
            SELECT c.name, s.kind, s.target FROM collections c
            LEFT JOIN knowledge_scopes s ON c.name=s.collection
        ''').fetchall()
        for row in rows:
            scoped_kind = row['kind'] or 'good'
            if kind != 'all' and kind != scoped_kind:
                continue
            # Programming languages do not imply a device/profile binding.
            # Imported baseline collections without an explicit target are universal.
            if target is None or row['target'] is None or target == row['target']:
                selected.append((row, scoped_kind))
        if not selected:
            return []
        backend = self.db.embeddings
        vectors = getattr(backend, 'embed_query', backend.embed)([query])
        if len(vectors) != 1:
            raise ValueError('embedding backend must return one query vector')
        candidates, lexical_ids, metadata = [], set(), {}
        for row, scoped_kind in selected:
            name = row['name']
            vector, lexical = self.db.ranked_candidates(
                name, query, limit=None, query_vector=vectors[0])
            candidates.extend((name, chunk_id, score) for chunk_id, score in vector)
            lexical_ids.update((name, hit['chunk_id']) for hit in lexical)
            metadata[name] = (scoped_kind, row['target'])
        # Retain the global cosine lane from the existing candidate-pool helper.
        # Local BM25 magnitudes and local ranks are not comparable across FTS
        # tables. Order keyword matches by that same corpus-wide cosine lane,
        # then fuse the two GLOBAL ranks before applying the shared result limit.
        candidates.sort(key=lambda item: (-item[2], item[0], item[1]))
        lexical_rank = {key: rank for rank, key in enumerate(
            ((name, chunk_id) for name, chunk_id, _ in candidates
             if (name, chunk_id) in lexical_ids), 1)}
        def fused():
            for rank, (name, chunk_id, _) in enumerate(candidates, 1):
                keyword_rank = lexical_rank.get((name, chunk_id))
                score = 1 / (RRF_K + rank)
                if keyword_rank is not None:
                    score += 1 / (RRF_K + keyword_rank)
                yield (-score, name, chunk_id, rank, keyword_rank)
        hits = []
        for negative_score, name, chunk_id, rank, keyword_rank in nsmallest(limit, fused()):
            hit = self.db.connection.execute('''SELECT chunk_id,path,start_line,end_line,text
                FROM chunks WHERE collection=? AND chunk_id=?''', (name, chunk_id)).fetchone()
            scoped_kind, scoped_target = metadata[name]
            manifest = self.db.manifest(name)
            hits.append(dict(hit, collection=name, kind=scoped_kind,
                             target=scoped_target, language=manifest.language,
                             kernel_languages=list(manifest.kernel_languages),
                             score=-negative_score, vector_rank=rank,
                             lexical_rank=keyword_rank, lexical_score=None,
                             source_sha256=self.db.source_fingerprint(name, hit['path']).sha256))
        return hits

    def read_entry(self, collection: str, entry_id: str):
        """Read indexed evidence without needing its original source checkout."""
        citation = self._citation({'collection': collection, 'entry_id': entry_id})
        row = self.db.connection.execute(
            'SELECT text FROM chunks WHERE collection=? AND chunk_id=?',
            (collection, entry_id)).fetchone()
        encoded = row['text'].encode('utf-8')
        text = encoded[:16384].decode('utf-8', errors='ignore')
        return dict(citation, text=text, total_bytes=len(encoded),
                    truncated=len(encoded) > 16384, evidence=[citation])

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
            if set(ref) - {'path', 'sha256', 'entry_id', 'collection', 'start_line', 'end_line'}:
                raise ValueError('evidence accepts only direct references, never raw_refs')
            if 'collection' in ref or 'entry_id' in ref:
                refs.append(self._citation(ref))
            elif 'path' in ref:
                path = Path(ref['path']).resolve()
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
                if ref.get('sha256', digest) != digest:
                    raise ValueError('evidence hash mismatch')
                refs.append({'path': str(path), 'sha256': digest})
            else:
                raise ValueError('evidence requires a file or indexed entry')
        # Outbox replay must identify the same logical delivery after a crash.
        # Evidence order does not change the lesson identity.
        refs.sort(key=lambda ref: json.dumps(ref, sort_keys=True))
        content = dict(lesson=lesson, outcome=outcome, applicability=applicability,
                       evidence=refs, attempt_id=attempt_id)
        identity = json.dumps(content, sort_keys=True, separators=(",", ":"))
        delivery_id = hashlib.sha256(identity.encode()).hexdigest()
        record = dict(delivery_id=delivery_id, **content)
        root = self.db.path.parent / (self.db.path.name + '.lessons')
        root.mkdir(exist_ok=True)
        source = root / (delivery_id + '.json')
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
        self.add_repo(str(root), 'lesson-' + delivery_id, 'lesson',
                      target=applicability, kind=outcome, paths=[source.name])
        collection = 'lesson-' + delivery_id
        references = [self._citation({'collection': collection, 'entry_id': row['chunk_id']})
                      for row in self.db.connection.execute(
                          'SELECT chunk_id FROM chunks WHERE collection=? ORDER BY start_line,chunk_id',
                          (collection,))]
        return dict(record, collection=collection, entry_id=references[0]['entry_id'],
                    references=references)

    def _citation(self, ref):
        collection, entry_id = ref.get('collection'), ref.get('entry_id')
        row = self.db.connection.execute(
            'SELECT path,start_line,end_line FROM chunks WHERE collection=? AND chunk_id=?',
            (collection, entry_id)).fetchone()
        if row is None:
            raise ValueError('evidence entry is absent')
        location = {key: ref.get(key, row[key]) for key in ('path', 'start_line', 'end_line')}
        self.db.validate_citation(collection, chunk_id=entry_id, **location)
        digest = self.db.source_fingerprint(collection, row['path']).sha256
        if ref.get('sha256', digest) != digest:
            raise ValueError('evidence hash mismatch')
        return dict(collection=collection, entry_id=entry_id, **location, sha256=digest)
