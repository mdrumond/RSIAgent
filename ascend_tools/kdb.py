"""Current-domain knowledge API over the content-deduplicated binary index."""
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
            CREATE TABLE IF NOT EXISTS repository_snapshots (
                snapshot_key TEXT NOT NULL, collection TEXT NOT NULL UNIQUE,
                active INTEGER NOT NULL, PRIMARY KEY(snapshot_key,collection)
            );
            CREATE UNIQUE INDEX IF NOT EXISTS active_snapshot
                ON repository_snapshots(snapshot_key) WHERE active=1;
            CREATE TABLE IF NOT EXISTS generation_collections (
                generation TEXT NOT NULL, collection TEXT NOT NULL,
                PRIMARY KEY(generation,collection)
            );
            CREATE TABLE IF NOT EXISTS generations (revision TEXT PRIMARY KEY);
            CREATE TABLE IF NOT EXISTS active_generation (
                singleton INTEGER PRIMARY KEY CHECK(singleton=1), revision TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS memory_collections (collection TEXT PRIMARY KEY);
            CREATE TABLE IF NOT EXISTS memory_revisions (revision TEXT PRIMARY KEY);
            CREATE TABLE IF NOT EXISTS retired_snapshots (collection TEXT PRIMARY KEY);
            CREATE TABLE IF NOT EXISTS cited_chunks (
                collection TEXT NOT NULL, chunk_id TEXT NOT NULL,
                PRIMARY KEY(collection,chunk_id)
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
               limit: int = 10, generation: str | None = None):
        if kind not in ('good', 'negative', 'all') or not 1 <= limit <= 50:
            raise ValueError('kind must be good/negative/all; limit must be 1..50')
        if generation is None:
            active = self.db.connection.execute('SELECT revision FROM active_generation').fetchone()
            generation = active[0] if active else None
        elif not self.db.connection.execute(
                'SELECT 1 FROM memory_revisions WHERE revision=?', (generation,)).fetchone():
            raise KeyError(generation)
        selected = []
        rows = self.db.connection.execute('''
            SELECT c.name, s.kind, s.target FROM collections c
            LEFT JOIN knowledge_scopes s ON c.name=s.collection
            LEFT JOIN repository_snapshots r ON r.collection=c.name
            LEFT JOIN memory_collections m ON m.collection=c.name
            WHERE (r.active IS NULL OR r.active=1) AND
                (m.collection IS NULL OR EXISTS (SELECT 1 FROM generation_collections g
                  WHERE g.collection=c.name AND g.generation=?))
        ''', (generation,)).fetchall()
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
            hits.append(dict(hit, collection=name, kind=scoped_kind, generation=generation,
                             target=scoped_target, language=manifest.language,
                             kernel_languages=list(manifest.kernel_languages),
                             score=-negative_score, vector_rank=rank,
                             lexical_rank=keyword_rank, lexical_score=None,
                             source_sha256=self.db.source_fingerprint(name, hit['path']).sha256))
        with self.db.connection:
            self.db.connection.executemany('INSERT OR IGNORE INTO cited_chunks VALUES(?,?)',
                                           ((hit['collection'], hit['chunk_id']) for hit in hits))
        return hits

    def read_entry(self, collection: str, entry_id: str, generation: str | None = None):
        """Read indexed evidence without needing its original source checkout."""
        explicit_generation = generation is not None
        is_memory = self.db.connection.execute(
            'SELECT 1 FROM memory_collections WHERE collection=?', (collection,)).fetchone()
        if generation is None:
            active = self.db.connection.execute('SELECT revision FROM active_generation').fetchone()
            generation = active[0] if active else None
        if is_memory and collection not in self.generation_collections(generation):
            if explicit_generation:
                raise ValueError('entry is not a member of the requested memory generation')
            member = self.db.connection.execute(
                'SELECT generation FROM generation_collections WHERE collection=? ORDER BY generation LIMIT 1',
                (collection,)).fetchone()
            generation = member[0] if member else None
        citation = self._citation({'collection': collection, 'entry_id': entry_id})
        row = self.db.connection.execute(
            'SELECT text FROM chunks WHERE collection=? AND chunk_id=?',
            (collection, entry_id)).fetchone()
        encoded = row['text'].encode('utf-8')
        text = encoded[:16384].decode('utf-8', errors='ignore')
        return dict(citation, generation=generation, text=text, total_bytes=len(encoded),
                    truncated=len(encoded) > 16384, evidence=[citation])

    def add_repo(self, path: str, collection: str, language: str,
                 target: str | None = None, kind: str = 'good',
                 paths: list[str] | None = None):
        if kind not in ('good', 'negative'):
            raise ValueError('kind must be good or negative')
        if self.db.connection.execute('SELECT 1 FROM retired_snapshots WHERE collection=?',
                                      (collection,)).fetchone():
            raise ValueError('snapshot was pruned; use a new collection')
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

    def replace_repo(self, snapshot_key: str, path: str, collection: str,
                     language: str, target=None, kind='good', paths=None):
        """Index before switching an active snapshot; old citations stay readable."""
        existing = self.db.connection.execute(
            'SELECT snapshot_key FROM repository_snapshots WHERE collection=?',
            (collection,)).fetchone()
        if existing and existing[0] != snapshot_key:
            raise ValueError('collection belongs to a different snapshot key')
        manifest = self.add_repo(path, collection, language, target, kind, paths)
        with self.db.connection:
            self.db.connection.execute(
                'UPDATE repository_snapshots SET active=0 WHERE snapshot_key=?', (snapshot_key,))
            self.db.connection.execute(
                'INSERT INTO repository_snapshots VALUES(?,?,1) '
                'ON CONFLICT(snapshot_key,collection) DO UPDATE SET active=1',
                (snapshot_key, collection))
        return manifest

    def pin_generation(self, revision: str, collections):
        """Freeze membership under an immutable memory Git revision."""
        names = tuple(sorted(set(collections)))
        for name in names:
            if self.db.connection.execute('SELECT 1 FROM retired_snapshots WHERE collection=?',
                                          (name,)).fetchone():
                raise ValueError('cannot pin a pruned snapshot')
            self.db.manifest(name)
        exists = self.db.connection.execute(
            'SELECT 1 FROM generations WHERE revision=?', (revision,)).fetchone()
        if exists and self.generation_collections(revision) != names:
            raise ValueError('generation membership is immutable')
        with self.db.connection:
            self.db.connection.execute('INSERT OR IGNORE INTO generations VALUES(?)', (revision,))
            self.db.connection.executemany(
                'INSERT OR IGNORE INTO generation_collections VALUES(?,?)',
                ((revision, name) for name in names))

    def generation_collections(self, revision: str):
        return tuple(row[0] for row in self.db.connection.execute(
            'SELECT collection FROM generation_collections WHERE generation=? ORDER BY collection',
            (revision,)))

    def activate_generation(self, revision: str):
        if not self.db.connection.execute(
                'SELECT 1 FROM generations WHERE revision=?', (revision,)).fetchone():
            raise KeyError(revision)
        with self.db.connection:
            self.db.connection.execute(
                'INSERT INTO active_generation VALUES(1,?) '
                'ON CONFLICT(singleton) DO UPDATE SET revision=excluded.revision', (revision,))

    def release_generation(self, revision: str):
        if self.db.connection.execute(
                'SELECT 1 FROM active_generation WHERE revision=?', (revision,)).fetchone():
            raise ValueError('cannot release the active generation')
        with self.db.connection:
            self.db.connection.execute('DELETE FROM generation_collections WHERE generation=?', (revision,))
            self.db.connection.execute('DELETE FROM generations WHERE revision=?', (revision,))
            self.db.connection.execute('DELETE FROM memory_revisions WHERE revision=?', (revision,))

    def current_memory_revision(self):
        row = self.db.connection.execute('SELECT revision FROM active_generation').fetchone()
        return row[0] if row else None

    def stage_memory_revision(self, revision: str, records: list[dict]):
        """Index a complete memory revision invisibly; reuse unchanged records."""
        if not revision:
            raise ValueError('memory revision is required')
        staged, identifiers = [], set()
        for record in records:
            identity = record['id']
            if not identity or identity in identifiers:
                raise ValueError('memory record ids must be nonempty and unique')
            identifiers.add(identity)
            if record['kind'] not in ('good', 'negative') or not record['lesson'].strip():
                raise ValueError('memory records require a lesson and good/negative kind')
            # A record is the immutable indexing unit, independent of its Git revision.
            payload = json.dumps({key: record[key] for key in
                ('id','delivery_id','lesson','evidence_refs','kind','target')},
                sort_keys=True, separators=(',', ':'))
            name = 'memory-' + hashlib.sha256(payload.encode()).hexdigest()
            staged.append((name, record, payload))
        names = tuple(sorted(name for name, _, _ in staged))
        exists = self.db.connection.execute('SELECT 1 FROM generations WHERE revision=?',
                                             (revision,)).fetchone()
        if exists and self.generation_collections(revision) != names:
            raise ValueError('generation membership is immutable')
        for name, record, payload in staged:
            # Register before index() commits so no reader can see staged memory.
            with self.db.connection:
                self.db.connection.execute('INSERT OR IGNORE INTO memory_collections VALUES(?)', (name,))
            with tempfile.TemporaryDirectory(prefix='kdb-memory-') as directory:
                source = Path(directory) / 'record.json'
                source.write_text(payload, encoding='utf-8')
                self.add_repo(directory, name, 'lesson', target=record['target'],
                              kind=record['kind'], paths=[source.name])
        self.pin_generation(revision, names)
        with self.db.connection:
            self.db.connection.execute('INSERT OR IGNORE INTO memory_revisions VALUES(?)', (revision,))
        return {'generation': revision, 'collections': list(names)}

    def adopt_memory_revision(self, revision: str, collections):
        """Adopt an indexed lesson baseline atomically, without embedding work."""
        if not revision:
            raise ValueError('memory revision is required')
        names = tuple(sorted(set(collections)))
        conn = self.db.connection
        with conn:
            conn.execute('BEGIN IMMEDIATE')
            for name in names:
                row = conn.execute('SELECT language FROM collections WHERE name=?', (name,)).fetchone()
                if row is None or row['language'] != 'lesson':
                    raise ValueError('baseline memory collections must be indexed lessons')
                if conn.execute('SELECT 1 FROM retired_snapshots WHERE collection=?', (name,)).fetchone():
                    raise ValueError('cannot adopt a pruned snapshot')
            exists = conn.execute('SELECT 1 FROM generations WHERE revision=?', (revision,)).fetchone()
            if exists and self.generation_collections(revision) != names:
                raise ValueError('generation membership is immutable')
            conn.execute('INSERT OR IGNORE INTO generations VALUES(?)', (revision,))
            conn.executemany('INSERT OR IGNORE INTO memory_collections VALUES(?)', ((name,) for name in names))
            conn.executemany('INSERT OR IGNORE INTO generation_collections VALUES(?,?)',
                             ((revision,name) for name in names))
            conn.execute('INSERT OR IGNORE INTO memory_revisions VALUES(?)', (revision,))
            conn.execute('INSERT INTO active_generation VALUES(1,?) '
                         'ON CONFLICT(singleton) DO UPDATE SET revision=excluded.revision', (revision,))
        return {'generation': revision, 'collections': list(names)}

    def activate_memory_revision(self, revision: str):
        """One metadata commit switches every memory record; repositories stay active."""
        if not self.db.connection.execute('SELECT 1 FROM memory_revisions WHERE revision=?',
                                          (revision,)).fetchone():
            raise KeyError(revision)
        self.activate_generation(revision)
        return {'generation': revision}

    def prune_snapshots(self, *, compact: bool = True):
        """Reclaim superseded, unpinned rows, retaining every issued citation.

        Only explicit snapshot members are eligible. Live generation pins retain
        entire collections; citations retain their exact content and provenance.
        VACUUM returns free pages to the filesystem when compact is true.
        """
        conn = self.db.connection
        removed = 0
        with conn:
            conn.execute('BEGIN IMMEDIATE')
            stale = conn.execute("""SELECT c.name AS collection FROM collections c
                LEFT JOIN repository_snapshots r ON r.collection=c.name
                LEFT JOIN memory_collections m ON m.collection=c.name
                WHERE (r.active=0 OR m.collection IS NOT NULL)
                  AND NOT EXISTS (SELECT 1 FROM generation_collections g
                    WHERE g.collection=c.name)""").fetchall()
            for row in stale:
                name = row['collection']
                table = self.db._fts_table_name(name)
                conn.execute(f'DROP TABLE {table}')
                cursor = conn.execute("""DELETE FROM chunk_provenance
                    WHERE collection=? AND NOT EXISTS (SELECT 1 FROM cited_chunks c
                        WHERE c.collection=chunk_provenance.collection
                          AND c.chunk_id=chunk_provenance.chunk_id)""", (name,))
                removed += cursor.rowcount
                if cursor.rowcount and not conn.execute(
                        'SELECT 1 FROM memory_collections WHERE collection=?', (name,)).fetchone():
                    conn.execute('INSERT OR IGNORE INTO retired_snapshots VALUES(?)', (name,))
                if conn.execute('SELECT 1 FROM chunk_provenance WHERE collection=? LIMIT 1',
                                (name,)).fetchone():
                    self.db._ensure_fts_table(table)
                    conn.execute(f'INSERT INTO {table}(rowid,text) '
                                 'SELECT rowid,text FROM chunks WHERE collection=?', (name,))
                else:
                    for metadata in ('repository_snapshots','knowledge_scopes','memory_collections'):
                        conn.execute(f'DELETE FROM {metadata} WHERE collection=?', (name,))
                    conn.execute('DELETE FROM collections WHERE name=?', (name,))
            conn.execute("""DELETE FROM chunk_vectors WHERE NOT EXISTS (
                SELECT 1 FROM chunk_provenance p JOIN collections c ON c.name=p.collection
                WHERE p.content_id=chunk_vectors.content_id
                  AND json_extract(c.manifest_json,'$.embedding_model')=chunk_vectors.model
                  AND json_extract(c.manifest_json,'$.embedding_revision')=chunk_vectors.revision)""")
            conn.execute('DELETE FROM chunk_contents WHERE NOT EXISTS ('
                         'SELECT 1 FROM chunk_provenance p WHERE p.content_id=chunk_contents.content_id)')
        for row in stale:
            self.db._invalidate(row['collection'])
        if compact:
            conn.execute('VACUUM')
        return {'removed_chunks': removed}

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
        with self.db.connection:
            self.db.connection.execute('INSERT OR IGNORE INTO cited_chunks VALUES(?,?)',
                                       (collection, entry_id))
        return dict(collection=collection, entry_id=entry_id, **location, sha256=digest)
