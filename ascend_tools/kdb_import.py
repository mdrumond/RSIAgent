"""Explicit, streaming conversion of a legacy JSON-vector DB to a NEW file."""
from __future__ import annotations

from contextlib import closing
import json
import os
from pathlib import Path
import sqlite3

from .knowledge import Chunk, CollectionManifest, KnowledgeDB


def import_legacy(source: Path, destination: Path, embeddings, batch_size: int = 128):
    """Leave source untouched; preserve manifests, citations, scopes and events.

    No embedding model is loaded or invoked. Partial destinations are removed on
    failure, so they cannot be mistaken for complete imports.
    """
    if batch_size < 1:
        raise ValueError('batch_size must be positive')
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    fd = os.open(destination, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    os.close(fd)
    try:
        with closing(sqlite3.connect(source.as_uri() + '?mode=ro', uri=True)) as old:
            old.row_factory = sqlite3.Row
            with KnowledgeDB(destination, embeddings) as new:
                with new.connection:
                    for row in old.execute('SELECT * FROM collections ORDER BY name'):
                        manifest = CollectionManifest.from_json(row['manifest_json'])
                        table = new._fts_table_name(row['name'])
                        new.connection.execute('INSERT INTO collections VALUES(?,?,?,?,?)',
                            (row['name'],row['language'],row['fingerprint'],row['manifest_json'],table))
                        new._ensure_fts_table(table)
                        cursor = old.execute('SELECT * FROM chunks WHERE collection=? ORDER BY rowid',
                                             (row['name'],))
                        while rows := cursor.fetchmany(batch_size):
                            for chunk in rows:
                                new._store_chunk(row['name'], Chunk(**{key: chunk[key] for key in
                                    ('chunk_id','path','start_line','end_line','text')}),
                                    json.loads(chunk['vector_json']),
                                    manifest.embedding_model,manifest.embedding_revision)
                    tables = {row[0] for row in old.execute("SELECT name FROM sqlite_master WHERE type='table'")}
                    # Only compact indexing metadata is copied; verbose logs stay external.
                    for name, ddl, columns in [
                        ('knowledge_scopes', 'collection TEXT PRIMARY KEY,kind TEXT NOT NULL,target TEXT', 3),
                        ('knowledge_events', 'id INTEGER PRIMARY KEY,timestamp TEXT,attempt_id TEXT,operation TEXT,data_json TEXT', 5),
                    ]:
                        if name in tables:
                            new.connection.execute(f'CREATE TABLE {name}({ddl})')
                            cursor = old.execute(f'SELECT * FROM {name}')
                            while rows := cursor.fetchmany(batch_size):
                                new.connection.executemany(
                                    f'INSERT INTO {name} VALUES({",".join("?" for _ in range(columns))})', rows)
    except BaseException:
        destination.unlink(missing_ok=True)
        raise
    return destination


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--destination', type=Path, required=True)
    parser.add_argument('--batch-size', type=int, default=128)
    args = parser.parse_args()
    # Import reads model identities from manifests and never embeds or loads a model.
    import_legacy(args.source, args.destination, None, args.batch_size)


if __name__ == '__main__':
    main()
