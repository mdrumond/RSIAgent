"""Bounded JSONL walking for plain Pi logs and storage_archive manifests.

walk_records returns {record, metadata}; record is the unchanged native JSON
value. Metadata includes byte cursors and raw references, never a synthesized
conversation. Cursors are byte boundaries in the uncompressed artifact; callers
must bind persisted cursors to the same immutable source/archive identity.
"""
import argparse
import json
from pathlib import Path

from .storage_archive import BLOCK_BYTES, _chunk_bytes, iter_artifact_bytes

MAX_RECORD_BYTES = 8 * 1024 * 1024


def open_artifact_bytes(artifact, *, cursor=0):
    """Iterate bytes of a plain path or a complete archive manifest dict."""
    if isinstance(artifact, dict) and 'manifest' in artifact:
        artifact = json.loads(Path(artifact['manifest']).read_text())
    if isinstance(artifact, dict):
        if not cursor:
            yield from iter_artifact_bytes(artifact)
            return
        if artifact.get('state') != 'complete':
            raise ValueError('archive is incomplete')
        offset = 0
        for chunk in artifact['chunks']:
            end = offset + chunk['size']
            if end > cursor:
                for block in _chunk_bytes(artifact, chunk):
                    end_block = offset + len(block)
                    if offset < cursor <= end_block and block[cursor - offset - 1] != 10:
                        raise ValueError('cursor is not a record boundary')
                    if end_block > cursor:
                        yield block[max(0, cursor - offset):]
                    offset = end_block
            offset = end
    else:
        with Path(artifact).open('rb') as stream:
            stream.seek(cursor)
            while block := stream.read(BLOCK_BYTES):
                yield block


def walk_records(artifact, *, cursor=0, record_bytes=MAX_RECORD_BYTES,
                 predicate=None):
    """Yield records at/after cursor, optionally selected by predicate(record, meta).

    Oversized/malformed lines yield record=None plus a raw_ref, so data is not
    silently discarded. Oversized lines are never JSON-decoded. Filtering is
    performed after cursor/metadata construction; use next_cursor to resume.
    Resumed line numbers are unknown. Exhaustion verifies visited archive chunks;
    only cursor=0 also verifies the full artifact digest. Early exit may leave the
    current chunk unchecked. A cursor inside a chunk must follow a newline.
    """
    if cursor < 0 or not 0 < record_bytes <= MAX_RECORD_BYTES:
        raise ValueError('invalid cursor or record limit')
    if isinstance(artifact, dict) and 'manifest' in artifact:
        artifact = json.loads(Path(artifact['manifest']).read_text())
    reference = ({'manifest': artifact['manifest_path']}
                 if isinstance(artifact, dict) else str(artifact))
    start = offset = cursor
    line = 0
    buffer = bytearray()
    oversized = False

    def emit(end):
        meta = dict(offset=start, next_cursor=end, line=None if cursor else line + 1,
                    verification=('visited_chunks' if cursor else 'full_artifact')
                    if isinstance(artifact, dict) else 'unverified_plain',
                    raw_ref=dict(artifact=reference, offset=start, length=end - start))
        record = None
        if oversized:
            meta['status'] = 'oversized'
        else:
            try:
                record = json.loads(buffer)
                meta['status'] = 'ok'
            except (ValueError, UnicodeDecodeError, RecursionError):
                meta['status'] = 'invalid'
        return dict(record=record, metadata=meta)

    for block in open_artifact_bytes(artifact, cursor=cursor):
        position = 0
        while position < len(block):
            newline = block.find(b'\n', position)
            end = len(block) if newline < 0 else newline + 1
            piece = block[position:end]
            offset += len(piece)
            if not oversized:
                if len(buffer) + len(piece) > record_bytes:
                    oversized = True
                    buffer.clear()
                else:
                    buffer.extend(piece)
            if newline >= 0:
                if start >= cursor:
                    item = emit(offset)
                    if predicate is None or predicate(item['record'], item['metadata']):
                        yield item
                start = offset
                line += 1
                buffer.clear()
                oversized = False
            position = end
    if offset > start and start >= cursor:
        item = emit(offset)
        if predicate is None or predicate(item['record'], item['metadata']):
            yield item


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('artifact', help='plain JSONL file, or manifest with --manifest')
    parser.add_argument('--manifest', action='store_true')
    parser.add_argument('--cursor', type=int, default=0)
    parser.add_argument('--type', dest='record_type', help='filter native top-level type')
    args = parser.parse_args(argv)
    artifact = json.loads(Path(args.artifact).read_text()) if args.manifest else args.artifact
    predicate = None
    if args.record_type:
        predicate = lambda record, meta: isinstance(record, dict) and record.get('type') == args.record_type
    for item in walk_records(artifact, cursor=args.cursor, predicate=predicate):
        print(json.dumps(item, ensure_ascii=True))


if __name__ == '__main__':
    main()
