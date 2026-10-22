"""Durable advertised curriculum context, using the Actor's authenticated pages."""
import json
import re

from benchmarks.a5kernels.artifacts import ContentAddressedArtifacts
from benchmarks.a5kernels.evidence import canonical_bytes
from .agent_presentation import artifact_page_presentation


class ContextFiles:
    def __init__(self, root):
        self.root = root
        self.artifacts = ContentAddressedArtifacts(root / 'artifacts')
        self.advertised = set()

    def put(self, name, value):
        data = canonical_bytes(value)
        digest = self.artifacts.put_bytes(data)
        filename = re.sub(r'[^a-zA-Z0-9_.-]', '-', name) + '-' + digest[:12] + '.json'
        path = self.root / filename
        if path.exists() and path.read_bytes() != data:
            raise ValueError('retained context file changed')
        if not path.exists():
            path.write_bytes(data)
        self.advertised.add(digest)
        return dict(filename=filename, sha256=digest, size_bytes=len(data),
                    read_artifact=dict(sha256=digest, offset=0, limit=8192))

    def read(self, sha256, offset=0, limit=8192):
        if sha256 not in self.advertised:
            raise ValueError('artifact is not advertised curriculum context')
        if type(offset) is not int or offset < 0 or type(limit) is not int or not 1 <= limit <= 8192:
            raise ValueError('artifact page requires nonnegative offset and limit 1..8192')
        page = artifact_page_presentation(dict(sha256=sha256, offset=offset, limit=limit), self.artifacts)
        return self.record_read(page)

    def record_read(self, page):
        # Save before returning: a subsequent provider error or elapsed deadline
        # must not discard the full page when no session can be committed.
        with (self.root / 'read-pages.jsonl').open('ab') as stream:
            stream.write(canonical_bytes(page) + b'\n')
        return page


def context_wire(messages):
    """Retain full audit elsewhere; only the newest artifact page stays inline."""
    result = [dict(row) for row in messages]
    pages = []
    for index, row in enumerate(result):
        if row['role'] != 'user':
            continue
        try:
            value = json.loads(row['content'])
        except (ValueError, TypeError):
            continue
        if isinstance(value, dict) and value.get('event') == 'artifact-page':
            pages.append((index, value))
    for index, page in pages[:-1]:
        result[index]['content'] = json.dumps({key: value for key, value in page.items()
            if key in {'event', 'sha256', 'offset', 'size_bytes', 'next_offset', 'full_observation_sha256'}})
    return result
