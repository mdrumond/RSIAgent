"""Immutable RSI output skills and caller-owned usage receipts.

No installed CPL skills are read or modified. Persist ``to_dict()`` in the
existing artifact store; send usage events to the existing run log. An applied
receipt links an observable action; the controller must verify that receipt.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from importlib.resources import files
import json
from pathlib import Path
import re
from time import time
from typing import Callable, Mapping

from rsi.contracts import Stage


def _hash(content: str) -> str:
    return sha256(content.encode('utf-8')).hexdigest()


def _sections(content: str) -> dict[str, str]:
    matches = list(re.finditer(r'^## ([^\n]+)\n', content, re.MULTILINE))
    result = {}
    for index, match in enumerate(matches):
        key = re.sub(r'[^a-z0-9]+', '-', match[1].lower()).strip('-')
        if not key or key in result:
            raise ValueError('section identifiers must be nonempty and unique')
        end = matches[index + 1].start() if index + 1 < len(matches) else len(content)
        result[key] = content[match.end():end]
    return result


@dataclass(frozen=True)
class SkillDocument:
    name: str
    content: str

    def __post_init__(self):
        if not re.fullmatch(r'[a-z0-9][a-z0-9_-]*', self.name):
            raise ValueError('invalid skill name')
        if not isinstance(self.content, str) or not self.content.strip():
            raise ValueError('empty skill document')
        _sections(self.content)

    @property
    def digest(self) -> str:
        return _hash(self.content)

    @property
    def sections(self) -> tuple[str, ...]:
        return tuple(_sections(self.content))

    def read(self, section: str | None = None) -> str:
        if section is None:
            return self.content
        try:
            return _sections(self.content)[section]
        except KeyError as exc:
            raise ValueError(f'unknown section: {section}') from exc


@dataclass(frozen=True)
class SkillBundle:
    documents: tuple[SkillDocument, ...]
    mapping: tuple[tuple[Stage, tuple[str, ...]], ...]

    def __post_init__(self):
        # Copy caller-owned containers before exposing an immutable snapshot.
        docs = tuple(self.documents)
        mapping = tuple((Stage(stage), tuple(names)) for stage, names in self.mapping)
        if any(not isinstance(doc, SkillDocument) for doc in docs):
            raise ValueError('expected skill documents')
        names = {doc.name for doc in docs}
        if len(names) != len(docs):
            raise ValueError('duplicate skill')
        if len(mapping) != len(Stage) or {s for s, _ in mapping} != set(Stage):
            raise ValueError('mapping must cover every stage exactly once')
        for _, assigned in mapping:
            if not assigned or len(set(assigned)) != len(assigned) or not set(assigned) <= names:
                raise ValueError('mapping requires existing unique skills')
        object.__setattr__(self, 'documents', tuple(sorted(docs, key=lambda doc: doc.name)))
        object.__setattr__(self, 'mapping', tuple(sorted(mapping, key=lambda item: item[0].value)))

    @classmethod
    def capture(cls, root: Path, mapping: Mapping[Stage, tuple[str, ...]]) -> SkillBundle:
        """Read a baseline once; subsequent source edits cannot affect it."""
        root = Path(root)
        documents = tuple(SkillDocument(path.parent.name, path.read_text(encoding='utf-8'))
                          for path in sorted(root.glob('*/SKILL.md')))
        return cls(documents, tuple(mapping.items()))

    def document(self, name: str) -> SkillDocument:
        for document in self.documents:
            if document.name == name:
                return document
        raise ValueError(f'unknown skill: {name}')

    def skills_for(self, stage: Stage) -> tuple[str, ...]:
        return dict(self.mapping)[Stage(stage)]

    def to_dict(self) -> dict:
        return {'schema_version': 1,
                'documents': [{'name': d.name, 'content': d.content, 'hash': d.digest}
                              for d in self.documents],
                'mapping': {stage.value: list(names) for stage, names in self.mapping}}

    @classmethod
    def from_dict(cls, value: dict) -> SkillBundle:
        if set(value) != {'schema_version', 'documents', 'mapping'} or value['schema_version'] != 1:
            raise ValueError('unsupported skill bundle schema')
        docs = []
        for item in value['documents']:
            if set(item) != {'name', 'content', 'hash'}:
                raise ValueError('invalid document schema')
            document = SkillDocument(item['name'], item['content'])
            if document.digest != item['hash']:
                raise ValueError('skill content hash mismatch')
            docs.append(document)
        return cls(tuple(docs), tuple((Stage(s), tuple(names))
                                      for s, names in value['mapping'].items()))

    @property
    def digest(self) -> str:
        return _hash(json.dumps(self.to_dict(), sort_keys=True, separators=(',', ':')))


def builtin_bundle() -> SkillBundle:
    """Load packaged templates; callers persist the resulting baseline snapshot."""
    root = files('rsi').joinpath('templates')
    docs = tuple(SkillDocument(folder.name, folder.joinpath('SKILL.md').read_text(encoding='utf-8'))
                 for folder in root.iterdir() if folder.is_dir())
    return SkillBundle(docs, tuple((stage, (stage.value.replace('_', '-'), 'kdb')) for stage in Stage))


class SkillUsage:
    """One operation's instrumented access to mapped skills.

    Exposure reports available names, never implies a read. ``read`` is the
    access boundary and returns bytes only after its log sink succeeds. A citation
    or application requires a prior full-document or matching section read.
    Receipts are controller-generated artifact references, not proof by themselves.
    Replay persisted events through the owning controller after process restart.
    """

    def __init__(self, bundle: SkillBundle, operation_id: str, stage: Stage,
                 emit: Callable[[dict], None], clock: Callable[[], float] = time):
        if not operation_id.strip():
            raise ValueError('operation_id is required')
        self.bundle = bundle
        self.operation_id = operation_id
        self.stage = Stage(stage)
        self.emit = emit
        self.clock = clock
        self._exposed: set[str] = set()
        self._read: set[tuple[str, str | None]] = set()
        self._sequence = 0
        self._receipts: dict[str, str] = {}

    def _event(self, kind: str, name: str, section: str | None = None,
               evidence_ref: str | None = None):
        document = self.bundle.document(name)
        event = {'schema_version': 1, 'sequence': self._sequence,
                 'operation_id': self.operation_id, 'stage': self.stage.value,
                 'bundle_hash': self.bundle.digest, 'skill': name,
                 'skill_hash': document.digest, 'section': section,
                 'kind': kind, 'evidence_ref': evidence_ref, 'timestamp': self.clock()}
        event['content_hash'] = _hash(document.read(section))
        encoded = json.dumps(event, sort_keys=True, separators=(',', ':'))
        event['receipt_id'] = _hash(encoded)
        # Retain an independent authoritative record, never the sink's mutable dict.
        record = json.dumps(event, sort_keys=True)
        self.emit(event)
        self._receipts[json.loads(record)['receipt_id']] = record
        self._sequence += 1
        return json.loads(record)

    def expose(self) -> tuple[str, ...]:
        names = self.bundle.skills_for(self.stage)
        for name in names:
            if name not in self._exposed:
                self._event('exposed', name)
                self._exposed.add(name)
        return names

    def read(self, name: str, section: str | None = None) -> str:
        if name not in self._exposed:
            raise ValueError('skill must be exposed before read')
        content = self.bundle.document(name).read(section)
        self.last_read_receipt = self._event('read', name, section)
        self._read.add((name, section))
        return content

    def _reference(self, kind: str, name: str, section: str | None, evidence_ref: str):
        self.bundle.document(name).read(section)
        if (name, None) not in self._read and (name, section) not in self._read:
            raise ValueError('skill section must be read first')
        if not isinstance(evidence_ref, str) or not evidence_ref.strip():
            raise ValueError('evidence reference is required')
        self._event(kind, name, section, evidence_ref)

    def cite(self, name: str, section: str | None, decision_ref: str):
        self._reference('cited', name, section, decision_ref)

    def apply(self, name: str, section: str | None, action_receipt_ref: str):
        self._reference('applied', name, section, action_receipt_ref)

    def resolve_read(self, receipt_id: str, *, stage: Stage, skill: str,
                     section: str | None = None) -> dict:
        """Resolve only host-recorded reads, scoped to this operation and bundle.

        Runtime evidence resolvers should call this method, never trust a record
        supplied by an actor. Persist these events in the host's run log for
        durable reconstruction of the owning operation after restart.
        """
        try:
            record = json.loads(self._receipts[receipt_id])
        except KeyError as exc:
            raise ValueError('unknown host receipt') from exc
        if (record['kind'] != 'read' or record['stage'] != Stage(stage).value
                or record['skill'] != skill or skill not in self._exposed
                or record['section'] not in (None, section)):
            raise ValueError('receipt does not establish requested stage read')
        if record['content_hash'] != _hash(self.bundle.document(skill).read(record['section'])):
            raise ValueError('receipt content hash mismatch')
        return record
