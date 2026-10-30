"""Publish new baseline snapshots without changing any admitted skill bundle."""
import hashlib
import json
import os
from pathlib import Path

from core.artifacts import ContentAddressedArtifacts, canonical_bytes
from .configuration import _bytes, _locations, _publish, set_configuration, show_configuration
from .contracts import Stage
from .skills import SkillBundle, SkillDocument, builtin_bundle, validate_metadata


def _capture(root, payload):
    def path(value):
        value = Path(value)
        return value if value.is_absolute() else root / value
    mapping = json.loads(path(payload.get('skill_map_ref', payload['skill_map'])).read_text())
    return SkillBundle.capture(path(payload.get('baseline_skills', payload['skills'])), _mapping(mapping))


def _mapping(value):
    if not isinstance(value, dict) or any(not isinstance(v, list) or
            any(not isinstance(name, str) for name in v) for v in value.values()):
        raise ValueError('skill map requires stage -> list of skill names')
    return {Stage(stage): tuple(names) for stage, names in value.items()}


def inspect_skills(instance, name=None, *, builtin=False):
    root, _ = _locations(instance)
    snapshot = show_configuration(root)
    bundle = builtin_bundle() if builtin else _capture(root, snapshot['payload'])
    result = dict(bundle_ref=hashlib.sha256(canonical_bytes(bundle.to_dict())).hexdigest(),
                  configuration_revision=snapshot['revision'], **bundle.to_dict())
    if name is not None:
        result['documents'] = [item for item in result['documents'] if item['name'] == bundle.document(name).name]
    return result


def update_skills(instance, *, name=None, content=None, mapping=None, expect_bundle=None):
    root, config = _locations(instance)
    snapshot = show_configuration(root)
    bundle = _capture(root, snapshot['payload'])
    previous = hashlib.sha256(canonical_bytes(bundle.to_dict())).hexdigest()
    if expect_bundle is not None and expect_bundle != previous:
        raise ValueError('baseline skill bundle conflict')
    if mapping is not None:
        bundle = SkillBundle(bundle.documents, tuple(_mapping(mapping).items()))
    else:
        bundle.document(name)  # Editing a known baseline skill, never implicit removal.
        replacement = SkillDocument(name, content)
        bundle = SkillBundle(tuple(replacement if d.name == name else d for d in bundle.documents), bundle.mapping)
    validate_metadata(bundle)
    reference = ContentAddressedArtifacts(root / 'artifacts').put_json(bundle.to_dict())
    directory = config / 'skill-bundles' / reference
    directory.mkdir(parents=True, exist_ok=True)
    for document in bundle.documents:
        folder = directory / document.name
        folder.mkdir(exist_ok=True)
        _publish(folder / 'SKILL.md', document.content.encode(), immutable=True)
    _publish(directory / 'map.json', _bytes({s.value: list(v) for s, v in bundle.mapping}), immutable=True)
    for folder in (directory.parent, config):
        descriptor = os.open(folder, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    payload = dict(snapshot['payload'], baseline_skills=str(directory.relative_to(root)),
                   skill_map_ref=str((directory / 'map.json').relative_to(root)))
    current = set_configuration(root, payload, expect_revision=snapshot['revision'])
    return dict(bundle_ref=reference, previous_bundle_ref=previous,
                configuration_revision=current['revision'], mapping=bundle.to_dict()['mapping'])
