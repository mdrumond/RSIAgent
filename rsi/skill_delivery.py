"""Deliver exact reusable baseline folders from immutable skill bundles."""
from hashlib import sha256
import os
from pathlib import Path
import shutil
import tempfile

from core.artifacts import canonical_bytes
from .configuration import _publish as publish_file
from .instances import _publish as publish_directory
from .skills import SkillBundle, validate_metadata


def export_bundle(bundle, destination, *, source=None):
    if not isinstance(bundle, SkillBundle):
        raise ValueError('a captured skill bundle is required')
    validate_metadata(bundle)
    destination = Path(destination).expanduser().absolute()
    if os.path.lexists(destination):
        raise FileExistsError(destination)
    destination = destination.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f'.{destination.name}-', dir=destination.parent))
    try:
        inventory = {}
        def write(name, content):
            path = staging / name
            path.parent.mkdir(parents=True, exist_ok=True)
            publish_file(path, content, immutable=True)
            inventory[name] = sha256(content).hexdigest()
        for document in bundle.documents:
            write(f'{document.name}/SKILL.md', document.content.encode('utf-8'))
        write('map.json', canonical_bytes(bundle.to_dict()['mapping']))
        encoded = canonical_bytes(bundle.to_dict())
        reference = sha256(encoded).hexdigest()
        write('bundle.json', encoded)
        manifest = dict(schema='rsi-skill-export-v1', bundle_ref=reference,
                        bundle_hash=bundle.digest, source=source, files=dict(inventory))
        write('export.json', canonical_bytes(manifest))
        publish_directory(staging, destination)
        descriptor = os.open(destination.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        return dict(destination=str(destination), baseline_skills=str(destination),
                    skill_map=str(destination / 'map.json'), bundle_ref=reference,
                    bundle_hash=bundle.digest, manifest=str(destination / 'export.json'))
    finally:
        if staging.exists():
            shutil.rmtree(staging)
