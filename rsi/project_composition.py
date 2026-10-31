"""Shared immutable project dependency composition for serial and spawn workers."""
import hashlib
import json
import time

from rsi.evidence_io import _references
from rsi.instance_ingestion import load_ingestion_records
from rsi.strategy import knowledge_db_query


def file_identity(path):
    stat = path.stat()
    return [stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns]


def verify_kdb(path, expected):
    before = file_identity(path)
    sha = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            sha.update(block)
    if sha.hexdigest() != expected or before != file_identity(path):
        raise ValueError('frozen KDB changed')
    return before


class ProjectComposition:
    def __init__(self, services, root, binding, spec):
        self.services, self.root, self.store = services, root, services.store
        self.binding, self.spec = binding, spec
        self.config = binding['config']['payload']
        self.database = None

    def ingestion_evidence(self):
        refs = self.config.get('kdb_ingestion_refs', [])
        evidence = load_ingestion_records(self.store, refs)
        previous = None
        for i, reference in enumerate(refs):
            receipt = self.store.read_json(reference)
            if not self.services.test and receipt.get('test_backend') is not False:
                raise ValueError('production query requires production ingestion provenance')
            if (receipt.get('previous_ingestion_refs') != refs[:i]
                    or previous is not None and receipt.get('previous_kdb_sha256') != previous
                    or receipt['manifest'].get('collection') not in self.config.get('kdb_collections', [])):
                raise ValueError('ingestion chain differs from frozen KDB configuration')
            if self.database is not None and json.loads(self.database.manifest(
                    receipt['manifest']['collection']).to_json()) != receipt['manifest']:
                raise ValueError('ingestion source manifest differs from frozen KDB')
            previous = receipt.get('kdb_sha256')
        if refs and previous != self.config.get('kdb_sha256'):
            raise ValueError('ingestion baseline checksum differs from frozen configuration')
        return evidence

    def query(self, memory_ref):
        snapshot = self.store.read_json(memory_ref)
        if snapshot.get('schema') != 'subject-memory-snapshot-v1':
            raise ValueError('accepted memory snapshot required')
        ingestion = self.ingestion_evidence()  # CAS/schema/source binding, never an external store.
        records = snapshot['memories']
        positives, negatives = [], []
        for record in records:
            reference = record.get('provenance', {}).get('packet')
            packet = self.store.read_json(reference) if reference else {}
            observations = record['proposal'].get('observations', packet.get('observations'))
            observation_ref = record['proposal'].get('observations_ref', packet.get('observations_ref'))
            if observation_ref and self.store.read_json(observation_ref) != observations:
                raise ValueError('accepted memory observation reference differs from retained facts')
            if observations is not None and packet.get('observations') not in (None, observations):
                raise ValueError('accepted observations differ from admitted host packet')
            metadata = dict(record, proposal={k: v for k, v in record['proposal'].items()
                if k not in ('observations', 'observations_ref')})
            if record['proposal']['kind'] == 'correctness':
                positives.append(dict(memory=metadata, observations=observations,
                    provenance=dict(snapshot=memory_ref, packet=reference, observations=observation_ref)))
            else:
                row = dict(metadata, observations=observations, observations_ref=observation_ref)
                if packet:
                    row['observed_evidence'] = packet['evidence']
                negatives.append(row)
        if self.services.test:
            adapter = self.services.test.query
        else:
            if self.database is None:
                from benchmarks.a5kernels.embeddings import PinnedBGEEmbeddings
                from benchmarks.a5kernels.knowledge import KnowledgeDB
                path = getattr(self, 'kdb_path', self.root / 'knowledge.sqlite3')
                frozen_stat = getattr(self, 'kdb_stat', None)
                if frozen_stat is not None and file_identity(path) != frozen_stat:
                    raise ValueError('frozen KDB changed')
                if frozen_stat is None:
                    verify_kdb(path, self.binding['kdb'])
                settings = dict(self.config.get('embeddings', {}))
                if settings.get('cache_dir'):
                    settings['cache_dir'] = self.services.path(settings['cache_dir'])
                embeddings = PinnedBGEEmbeddings(**settings)
                with self.services.setup.measure('bge-dependency-model-load', self.spec.cycle_id):
                    embeddings._load()
                self.database = KnowledgeDB.open_read_only(path, embeddings)
            self.ingestion_evidence()  # Authenticate each indexed source manifest as well.
            adapter = knowledge_db_query(self.database, tuple(self.config['kdb_collections']),
                                         negative_query=lambda _: [])
        def query(payload):
            result = json.loads(json.dumps(adapter(payload), allow_nan=False))
            result['negative_records'].extend([*negatives, *ingestion['negative_records']])
            result['results'].extend(positives)
            result['provenance'].update(memory_snapshot=memory_ref,
                                        ingestions=ingestion['provenance'])
            return result
        return query

    def runner(self, root, *, spec=None, bundle=None, config=None, memory_ref,
               backend_identity, reopen=False, configuration_ref=None, command_guard=None):
        from rsi.actor import PiSessionAdapter
        from rsi.project_runner import ProjectRunner
        runner_class = getattr(self, 'runner_class', ProjectRunner)
        admitted_at = time.time()
        configuration_ref = configuration_ref or self.spec.configuration_ref
        frozen_config = self.store.read_json(configuration_ref)['payload']
        provider = frozen_config['provider']
        if not self.services.test:
            from benchmarks.a5kernels.phase1_experiments import MODEL_IDENTITIES
            from benchmarks.a5kernels.phase1_provider import direct_profile
            profiles = [direct_profile(i) for i in MODEL_IDENTITIES if i.profile_id == provider['profile_id']]
            if len(profiles) != 1 or profiles[0].model != provider['model']:
                raise ValueError('registered actor model/profile mismatch')
        def session():
            with self.services.setup.measure('actor-dependency-factory', str(root)):
                adapter = (self.services.test.session_factory() if self.services.test else
                    PiSessionAdapter(model=provider['model'], environ=self.services.environment))
            prepare = adapter.prepare
            def ready(*args, **kwargs):
                with self.services.setup.measure('actor-process-readiness', str(root)):
                    return prepare(*args, **kwargs)
            adapter.prepare = ready
            return adapter
        injections = dict(query=self.query(memory_ref), backend_identity=backend_identity,
                          session_factory=session, test_backend=self.services.test is not None,
                          executor_factory=self.services.executor_factory)
        if command_guard is not None:
            injections['command_guard'] = command_guard
        if not reopen:
            first = True
            def clock():
                nonlocal first
                if first:
                    first = False
                    return admitted_at
                return time.time()
            injections['clock'] = clock
        with self.services.setup.measure('project-open' if reopen else 'project-snapshot-copy-and-bind', str(root)):
            runner = runner_class(root, **injections) if reopen else runner_class.create(
                root, spec, bundle, config, **injections)
            # Reuse export's reference contract to retain the query's artifact closure.
            pending = [*frozen_config.get('kdb_ingestion_refs', []), memory_ref, configuration_ref]
            for record in self.store.read_json(memory_ref)['memories']:
                if record['proposal'].get('observations_ref'):
                    pending.append(record['proposal']['observations_ref'])
                packet_ref = record.get('provenance', {}).get('packet')
                if packet_ref:
                    packet = self.store.read_json(packet_ref)
                    pending.extend(packet.get('observations', {}).get('artifact_refs', []))
                    if packet.get('observations_ref'):
                        pending.append(packet['observations_ref'])
            copied = set()
            while pending:
                reference = pending.pop()
                if reference in copied:
                    continue
                data = self.store.read_bytes(reference)
                runner.store.put_bytes(data)
                copied.add(reference)
                try:
                    value = json.loads(data)
                except (ValueError, UnicodeDecodeError):
                    continue
                pending.extend(_references(value))
        context = runner.context
        runner.context = lambda: dict(context(), user_guidance=frozen_config.get('prompts', {}),
                                      configuration_ref=configuration_ref)
        return runner
