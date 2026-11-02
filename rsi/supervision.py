"""Durable project log audits, driven by existing controller progress hooks."""
from contextlib import contextmanager
import fcntl
import json
import math
from pathlib import Path
import time

from core.artifacts import ContentAddressedArtifacts
from rsi.event_journal import read_event_refs
from rsi.python_execution import atomic_json


class ProjectSupervision:
    def __init__(self, root, *, clock=time.time):
        self.root, self.clock = Path(root), clock
        self.store = ContentAddressedArtifacts(self.root / 'artifacts')
        self.path = self.root / 'supervision.json'

    def _identity(self):
        return dict(contract_ref=json.loads((self.root / 'runner.json').read_text())['contract'])

    @contextmanager
    def _state(self):
        with (self.root / 'supervision.lock').open('a+') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            now = self.clock()
            if not math.isfinite(now):
                raise ValueError('finite supervision clock required')
            scope = self._identity()
            state = (json.loads(self.path.read_text()) if self.path.exists() else
                dict(schema='rsi-supervision-v1', scope=scope,
                     last_audit=now, last_clock=now, host_cursor=0, native_cursor=None,
                     faults={}, reports=[]))
            if state['scope'] != scope or now < state['last_clock']:
                raise ValueError('supervision contract or elapsed clock changed')
            before = json.dumps(state, sort_keys=True) if self.path.exists() else None
            yield state, now
            if before != json.dumps(state, sort_keys=True):
                atomic_json(self.path, state)

    def _native(self, cursor):
        path = self._native_path()
        if not path.exists():
            return cursor, []
        stat = path.stat()
        identity = [stat.st_dev, stat.st_ino]
        offset = cursor['offset'] if cursor and cursor['identity'] == identity else 0
        if stat.st_size < offset:
            offset = 0
        refs = []
        with path.open('rb') as stream:
            stream.seek(offset)
            while True:
                line = stream.readline()
                if not line or not line.endswith(b'\n'):
                    break  # A writer's partial record remains for the next audit.
                refs.append(self.store.put_bytes(line))
                offset = stream.tell()
        return dict(identity=identity, offset=offset), refs

    def _native_path(self):
        return self.root / 'native' / 'session' / 'native-events.jsonl'

    def poll(self, *, fault_ref=None):
        """Read committed evidence; a fault fact never proves external closure."""
        with self._state() as (state, now):
            if fault_ref is None and now - state['last_audit'] < 600:
                return None
            refs = read_event_refs(self.root)
            fault = None
            if fault_ref is not None:
                if fault_ref not in refs:
                    raise ValueError('fault must belong to the canonical host journal')
                event = self.store.read_json(fault_ref)
                fact = event.get('event', {})
                if (event.get('type') != 'pi-native-event' or fact.get('type') != 'native_fault'
                        or fact.get('origin') not in ('protocol', 'controller', 'transport')
                        or not fact.get('fault_id') or not fact.get('operation_id')):
                    raise ValueError('scoped typed native fault required')
                flow = json.loads((self.root / 'workflow.json').read_text())
                if not flow.get('pending') or fact['operation_id'] != flow['pending']['id']:
                    raise ValueError('native fault differs from the pending operation')
                previous = state['faults'].get(fact['fault_id'])
                if previous is not None:
                    if previous['event_ref'] != fault_ref:
                        raise ValueError('native fault identity changed')
                    return previous['audit_ref']
                fault = fact
            if state['host_cursor'] > len(refs):
                raise ValueError('canonical host journal moved backwards')
            cursor, native_refs = self._native(state['native_cursor'])
            workflow = json.loads((self.root / 'workflow.json').read_text())
            contract = self.store.read_json(state['scope']['contract_ref'])
            report = dict(schema='rsi-project-audit-v1', contract_ref=state['scope']['contract_ref'],
                project_id=contract['spec']['project_id'], operation=workflow.get('pending'),
                stage=workflow['stage'], source_hash=workflow['source'],
                observed_at=now, elapsed_since_audit=now-state['last_audit'],
                host_refs=refs[state['host_cursor']:], native_refs=native_refs,
                native_cursor=cursor, fault_ref=fault_ref, fault=fault,
                closure='unknown', replay_allowed=False)
            reference = self.store.put_json(report)
            state.update(last_audit=now, last_clock=now, host_cursor=len(refs), native_cursor=cursor)
            state['reports'].append(reference)
            if fault:
                state['faults'][fault['fault_id']] = dict(event_ref=fault_ref, audit_ref=reference)
            return reference

    def diagnostic(self):
        if not self.path.exists():
            return None
        state = json.loads(self.path.read_text())
        current = dict(closure='unknown')
        if 'contract_ref' in state['scope']:
            workflow = json.loads((self.root / 'workflow.json').read_text())
            for ref in reversed(read_event_refs(self.root)):
                event = self.store.read_json(ref)
                if event.get('type') != 'supervision-terminal-proof':
                    continue
                proof = self.store.read_json(event['reference'])
                if (proof['project_contract_ref'] == state['scope']['contract_ref']
                        and self.store.read_json(proof['accounting_snapshot']) == workflow):
                    current = dict(closure='known', classification=proof['classification'],
                                   terminal_failure_ref=event['reference'])
                    break
        else:
            role = json.loads((self.root / 'role.json').read_text())
            current = dict(closure='role_lifecycle_authority', role_status=role['status'])
        return dict(scope=state['scope'], audit_ref=state['reports'][-1] if state['reports'] else None,
                    fault_audits=[row['audit_ref'] for row in state['faults'].values()],
                    last_audit=state['last_audit'], audit_closure='unknown', **current)

    def terminal_audit(self, pending, *, result_ref=None):
        """Called only after the runner authenticates all terminal closure proofs."""
        if not pending or not self.path.exists() or result_ref is None:
            return None
        state = json.loads(self.path.read_text())
        contract = json.loads((self.root / 'runner.json').read_text())['contract']
        refs = set(read_event_refs(self.root))
        result = self.store.read_json(result_ref)['result']
        for entry in reversed(list(state['faults'].values())):
            audit = self.store.read_json(entry['audit_ref'])
            event = self.store.read_json(entry['event_ref'])
            fact = event.get('event', {})
            if (entry['event_ref'] in refs and audit['contract_ref'] == contract
                    and entry['event_ref'] in result['events']
                    and audit['operation'] == pending and audit['fault'] == fact
                    and audit['fault_ref'] == entry['event_ref']
                    and event.get('type') == 'pi-native-event'
                    and fact.get('type') == 'native_fault'
                    and fact.get('operation_id') == pending['id']
                    and fact.get('origin') in ('controller', 'protocol')):
                return entry['audit_ref']
        return None


def retain_terminal(runner, proof):
    """Publish the already-authenticated terminal proof for read-only UI consumers."""
    if proof['classification'] != 'harness_failure':
        return
    reference = runner.store.put_json(proof)
    if not any(event['type'] == 'supervision-terminal-proof' and event['reference'] == reference
               for event in runner.events):
        runner.append(dict(type='supervision-terminal-proof', reference=reference))


class RoleSupervision(ProjectSupervision):
    """Audit a native role's logs; role lifecycle retains closure authority."""
    def __init__(self, root, *, store, scope, host_rows, clock=time.time):
        super().__init__(root, clock=clock)
        self.store, self.scope, self.host_rows = store, scope, host_rows

    def _identity(self):
        retained = json.loads((self.root / 'role.json').read_text())
        if any(retained[key] != self.scope[key] for key in ('operation_id', 'request_ref')):
            raise ValueError('role audit differs from retained role admission')
        return self.scope

    def _native_path(self):
        return self.root / 'native' / 'native-events.jsonl'

    def poll(self, *, fault_ref=None):
        with self._state() as (state, now):
            fault = None
            if fault_ref is not None:
                if not any(row['event'] == 'role_native_event'
                           and row['payload'].get('artifact_ref') == fault_ref for row in self.host_rows):
                    raise ValueError('role fault must belong to its host lifecycle journal')
                fault = self.store.read_json(fault_ref)
                if (fault.get('type') != 'native_fault' or not fault.get('fault_id')
                        or fault.get('operation_id') != self.scope['operation_id']
                        or fault.get('origin') not in ('protocol', 'controller', 'transport')):
                    raise ValueError('typed role fault differs from role operation')
                previous = state['faults'].get(fault['fault_id'])
                if previous:
                    if previous['event_ref'] != fault_ref:
                        raise ValueError('role fault identity changed')
                    return previous['audit_ref']
            if fault is None and now-state['last_audit'] < 600:
                return None
            if state['host_cursor'] > len(self.host_rows):
                raise ValueError('role lifecycle journal moved backwards')
            cursor, native_refs = self._native(state['native_cursor'])
            reference = self.store.put_json(dict(schema='rsi-role-audit-v1', scope=self.scope,
                observed_at=now, elapsed_since_audit=now-state['last_audit'],
                host_refs=[self.store.put_json(row) for row in self.host_rows[state['host_cursor']:]],
                native_refs=native_refs, native_cursor=cursor, fault_ref=fault_ref, fault=fault,
                closure='role_lifecycle_authority', replay_allowed=False))
            state.update(last_audit=now, last_clock=now, host_cursor=len(self.host_rows), native_cursor=cursor)
            state['reports'].append(reference)
            if fault:
                state['faults'][fault['fault_id']] = dict(event_ref=fault_ref, audit_ref=reference)
            return reference

    def terminal_audit(self, pending, *, result_ref=None):
        raise ValueError('role audits cannot authorize project terminal failure')
