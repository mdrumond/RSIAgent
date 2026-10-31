"""Phase1 curriculum bindings for the existing scheduler, roles and project runner."""
from dataclasses import asdict, replace
import json
from pathlib import Path
import time

from core.artifacts import ContentAddressedArtifacts
from rsi.contracts import ProjectSpec
from rsi.configuration import _snapshot
from rsi.cycles import SimplificationDeferred, SimplificationResult
from rsi.project_runner import ProjectConfig
from rsi.python_execution import atomic_json


def configured(config):
    return config.get('curriculum', {}).get('enabled') is True


class Phase1Recovery:
    """Host bindings only: no separate actor or controller execution loop."""
    def __init__(self, cycle):
        self.cycle = cycle
        self.researcher = cycle.services.curriculum_researcher
        self.materializer = cycle.services.curriculum_materializer
        self.query_adapter = None

    def project_config(self, spec):
        row = self.cycle.scheduler.state.get('derived_projects', {}).get(spec.project_id)
        if row:
            config = ProjectConfig.from_dict(self.cycle.store.read_json(row['config_ref']))
            if tuple(config.commands['test']) != spec.command:
                raise ValueError('derived command differs from its retained config')
            return config
        return self.cycle.services.project_config(self.cycle.config, spec)

    def observed_proof(self, admitted, facts):
        proof = dict(facts.get('proof_scope', {}))
        if proof.get('provider_settled') is not True:
            origin = facts['origin_task_archive']
            if origin['scope'] != 'origin-task-archive':
                raise ValueError('historical archive scope required')
            archive = ContentAddressedArtifacts(Path(origin['instance']) / 'artifacts')
            packet = archive.read_json(origin['service_result'])
            if (packet['operation_id'] != admitted['operation_id'] or packet['original_id'] != admitted['original_id']
                    or packet['result']['source_hash'] != facts['source_hash']
                    or packet['result']['contract_ref'] != origin['original_contract']):
                raise ValueError('historical host outcome differs from the failed-parent admission')
            closure = archive.read_json(packet['result']['actor_closure_ref'])
            if closure['source_hash'] != facts['source_hash'] or closure['contract_ref'] != origin['original_contract']:
                raise ValueError('historical closure does not authenticate the admitted source')
            proof.update({key: closure['observation'][key] for key in
                ('actor_closed', 'actor_absent', 'provider_settled', 'accounting_unknown') if key in closure['observation']})
        return proof

    def seed(self, references, *, only_derived=False):
        if not references:
            if only_derived:
                raise ValueError('curriculum-only admission requires retained failed parents')
            return
        rows = []
        for reference in references:
            admitted = self.cycle.store.read_json(reference)
            facts = self.cycle.store.read_json(admitted['host_failure_ref'])
            spec = ProjectSpec.from_dict(admitted['project'])
            proof = self.observed_proof(admitted, facts)
            if (facts['project_id'] != spec.project_id or facts['outcome'] not in ('failed', 'budget_exhausted', 'rejected')
                    or proof.get('provider_settled') is not True or proof.get('accounting_unknown') is True
                    or not (proof.get('actor_closed') is True or proof.get('actor_absent') is True)):
                raise ValueError('failed-parent facts differ from the admitted project')
            observations = self.cycle.store.read_json(facts['observations_ref'])
            applicability = observations['applicability']
            if applicability['project_id'] != spec.project_id or applicability['final_source_hash'] != facts['source_hash']:
                raise ValueError('admitted observations differ from the failed source')
            for artifact in facts.get('artifact_refs', ()):
                self.cycle.store.read_bytes(artifact)
            facts = dict(facts, proof_scope=proof)
            host_failure_ref = self.cycle.store.put_json(facts)
            rows.append(dict(project=spec.to_dict(), original_id=admitted['original_id'],
                operation_id=admitted['operation_id'], outcome=facts['outcome'],
                artifact_ref=host_failure_ref, host_failure_ref=host_failure_ref,
                original_actor_seconds_total=facts.get('original_actor_seconds_total', facts.get('actor_seconds_total')),
                parent_config=admitted['parent_config'], actor_seconds=0.))
        receipt = self.cycle.store.put_json(dict(failed_parent_refs=list(references), only_derived=only_derived))
        if only_derived:
            self.cycle.scheduler.admit_failed_parents(rows, receipt)
        else:
            existing = self.cycle.scheduler.state.get('historical_failed_parents')
            if existing is not None and existing != rows:
                raise ValueError('retained failed-parent catalog differs')
            self.cycle.scheduler.state['historical_failed_parents'] = rows
            self.cycle.scheduler._save()

    def executor_factory(self, spec):
        factory = self.cycle.services.executor_factory
        row = self.cycle.scheduler.state.get('derived_projects', {}).get(spec.project_id)
        binding = self.cycle.store.read_json(row['provenance_ref']).get('binding') if row else None
        if binding:
            if not callable(getattr(factory, 'with_project_configuration', None)):
                raise ValueError('derived target requires a scoped registered binder')
            return factory.with_project_configuration(binding)
        return factory

    def configuration_ref(self, spec):
        entry = self.cycle.scheduler.state.get('derived_projects', {}).get(spec.project_id)
        return entry['configuration_ref'] if entry else self.cycle.spec.configuration_ref

    def parent(self, row):
        c = self.cycle
        if row.get('parent_config'):
            return ProjectSpec.from_dict(row['project']), row['parent_config']
        packet = c.store.read_json(row['artifact_ref'])
        root = Path(packet['root'])
        local = ContentAddressedArtifacts(root / 'artifacts')
        frozen = local.read_json(json.loads((root / 'runner.json').read_text())['contract'])
        spec = ProjectSpec.from_dict(frozen['spec'])
        config = dict(frozen['config'], source=str(root / 'project'), verifier=str(root / 'frozen/verify.py'))
        spec = replace(spec, starting_folder=config['source'])
        binding = json.loads((root / 'service-binding.json').read_text())
        profile = binding.get('worker_profile') or config.get('executor_identity')
        if profile:
            registered = c.services.executor_factory.project_binding(profile)['configuration']
            config = dict(config=config, binding=registered)
        return spec, config

    def failure(self, row):
        """Admit facts, not the historical service envelope's SFT event graph."""
        if row.get('host_failure_ref'):
            facts = self.cycle.store.read_json(row['host_failure_ref'])
            return dict(facts, host_failure_ref=row['host_failure_ref'])
        packet = self.cycle.store.read_json(row['artifact_ref'])
        result = packet['result']
        facts = {key: result[key] for key in ('status', 'stage', 'classification', 'error',
            'source_hash', 'correctness') if key in result}
        facts.update(project_id=row['project']['project_id'], outcome=row['outcome'],
                     actor_seconds_total=packet['actor_seconds_total'])
        if packet.get('failure_proof_ref'):
            proof = self.cycle.store.read_json(packet['failure_proof_ref'])
            facts['proof_scope'] = {key: proof[key] for key in ('classification', 'error',
                'source_hash', 'actor_seconds_total', 'provider_settled', 'accounting_unknown') if key in proof}
        closure_ref = proof['closure_ref'] if packet.get('failure_proof_ref') else result['actor_closure_ref']
        closure = self.cycle.store.read_json(closure_ref)['observation']
        facts.setdefault('proof_scope', {}).update({key: closure[key] for key in
            ('actor_closed', 'actor_absent', 'provider_settled', 'accounting_unknown') if key in closure})
        from rsi.observations import capture
        observations, summary = capture(self.cycle.open_project(row['operation_id']), self.cycle.store, row['outcome'])
        facts.update(observations_ref=self.cycle.store.put_json(observations), summary=summary,
                     artifact_refs=observations['artifact_refs'])
        if 'proof_scope' in facts:
            facts['artifact_refs'].append(self.cycle.store.put_json(facts['proof_scope']))
        facts['origin_task_archive'] = dict(scope='origin-task-archive', service_result=row['artifact_ref'])
        reference = self.cycle.store.put_json(facts)
        return dict(facts, host_failure_ref=reference)

    def remaining(self, context, spent):
        remaining = self.cycle.scheduler.remaining()
        for key in ('actor_seconds', 'recovery_seconds'):
            limit = context['remaining'][key]
            if limit is not None:
                remaining[key] = max(0., min(remaining[key] if remaining[key] is not None else limit, limit - spent))
        return remaining

    def simplify(self, parent, failure, context):
        from rsi.curriculum_materialization import capability_catalog, materialize_child, MaterializationRejected
        from rsi.curriculum_research import Researcher, ResearchRejected, DiagnosticProposal
        c = self.cycle
        root = c.root / 'curriculum' / context['operation_id']
        root.mkdir(parents=True, exist_ok=True)
        path = root / 'integration.json'
        state = json.loads(path.read_text()) if path.exists() else dict(history=[], rejections=[])
        if state.get('complete'):
            return self.result(state)
        # Eager BGE loading is setup and precedes actor-deadline anchoring.
        self.query_adapter = c.query(c.scheduler.state['memory'])
        if self.researcher is None:
            self.researcher = Researcher(c.roles(), self.query, c.store)
        parents = []
        inputs = {}
        failures = {}
        for row in c.scheduler.state['recovery_queue']:
            spec, config = self.parent(row['result'])
            inputs[spec.project_id] = (spec, config)
            failures[spec.project_id] = self.failure(row['result'])
            parents.append(dict(project_id=spec.project_id, spec=spec.to_dict(),
                failure=failures[spec.project_id], templates=capability_catalog(spec, config)))
        while True:
            spent = sum(item['actor_seconds'] for item in state['history'])
            remaining = self.remaining(context, spent)
            bounds = [remaining[key] for key in ('actor_seconds', 'recovery_seconds') if remaining[key] is not None]
            if not bounds or any(value <= 0 for value in bounds) or remaining['wall_seconds'] == 0:
                state.update(complete=True, deferred=True)
                break
            operation = context['operation_id'] + '-research-' + str(len(state['history']))
            selected = state.get('selected')
            if selected is None:
                scoped = dict(context, operation_id=operation, root=root,
                    deadline=time.monotonic() + min(bounds), remaining=remaining,
                    wall_deadline=None if remaining['wall_seconds'] is None else time.monotonic() + remaining['wall_seconds'],
                    failed_parents=parents, templates=parents[0]['templates'],
                    memory_ref=c.scheduler.state['memory'], skills_ref=c.scheduler.state['skills'],
                    configuration_ref=c.spec.configuration_ref, previous_rejections=state['rejections'],
                    user_guidance={role: c.config.get('prompts', {}).get(role, '') for role in ('curriculum', 'waves')})
                atomic_json(path, state)  # Producer owns durable pending/completed research receipts.
                previous_operation = c.role_operation
                c.role_operation = operation
                try:
                    result = self.researcher.research(parent, failures[parent.project_id], scoped)
                except ResearchRejected as error:
                    state['history'].append(dict(receipt_ref=error.receipt_ref, actor_seconds=error.actor_seconds))
                    state['rejections'].append(error.receipt_ref)
                    atomic_json(path, state)
                    continue
                finally:
                    c.role_operation = previous_operation
                state['history'].append(dict(receipt_ref=result.receipt_ref, actor_seconds=result.actor_seconds))
                if result.proposal is None:
                    state.update(complete=True, deferred=True)
                    break
                selected = dict(proposal=asdict(result.proposal), evidence_refs=list(result.evidence_refs),
                                operation_id=operation, research_receipt_ref=result.receipt_ref)
                state['selected'] = selected
                atomic_json(path, state)
            proposal = DiagnosticProposal(**selected['proposal'])
            chosen = proposal.parent_project_id or parent.project_id
            if chosen not in inputs:
                raise ValueError('curriculum selected a parent outside the admitted catalog')
            spec, config = inputs[chosen]
            def remaining():
                limits = self.remaining(context, sum(item['actor_seconds'] for item in state['history']))
                cap = c.config['curriculum'].get('project_actor_seconds')
                if cap is not None:
                    limits['actor_seconds'] = min(cap, limits['actor_seconds']) if limits['actor_seconds'] is not None else cap
                return limits
            try:
                with c.services.setup.measure('curriculum-materialization', context['operation_id']):
                    child = (self.materializer or materialize_child)(spec, config, proposal,
                        root=root / (selected['operation_id'] + '-child'), remaining=remaining,
                        evidence_refs=selected['evidence_refs'], operation_id=selected['operation_id'])
            except MaterializationRejected as error:
                state['rejections'].append(c.store.put_json(dict(status='rejected', error=str(error))))
                state.pop('selected')
                atomic_json(path, state)
                continue
            config = asdict(child.config) if isinstance(child.config, ProjectConfig) else child.config
            if child.spec.project_id == chosen or tuple(config['commands']['test']) != child.spec.command:
                raise ValueError('new linked identity and matching runnable command required')
            derived = dict(c.config, project_configs={**c.config.get('project_configs', {}), child.spec.project_id: config})
            derived['prompts'] = dict(c.config.get('prompts', {}))
            additions = dict(actor=child.spec.goal + '\nAcceptance: ' + proposal.acceptance,
                curriculum='This is a newly admitted linked diagnostic; investigate admitted documentation and evidence.',
                waves='Execute this frozen derived project; historical parent cases and outcomes remain unchanged.')
            for role, guidance in additions.items():
                derived['prompts'][role] = derived['prompts'].get(role, '') + '\n' + guidance
            research = self.researcher.actor_context(selected['research_receipt_ref'])
            rendered = json.dumps(research, ensure_ascii=False)
            if len(rendered.encode()) > 4096:
                raise ValueError('curriculum actor handoff exceeds its bounded wire contract')
            derived['prompts']['actor'] += '\nCurriculum research:\n' + rendered
            provenance_ref = c.store.put_json(child.provenance)
            receipt = child.provenance['receipt']
            if c.store.put_bytes(Path(receipt['path']).read_bytes()) != receipt['sha256']:
                raise ValueError('materialization receipt differs from its retained hash')
            derived['curriculum_context'] = dict(parent_project_id=chosen, project_id=child.spec.project_id,
                allocation_class='semantic-diagnostic-recovery', research=research,
                artifact_refs=[provenance_ref, receipt['sha256'],
                    *selected['evidence_refs'], *(item['receipt_ref'] for item in state['history'])])
            entry = dict(config_ref=c.store.put_json(config), provenance_ref=provenance_ref,
                         configuration_ref=c.store.put_json(_snapshot(derived)))
            entry['allocation_ref'] = c.store.put_json(dict(allocation_class='semantic-diagnostic-recovery',
                configured_pool=c.spec.configuration_ref, parent_project_id=chosen,
                operation_id=context['operation_id'], actor_seconds=child.spec.budgets.actor_seconds,
                research_actor_seconds=sum(item['actor_seconds'] for item in state['history'])))
            c.scheduler.state.setdefault('derived_projects', {})[child.spec.project_id] = entry
            c.scheduler._save()
            state.update(complete=True, spec=child.spec.to_dict(), parent_project_id=chosen, provenance=entry)
            break
        state['actor_seconds'] = sum(item['actor_seconds'] for item in state['history'])
        state['receipt_ref'] = c.store.put_json({key: value for key, value in state.items() if key != 'receipt_ref'})
        atomic_json(path, state)
        return self.result(state)

    @staticmethod
    def result(state):
        if state.get('deferred'):
            return SimplificationDeferred(state['actor_seconds'], state['receipt_ref'])
        return SimplificationResult(ProjectSpec.from_dict(state['spec']), state['actor_seconds'],
                                    state['receipt_ref'], state['parent_project_id'])

    def retained(self, operation):
        from rsi.runtime import WorkflowError
        path = self.cycle.root / 'curriculum' / operation / 'integration.json'
        state = json.loads(path.read_text()) if path.exists() else {}
        if not state.get('complete'):
            raise WorkflowError('curriculum producer is pending; observe its retained research/materialization receipts')
        receipt = self.cycle.store.read_json(state['receipt_ref'])
        if receipt != {key: value for key, value in state.items() if key != 'receipt_ref'}:
            raise ValueError('curriculum result differs from its authoritative receipt')
        return self.result(state)

    def query(self, payload, *, deadline):
        if time.monotonic() >= deadline:
            raise TimeoutError('research retrieval allocation exhausted')
        result = self.query_adapter(payload)
        if time.monotonic() >= deadline:
            raise TimeoutError('research retrieval allocation exhausted')
        return result
