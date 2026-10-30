"""Production heldout factory bindings for the existing tuning controller."""
import json

from rsi.contracts import BudgetLimits, ProjectSpec
from rsi.evaluation import SkillEvaluation, case_identity
from rsi.project_runner import ProjectConfig
from rsi.skills import SkillBundle


class TuningBindings:
    def tuning_settings(self):
        settings = self.config.get('tuning', {})
        for name, default in (('heldout_projects', 5), ('runs_per_arm', 2)):
            if type(settings.get(name, default)) is not int or settings.get(name, default) <= 0:
                raise ValueError('positive heldout count and repetitions required')
        cases = settings.get('heldout_cases', [])
        if len(cases) != settings.get('heldout_projects', 5):
            raise ValueError('configure independently selected tuning.heldout_cases matching heldout_projects')
        for case in cases:
            project = ProjectSpec.from_dict(case['project'])
            config = dict(self.config, project_configs={project.project_id: case['execution']})
            self.services.project_config(config, project)
        return settings, cases

    def tuning_candidate(self, event):
        incumbent = self.scheduler.state['skills']
        candidates = [ref for ref, row in self.learning.state['candidates'].items()
                      if row['incumbent'] == incumbent and not row['quarantined']]
        if not candidates:
            raise ValueError('no evidence-supported candidate for active skill bundle')
        return candidates[-1]

    def request_tuning(self, steering):
        self.tuning_settings()  # Fail before enqueueing work when curriculum is absent.
        return self.tuning.request(steering)

    def tuning_project(self, event):
        rows = self.scheduler.state['originals']
        requested = event.get('steering', {}).get('project_ref')
        selected = [row for row in rows if not requested or requested in (
            row['original_id'], row['artifact_ref'], row['operation_id'])]
        if not selected:
            raise ValueError('manual tuning needs an observed project, not invented outcome evidence')
        return selected[-1]

    def checkpoint(self, path, spec, config, memory, skills):
        binding = self.store.put_json(dict(checkpoint=spec.to_dict(), memory=memory,
            skills=skills, configuration=self.spec.configuration_ref, kdb=self.binding['kdb']))
        return self.runner(path, spec=spec, config=config,
            bundle=SkillBundle.from_dict(self.store.read_json(skills)), memory_ref=memory,
            backend_identity=binding, reopen=(path / 'runner.json').exists())

    def evaluation(self, event, operation, remaining):
        root = self.root / 'evaluations' / operation
        path = root / 'evaluation.json'
        if path.exists():
            return SkillEvaluation(path, artifacts=self.store, runner_factory=self.trial)
        settings, cases = self.tuning_settings()
        memory, incumbent = self.scheduler.state['memory'], self.scheduler.state['skills']
        matching = [ref for ref, record in self.learning.state['candidates'].items()
                    if record['incumbent'] == incumbent and not record['quarantined']]
        if event['kind'] == 'manual_tuning' or not matching:
            # User guidance steers a real writer on admitted project evidence.
            row = self.tuning_project(event)
            writer_id = operation + '-writer'
            self.tuning_guidance = event.get('steering', {})
            try:
                update = self.write_skills(incumbent, memory, row,
                    dict(operation_id=writer_id, remaining=self.scheduler.remaining()))
            finally:
                self.tuning_guidance = {}
            if not any(e.get('id') == writer_id for e in self.scheduler.state['events']):
                self.scheduler.state['events'].append(dict(kind='manual_writer', id=writer_id))
                self.scheduler._save()
            self.scheduler.complete_extension(writer_id, update.reference, actor_seconds=update.actor_seconds)
            candidate = update.reference
        else:
            candidate = matching[-1]
        training = []
        admitted = {r['project']['project_id']: r for r in self.scheduler.state['originals']}
        for i, project in enumerate(self.spec.projects):
            row = admitted.get(project.project_id)
            if row:
                runner = self.open_project(row['operation_id'])
                initial = runner.root / 'training-identity.json'
                if initial.exists():
                    training.append(json.loads(initial.read_text()))
            else:
                runner = self.checkpoint(root / 'training' / str(i), project,
                    self.services.project_config(self.config, project), memory, incumbent)
            training.append(case_identity(runner))
        heldouts = []
        for i, case in enumerate(cases):
            project = ProjectSpec.from_dict(case['project'])
            local = dict(self.config, project_configs={project.project_id: case['execution']})
            heldouts.append(self.checkpoint(root / 'heldouts' / str(i), project,
                self.services.project_config(local, project), memory, incumbent))
        return SkillEvaluation.create(path, checkpoints=heldouts, training=training,
            artifacts=self.store, memory_ref=memory, kdb_ref=self.binding['kdb'],
            model_config=self.config['provider'], incumbent=incumbent, candidate=candidate,
            joint_limits=BudgetLimits.from_dict(settings.get('budgets', self.config['budgets'])),
            remaining_parent=self.scheduler.remaining(), runner_factory=self.trial,
            case_count=settings.get('heldout_projects', 5), repetitions=settings.get('runs_per_arm', 2),
            steering=event.get('steering', {}))

    def suite_validation(self, incumbent, candidate, cases, operation):
        for path in (self.root / 'evaluations').glob('*/evaluation.json'):
            evaluation = SkillEvaluation(path, artifacts=self.store, runner_factory=self.trial)
            result = evaluation.result()
            if result and tuple(cases) == (result['evidence_ref'],):
                return evaluation.validation(incumbent, candidate, cases, operation)
        return None
