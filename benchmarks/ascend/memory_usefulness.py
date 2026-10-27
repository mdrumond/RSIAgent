"""Optional, record-only paired experiments for individual reusable lessons.

Host verdicts determine outcomes. AI interpretation cannot admit a candidate,
change retrieval, or promote learning from either experiment arm.
"""
from dataclasses import asdict
from copy import copy
import time
import math
import fcntl
import hashlib
import json
from pathlib import Path

from benchmarks.a5kernels.evidence import canonical_bytes
from benchmarks.a5kernels.phase1_adaptive import AdaptiveProjectRequest
from benchmarks.a5kernels.phase1_curriculum_selector import CurriculumContext, WaveDecision
from benchmarks.a5kernels.phase1_project import materialize_inputs, evaluate_reference
from benchmarks.a5kernels.phase1_wave_runtime import SemanticProjectFailure


def digest(value):
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def actor_project(spec):
    """Keep host semantics while withholding curriculum-authored teaching prose."""
    from benchmarks.a5kernels.phase1_curriculum import ProjectSpec
    value = spec.as_dict()
    value['project_id'] = 'usefulness-project'
    value['hypothesis'] = 'Implement the specified computation and satisfy its host acceptance checks.'
    names = {}
    for index, item in enumerate(value['graph']['inputs']):
        names[item['name']] = 'input' + str(index)
        item['name'] = names[item['name']]
    for index, item in enumerate(value['graph']['nodes']):
        names[item['name']] = 'node' + str(index)
        item['name'] = names[item['name']]
        item['operands'] = [names[name] for name in item['operands']]
    value['graph']['output'] = names[value['graph']['output']]
    return ProjectSpec.from_mapping(value)


def paired_outcome(control, treatment):
    if any(arm.get('verified') is None for arm in (control, treatment)):
        return 'inconclusive'
    return {(False, True): 'demonstrated-benefit', (True, True): 'both-succeeded',
            (False, False): 'neither-succeeded', (True, False): 'treatment-regressed'}[
                control['verified'], treatment['verified']]


class UsefulnessVerifier:
    def __init__(self, root, curriculum, execute_project, verify, *, enabled=False,
                 pair_seconds=600, memory_root=None, remaining_seconds=None, clock=time.monotonic):
        if (isinstance(pair_seconds, bool) or not isinstance(pair_seconds, (int, float))
                or not math.isfinite(pair_seconds) or pair_seconds <= 0):
            raise ValueError('pair allowance must be finite and positive')
        self.root, self.curriculum = Path(root), curriculum
        self.execute_project, self.verify = execute_project, verify
        self.enabled, self.pair_seconds = enabled, pair_seconds
        self.memory_root = memory_root
        self.remaining_seconds, self.clock = remaining_seconds, clock
        self._experiment_remaining = None

    def _append(self, directory, kind, **body):
        events = list(directory.glob('[0-9]*.json'))
        sequence = max((int(path.stem) for path in events if path.stem.isdigit()), default=-1) + 1
        event = {'schema': 'memory-usefulness-event-v1', 'kind': kind, **body}
        with (directory / f'{sequence:04d}.json').open('x') as stream:
            json.dump(event, stream, sort_keys=True, allow_nan=False)
            stream.flush()
        return event

    def assess_memory(self, memory, prior_memory, source_id):
        if not self.enabled:
            return []
        return [self._assess(asdict(lesson), prior_memory, source_id,
                            [asdict(fact) for fact in memory.host_facts])
                for lesson in memory.lessons]

    def assess_lesson(self, lesson, prior_memory, source_id, *, host_facts=()):
        """Assess a producer's reusable lesson without changing its stored bytes."""
        if not self.enabled:
            return None
        return self._assess(dict(lesson), prior_memory, source_id, list(host_facts))

    def _assess(self, lesson, prior_memory, source_id, host_facts):
        lesson_id = digest(lesson)
        experiment_id = digest({'lesson_id': lesson_id, 'source_id': source_id,
                                'prior_memory_sha256': hashlib.sha256(prior_memory).hexdigest(),
                                'pair_seconds': self.pair_seconds})
        directory = self.root / experiment_id
        directory.mkdir(parents=True, exist_ok=True)
        with (directory / '.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            events, unreadable = [], []
            for path in sorted(directory.glob('[0-9]*.json')):
                try:
                    event = json.loads(path.read_bytes())
                    if not isinstance(event, dict) or not isinstance(event.get('kind'), str):
                        raise ValueError('Invalid usefulness event')
                    events.append(event)
                except (OSError, ValueError):
                    unreadable.append(path.name)
            if events or unreadable:
                if events and events[-1]['kind'] == 'assessment':
                    return events[-1]
                # A retained operation might still be running. Never duplicate it.
                return self._append(directory, 'assessment', experiment_id=experiment_id,
                    lesson_id=lesson_id, outcome='inconclusive',
                    reason='Interrupted experiment; inspect retained arm operations before any new experiment.',
                    unreadable_events=unreadable)
            self._append(directory, 'started', experiment_id=experiment_id,
                         lesson_id=lesson_id, source_id=source_id, lesson=lesson,
                         prior_memory_sha256=hashlib.sha256(prior_memory).hexdigest(),
                         pair_seconds=self.pair_seconds, arm_seconds=self.pair_seconds / 2)
            deadline = self.clock() + self.pair_seconds
            arms = {}
            def remaining():
                available = max(0, deadline - self.clock())
                return min(available, max(0, self.remaining_seconds())) if self.remaining_seconds else available
            def require_time():
                available = remaining()
                if available <= 0:
                    raise TimeoutError('Memory usefulness active-time allowance exhausted')
                return available
            self._experiment_remaining = require_time
            try:
                require_time()
                inspection = self.verify('inspect', {'lesson': lesson, 'host_facts': host_facts})
                context = CurriculumContext(
                    objective=('Create exactly ONE SMALL project that should benefit from this lesson. '
                        f'Total A/B pair allowance is {self.pair_seconds} active seconds; each fresh actor '
                        f'has {self.pair_seconds / 2} seconds. Keep the project small enough for that allowance. '
                        'Freeze normal host acceptance criteria. Lesson: ' + json.dumps(lesson)),
                    wave_index=1, pre_wave_memory_sha256=hashlib.sha256(prior_memory).hexdigest(),
                    pre_wave_coverage=(), prior_outcomes=(), remaining_projects=1,
                    remaining_seconds=require_time())
                require_time()
                curriculum = copy(self.curriculum)
                if callable(getattr(curriculum, '_complete', None)):
                    complete = curriculum._complete
                    def guarded_complete(messages):
                        require_time()
                        return complete(messages)
                    curriculum._complete = guarded_complete
                decision, _ = curriculum.select(context, remaining_projects=1)
                if not isinstance(decision, WaveDecision) or len(decision.wave.projects) != 1:
                    raise ValueError('usefulness curriculum must generate exactly one project')
                authored = decision.wave.projects[0].spec
                spec = actor_project(authored)
                self._append(directory, 'project-frozen', project=spec.as_dict(),
                             curriculum_project=authored.as_dict(),
                             project_sha256=digest(spec.as_dict()), inspection=inspection)
                inputs = materialize_inputs(spec)
                oracle = evaluate_reference(spec, inputs)
                arm_allowance = min(self.pair_seconds / 2, require_time() / 2)
                for name in ('control', 'treatment'):
                    admitted_seconds = min(arm_allowance, require_time())
                    raw = prior_memory
                    if name == 'treatment':
                        envelope = json.loads(prior_memory)
                        envelope['usefulness_lesson'] = lesson
                        raw = canonical_bytes(envelope)
                    branch = directory / name
                    branch.mkdir(exist_ok=False)
                    (branch / 'memory.bin').write_bytes(raw)
                    self._append(directory, 'arm-started', arm=name,
                                 memory_sha256=hashlib.sha256(raw).hexdigest(), admitted_seconds=admitted_seconds)
                    request = AdaptiveProjectRequest(spec, 1, raw, branch, inputs, oracle,
                        memory_root=self.memory_root, remaining_seconds=admitted_seconds,
                        publish_learning=False)
                    try:
                        result = self.execute_project(request)
                        arms[name] = {'verified': result.evidence.verified,
                            'classification': result.evidence.classification.value,
                            'detail': result.evidence.detail,
                            'evidence_sha256': hashlib.sha256(result.evidence_payload).hexdigest()}
                        (branch / 'evidence.json').write_bytes(result.evidence_payload)
                    except SemanticProjectFailure as error:
                        payload = canonical_bytes({'verified': False, 'detail': error.summary,
                                                   'evidence': error.evidence})
                        (branch / 'evidence.json').write_bytes(payload)
                        arms[name] = {'verified': False, 'detail': error.summary, 'evidence': error.evidence,
                                      'evidence_sha256': hashlib.sha256(payload).hexdigest()}
                    self._append(directory, 'arm-completed', arm=name, **arms[name])
                outcome = paired_outcome(arms['control'], arms['treatment'])
                interpretation, interpretation_error = None, None
                try:
                    require_time()
                    interpretation = self.verify('interpret', {'lesson': lesson, 'arms': arms,
                        'outcome': outcome, 'warning': 'Both succeeding does not establish uselessness.'})
                    require_time()
                except Exception as error:
                    interpretation_error = dict(error_type=type(error).__name__, reason=str(error))
                return self._append(directory, 'assessment', experiment_id=experiment_id,
                    lesson_id=lesson_id, outcome=outcome, arms=arms, interpretation=interpretation,
                    interpretation_error=interpretation_error)
            except Exception as error:
                # Assessment is best effort and never gates original publication.
                return self._append(directory, 'assessment', experiment_id=experiment_id,
                    lesson_id=lesson_id, outcome='inconclusive', reason=str(error), arms=arms,
                    error_type=type(error).__name__)
            finally:
                self._experiment_remaining = None


def build_usefulness_verifier(*, root, provider, profile, curriculum, project_actor,
                             enabled=False, pair_seconds=600, remaining_seconds=None):
    """Bind production curriculum, provider and host-verified actor execution."""
    from copy import copy
    curriculum = copy(curriculum)
    curriculum.failure_feedback = None
    def allowance():
        available = (verifier._experiment_remaining() if verifier._experiment_remaining
                     else min(pair_seconds, verifier.remaining_seconds()) if verifier.remaining_seconds
                     else pair_seconds)
        if available <= 0:
            raise TimeoutError('Memory usefulness parent allowance exhausted')
        return available
    def complete(messages):
        return provider.complete(profile, messages, max_seconds=allowance()).text
    curriculum._complete = complete
    def verify(stage, evidence):
        return complete((
            {'role': 'system', 'content': 'You are the memory usefulness verifier. Inspect retained '
             'host evidence and explain applicability and uncertainty. Host verification is authoritative. '
             'Record usefulness only; never reject memory or claim both arms succeeding proves uselessness.'},
            {'role': 'user', 'content': json.dumps({'stage': stage, **evidence}, sort_keys=True)},
        ))
    verifier = UsefulnessVerifier(Path(root) / 'memory-usefulness', curriculum,
        project_actor, verify, enabled=enabled, pair_seconds=pair_seconds, memory_root=Path(root), remaining_seconds=remaining_seconds)
    return verifier
