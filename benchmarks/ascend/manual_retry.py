"""One explicit original Phase 1 acceptance retry using reviewed campaign memory."""
from pathlib import Path

from .automatic_remediation import AutomaticRemediation
from .phase2_feedback import FeedbackStore
from .phase2_time import Phase2TimeBudget


class ManualPhase1Retry:
    def __init__(self, runner, job, selection):
        from benchmarks.a5kernels.phase1_curriculum import ProjectSpec
        self.runner, self.job = runner, job
        self.row = selection['selected'][0]
        ProjectSpec.from_mapping(self.row['outcome']['original_spec'])
        self.root = Path(job['root'])
        self.store = FeedbackStore(job['feedback_root'])
        self.time = Phase2TimeBudget(self.root, job['phase2_max_seconds'])
        runner._remaining_seconds = lambda: self.time.remaining
        # Reuse the same original-spec execution and evidence import as automatic remediation.
        self.retry = object.__new__(AutomaticRemediation)
        self.retry.runner, self.retry.target, self.retry.store = runner, job['target'], self.store

    def resume(self):
        if not (self.root / 'retry-memory.bin').exists() and self.time.remaining <= 0:
            raise TimeoutError('Original project retry allowance exhausted')
        with self.time.active():
            state = self.runner._load_state()
            if "original-retry" not in state["completed_waves"]:
                self.retry._phase1_retry(self.row, self.root, state, retain_campaign=True)
            else:
                self.store.import_phase1(self.root, target=self.job["target"],
                    language="ascend-c" if self.job["target"] == "a3" else "catlass-dsl")
            records = [row for row in self.store.records()
                       if row['outcome']['source_root'] == str(self.root.resolve())]
            if not records:
                raise ValueError('Original retry did not retain a committed observation')
            for row in records:
                self.store.link_original_retry(self.row['feedback_id'], row['feedback_id'])
        return self.report()

    def report(self):
        original = next(row for row in self.store.records() if row['feedback_id'] == self.row['feedback_id'])
        return dict(phase='manual-phase1-retry', status='FIXED' if original['lifecycle'] == 'fixed' else 'FAILED',
                    feedback_id=original['feedback_id'], retries=original['investigation']['retries'])
