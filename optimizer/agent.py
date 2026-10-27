"""Pi coding adapter; host stages and frozen reference contracts remain authoritative."""
import hashlib

from benchmarks.a3_experiments import BackendModel
from benchmarks.a3_model_profiles import load_a3_model_profile
from coding.pi import PiCodingSession, ENGINE_IDENTITY, runtime_preflight
from coding.pi.adapters import OPTIMIZER_SCHEMAS
from benchmarks.ascend.status_descriptions import explain_status


SYSTEM = """Optimize deterministic inference Torch code using Ascend C on A3.
Use the typed tool allowed in the current stage. Never alter the supplied
representatives, reference source, initialization, or frozen comparison policy.
Use compile for host compilation feedback without spending an evaluated candidate.
Read retained evidence with read-artifact using its exact 64-character SHA256 and
next_offset; use read-memory with an admitted index when the digest is unknown.
Invalid reads return host feedback. In single-evaluation mode, submit one candidate
after compile repairs; finish alone does not establish correctness.
Candidates define extern \"C\" __global__ __aicore__ void optimizer_kernel(...)
with GM_ADDR arguments in inputs, frozen state, outputs order. The host defines
INPUTi/STATEi/OUTPUTi_NUMEL,RANK,SIZEj,STRIDEj and scalar ARGi constants.
Declare KERNEL_TASK_TYPE_DEFAULT inside optimizer_kernel: KERNEL_TYPE_AIV_ONLY
for pure vector/scalar work, KERNEL_TYPE_AIC_ONLY for pure Cube work, or the
appropriate supported mixed type. Use only the supplied native contracts and
guide/KDB evidence. Every frozen case must pass host verification before timing.
Profiling analysis runs remotely; consume the supplied diagnostic digests, not raw
report trees. Digests cover explicitly selected cases/groups, not the full suite.
Missing metrics are unavailable, not zero. Keep canonical timing separate from
diagnostic replay data; do not average unrelated kernel/pipe ratios. Full raw and
normalized reports remain at the retained remote paths; read-artifact only accesses
digests/evidence already retained locally, not arbitrary remote files.
Summarize changes briefly; do not provide private chain-of-thought."""


class OptimizerAgent:
    def __init__(self, *, guide, model=BackendModel.DEEPSEEK_FLASH, session_factory=PiCodingSession,
                 max_provider_calls=200, budget=None):
        self.guide = guide
        self.model = model
        if session_factory is PiCodingSession:
            runtime_preflight()
        self.session_factory = session_factory
        self.session = None
        self.last_source = None
        self.last_block_count = 1
        self.own_process_group = True
        self.deadline = None
        self.identity = {**load_a3_model_profile(model).as_dict(), **ENGINE_IDENTITY,
                         "budget_policy": "wall-time-v1",
                         "guide_sha256": hashlib.sha256(guide.encode()).hexdigest()}

    def close(self):
        if self.session is not None:
            self.session.close()
            self.session = None

    def complete(self, stage, context, record):
        if stage == 'cases':
            directive = ('CURRENT STAGE: cases. Call cases with signatures and summary. Propose additional signatures only; '
                'never change the supplied representatives, source or initialization. '
                f'The configured max_cases is {context["max_cases"]}; remaining_signatures is '
                f'{context["remaining_signatures"]}. Do not propose more than this remaining count. '
                'If it is zero, return signatures: []. Do not return candidate/compile/tools/finish.')
        elif stage == 'optimize':
            directive = ('CURRENT STAGE: optimize. The reference and complete case suite are ALREADY '
                'PREPARED AND FROZEN. Do NOT call cases, propose signatures, or modify '
                'the frozen suite, reference source or reference entrypoint. Select exactly one candidate, compile, query-knowledge, '
                'read-memory, read-artifact, or finish action using the schemas below. '
                'Use the supplied frozen native contracts to implement optimizer_kernel.')
        else:
            raise ValueError('unknown optimizer model stage')
        if self.session is None:
            self.session = self.session_factory(model=self.model, schemas=OPTIMIZER_SCHEMAS,
                deadline=self.deadline, own_process_group=self.own_process_group)
        allowed = ["cases"] if stage == "cases" else [name for name in OPTIMIZER_SCHEMAS if name != "cases"]
        if context.get("knowledge_available") is False:
            allowed = [name for name in allowed if name != "query-knowledge"]
        name, args = self.session.next_action({**context, "guide": self.guide,
            "stage": stage, "stage_explanation": explain_status(stage, field="stage"), "contract": SYSTEM, "current_stage_instruction": directive}, allowed,
            record=record,
            source=self.last_source if stage == "optimize" else None,
            write_action=(lambda s: ("compile", {"source": s, "block_count": self.last_block_count,
                                                  "summary": "Targeted source correction"})) if stage == "optimize" else None)
        if name == "cases" and len(args["signatures"]) > context["remaining_signatures"]:
            raise ValueError("Pi proposed more than the remaining signatures")
        if name in ("candidate", "compile"):
            self.last_source, self.last_block_count = args["source"], args["block_count"]
        return {"action": name, **args}
