"""A3 direct-provider adapter; summaries are explicit model output, never reasoning."""
import json

from benchmarks.a3_experiments import BackendModel
from benchmarks.a3_model_profiles import complete_a3, load_a3_model_profile


SYSTEM = '''Optimize deterministic inference Torch code using Ascend C on A3.
Return exactly one JSON object for the CURRENT STAGE specified below.
Never alter supplied representatives/source/initialization or the frozen suite.
Optimization-stage actions: {"action":"candidate","source":"Ascend C device source",
"block_count":1,"summary":"change and expected effect"},
{"action":"query-knowledge","query":"..."},
{"action":"read-memory","index":0}, or {"action":"finish","summary":"..."}.
Read full retained evidence or truncated memory using
{"action":"read-artifact","sha256":"artifact digest","offset":0}; copy the exact
64-character lowercase hexadecimal sha256 from a retained artifact record, never
abbreviate or guess it. If unsure, use read-memory with an index from memory_summary
to retrieve the entry and its exact artifact references. Invalid or missing reads
return tool-error feedback; repair the request rather than repeating it. Advance
the returned next_offset to page through JSON text (1,500 characters per page).
Compile without executing or spending an evaluated candidate trial with
{"action":"compile","source":"Ascend C device source","block_count":1,"summary":"..."}.
In Phase 2 single-evaluation mode, submit one candidate after any compile repairs;
finish without a candidate does not establish correctness. Curriculum feedback and
updated memory, when supplied, describe learning since this task was frozen.
Candidates define extern "C" __global__ __aicore__ void optimizer_kernel(...)
with GM_ADDR arguments in inputs, frozen state, outputs order. Host defines
Declare the kernel task type explicitly inside optimizer_kernel with
KERNEL_TASK_TYPE_DEFAULT: KERNEL_TYPE_AIV_ONLY for pure vector/scalar work,
KERNEL_TYPE_AIC_ONLY for pure Cube work, or the appropriate supported mixed type.
INPUTi/STATEi/OUTPUTi_NUMEL,RANK,SIZEj,STRIDEj and scalar ARGi constants.
Use only supplied native contracts and guide/KDB evidence. Every frozen case must
pass the host comparison policy before timing. Never remove cases or loosen policy.
Summarize decisions briefly; do not provide private chain-of-thought.'''


class OptimizerAgent:
    def __init__(self, *, guide, model=BackendModel.GPT_5_6_SOL, complete=complete_a3):
        self.guide = guide
        self.model = model
        self._complete = complete
        self.identity = load_a3_model_profile(model).as_dict()

    def complete(self, stage, context, record):
        if stage == 'cases':
            directive = ('CURRENT STAGE: cases. Return ONLY {"action":"cases","signatures":[],'
                '"summary":"why these additional cases matter"}. Propose additional signatures only; '
                'never change the supplied representatives, source or initialization. '
                f'The configured max_cases is {context["max_cases"]}; remaining_signatures is '
                f'{context["remaining_signatures"]}. Do not propose more than this remaining count. '
                'If it is zero, return signatures: []. Do not return candidate/compile/tools/finish.')
        elif stage == 'optimize':
            directive = ('CURRENT STAGE: optimize. The reference and complete case suite are ALREADY '
                'PREPARED AND FROZEN. Do NOT return action:"cases", propose signatures, or modify '
                'the frozen suite, reference source or reference entrypoint. Return exactly one candidate, compile, query-knowledge, '
                'read-memory, read-artifact, or finish action using the schemas below. '
                'Use the supplied frozen native contracts to implement optimizer_kernel.')
        else:
            raise ValueError('unknown optimizer model stage')
        profile = load_a3_model_profile(self.model)
        system = directive + '\n\n' + SYSTEM
        user = json.dumps({**context, 'stage': stage, 'guide': self.guide,
                           'current_stage_instruction': directive}, allow_nan=False)
        record('request', profile.request(system, user))
        completion = self._complete(system, user, model=self.model)
        record('response', completion.as_dict())
        return json.loads(completion.text)
