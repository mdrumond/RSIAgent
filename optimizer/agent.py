"""A3 direct-provider adapter; summaries are explicit model output, never reasoning."""
import json

from benchmarks.a3_experiments import BackendModel
from benchmarks.a3_model_profiles import complete_a3, load_a3_model_profile


SYSTEM = '''Optimize deterministic inference Torch code using Ascend C on A3.
Return exactly one JSON object. First stage: {"action":"cases","signatures":[],
"summary":"why these additional corner cases matter"}. Supply additional signatures
only; never alter supplied representatives/source/initialization. At most 24 total.
Optimization actions: {"action":"candidate","source":"Ascend C device source",
"block_count":1,"summary":"change and expected effect"},
{"action":"query-knowledge","query":"..."},
{"action":"read-memory","index":0}, or {"action":"finish","summary":"..."}.
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
        profile = load_a3_model_profile(self.model)
        user = json.dumps({'stage': stage, 'guide': self.guide, **context}, allow_nan=False)
        record('request', profile.request(SYSTEM, user))
        completion = self._complete(SYSTEM, user, model=self.model)
        record('response', completion.as_dict())
        return json.loads(completion.text)
