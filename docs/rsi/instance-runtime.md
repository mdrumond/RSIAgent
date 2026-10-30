# Instance runtime smoke scope

`tests/rsi/test_instance_runtime.py` checks setup target selection and failure
receipts, actual offline wheel installation and damaged-package repair, dashboard
process ownership and port allocation, and setup/job lock exclusion. Installation
probes use a tiny local wheel, not production dependency downloads. Dashboard
lifecycle fixtures explicitly reuse existing test dependencies through their own
venv configuration; they do not establish a full isolated production dependency
installation. The separate wheel import scenario runs outside the checkout with
isolated Python startup.

HTTP probes connect directly to `127.0.0.1` using a proxy-free urllib opener.
The single-request server and reader each have finite waits. Reader failures are
reported as test failures, rather than lost in a background thread. Registered
hosts can retain their normal proxy environment; these probes do not alter
production proxy or dashboard startup behavior.

`tests/rsi/test_instance_dispatch.py` additionally checks a real owned venv with
an installed local-only dependency, pinned source imports, receipt-only Node
PATH, refusal of stale/copied receipts, and the setup repair route. Supply an
existing Node >=22 on the test runner's PATH. No Node manager or online install
is required by these fixtures.

These CPU tests do not establish DeepSeek/BGE integration or device behavior.
A production packaged-instance smoke must separately run actual offline setup
with the pinned CPU/dependency wheelhouse and npm cache, then use the resulting
ready receipt. Retain setup elapsed time separately from actor elapsed time.
