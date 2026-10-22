"""Human explanations for presentation; never evidence or a status transition.

Keep codes stable. Consumers may render ``status_explanations`` alongside the
original fields, including on historical records that predate these descriptions.
"""
from collections.abc import Mapping


# Each entry says what is established and the useful next step. In particular,
# execution success, project acceptance, and memory usefulness are separate facts.
_DESCRIPTIONS = {
    'host-mismatch': ('The executed output differs from the host reference under the configured comparison policy.',
                      'Read the retained failing inputs, expected and actual outputs, source, and comparison diagnostics; repair and host-verify the source.'),
    'compile-failure': ('The candidate did not compile; no correct execution has been established.',
                        'Read the failing source and compiler diagnostics, repair the build, and execute host verification.'),
    'runtime-failure': ('The candidate compiled but execution failed before correctness could be established.',
                        'Read the execution diagnostics and retained inputs, repair the failure, and rerun host verification.'),
    'host-verified': ('The tested output matches the host reference for the evaluated cases. Other project requirements may still be outstanding.',
                      'Check remaining submission requirements; a remedial success alone does not establish that the original project is fixed.'),
    'actor-unverified': ('The actor did not complete an accepted, host-verified project submission.',
                         'Inspect the latest host observations and missing submission requirements; passing an intermediate run may be insufficient.'),
    'verified': ('The verifier accepted the evaluated candidate under its configured checks.',
                 'Consult the evaluated cases and evidence before extending this result to other inputs or tasks.'),
    'incorrect': ('The evaluated result did not satisfy the correctness checks.',
                  'Read the failing case and comparison diagnostics, then verify a repaired candidate.'),
    'failed': ('This work ended unsuccessfully. This status alone does not identify whether the cause is the candidate, provider, or infrastructure.',
               'Read the attached error, classification, and retained evidence before choosing repair or retry.'),
    'remediated': ('Remedial evidence is available, but the original failed project has not yet passed its retry.',
                   'Rerun the original project with its original acceptance checks.'),
    'fixed': ('The original failed project passed its required retry checks.',
              'Consult the original retry evidence for the scope of the confirmed fix.'),
    'open': ('The failure is recorded and has no completed investigation yet.',
             'Inspect the observed failure and select a relevant investigation or explain a deferral.'),
    'investigating': ('The failure is under investigation; there is not yet an accepted recommendation to retry the original.',
                      'Use linked experiments to resolve the remaining uncertainty and explain why further investigation is needed.'),
    'ready-to-retry': ('Linked investigation evidence supports retrying the original failure; it does not prove a fix.',
                       'Run the original project again and record whether its original acceptance checks pass.'),
    'resolved': ('The investigation has been resolved according to the recorded evidence.',
                 'Read the resolution and original retry evidence; do not infer broader correctness.'),
    'deferred': ('Investigation or execution has been postponed for the recorded reason.',
                 'Read the reason and required prerequisites before scheduling further work.'),
    'learnable': ('The failure is classified as a candidate issue that may yield reusable knowledge.',
                  'Investigate the concrete failing example and test any proposed lesson.'),
    'infrastructure': ('The failure is attributed to execution infrastructure rather than established candidate behavior.',
                       'Inspect transport, device, and environment diagnostics; restore the required environment before retrying.'),
    'provider': ('The model provider request failed; this does not establish candidate correctness.',
                 'Inspect the provider error and retry only when provider access is available and elapsed allowance remains.'),
    'invalid-specification': ('The requested task is classified as invalid under its recorded specification.',
                              'Read the evidence and reason, and correct the specification before execution.'),
    'unsupported': ('The requested capability is not supported by the selected environment or interface.',
                    'Read the capability diagnostic and select a supported configuration or explicitly defer the task.'),
    'impossible': ('The investigation claims the requirement cannot be satisfied under the stated constraints.',
                   'Inspect the supporting evidence; elapsed allowance exhaustion alone cannot establish impossibility.'),
    'budget-exhausted': ('The allocated elapsed time has been used up. Correctness or impossibility is not implied.',
                         'Inspect retained progress and outcomes before explicitly allocating additional time.'),
    'queued': ('Work is recorded and waiting to start.', 'Observe the existing job rather than submitting a duplicate.'),
    'running': ('Work is executing; no final result is established yet.', 'Observe the existing job and retained progress.'),
    'stopped': ('Execution has stopped and may have retained partial progress.', 'Inspect the stop reason and retained state before resuming.'),
    'complete': ('The workflow step completed. Completion alone does not mean every candidate passed.',
                 'Read the verification and project outcomes for correctness.'),
    'completed': ('The workflow step completed. Completion alone does not mean every candidate passed.',
                  'Read the verification and project outcomes for correctness.'),
    'pending': ('This result or prerequisite is still pending.', 'Read the associated detail and observe or satisfy the pending prerequisite.'),
    'available': ('The requested artifact or resource is available.', 'Read the referenced resource when its evidence is needed.'),
    'unavailable': ('The requested artifact, diagnostic, or resource could not be obtained.',
                    'Read the associated error; absence of diagnostics is not evidence that the candidate passed.'),
    'approved': ('The proposal has approval to run; execution and verification are separate steps.', 'Run or observe the approved proposal.'),
    'rejected': ('The proposal was rejected and is not approved for execution.', 'Read the rejection reason before submitting a revised proposal.'),
    'awaiting-approval': ('The proposal requires approval before execution.', 'Review the proposed projects and record approval or rejection.'),
    'observation-unavailable': ('The worker could not obtain a usable observation of the running work.',
                                'Recover observation of the existing handle before deciding whether to resume.'),
    'both-succeeded': ('Both memory and no-memory runs passed. This does not establish that the memory is useless.',
                       'Retain both results as usefulness evidence; do not infer a causal memory benefit from this experiment.'),
    'memory-only-succeeded': ('Only the run with the new memory passed this paired experiment.',
                             'Retain the paired evidence as support for usefulness on this task, without claiming general causality.'),
    'without-memory-only-succeeded': ('Only the run without the new memory passed this paired experiment.',
                                     'Inspect both traces and retain the result as possible regression evidence.'),
    'neither-succeeded': ('Neither paired run passed, so this experiment did not demonstrate a successful repair.',
                          'Inspect both failures and retain the outcome without declaring the memory useless.'),
    'inconclusive': ('The available evidence does not establish the requested conclusion.',
                     'Read the missing evidence or failed checks; record uncertainty rather than assigning a stronger result.'),
    'performance-study': ('Required elapsed-time measurements have not been retained for this project.',
                          'Host-verify the candidate and run the configured timing study; retain its measurement artifact.'),
    'profiling': ('Required profiler evidence has not been retained for this project.',
                  'Run the configured profiling action for a verified candidate and retain the required metric evidence.'),
    'observed-host-mismatch': ('This repair exercise lacks an accepted failing execution whose output differs from the host reference, followed by a verified repair.',
                               'Read the supplied original reproducer and linked evidence. If none is available, report that missing evidence; do not invent a failing input.'),
    'observed-compile-failure': ('This repair exercise lacks an accepted compiler failure followed by a verified repair.',
                                'Read the supplied failing source and compiler diagnostics. If none is available, report that missing evidence; do not invent a failure.'),
}

_DESCRIPTIONS.update({
    'demonstrated-benefit': ('The memory treatment passed while the paired no-memory control failed.',
                             'Retain both traces as evidence of benefit for this experiment, without inferring universal usefulness.'),
    'treatment-regressed': ('The memory treatment failed while the paired no-memory control passed.',
                           'Inspect both traces and record possible regression evidence without changing memory admission automatically.'),
    'READY': ('The Phase 2 curriculum declared readiness after a verified target attempt and no further useful practice.',
              'Inspect the target attempt evidence and curriculum reason; readiness does not establish correctness on other targets.'),
    'cases': ('The optimizer is selecting additional correctness-case signatures before the suite is frozen.',
              'Propose only the allowed additional signatures; do not alter supplied representatives or the reference.'),
    'optimize': ('The optimizer is working against the frozen reference and correctness suite.',
                 'Repair and verify candidates against every frozen case before interpreting performance measurements.'),
    'ready': ('This treatment or work item is eligible to run; the status alone does not establish verification.',
              'Inspect the task-specific prerequisites and run or observe the eligible work.'),
    'RECOVERY_REQUIRED': ('An earlier operation is pending and must be recovered before new work proceeds.',
                          'Observe or recover the retained operation handle rather than submitting duplicate work.'),
    'context-budget-exhausted': ('The agent could not fit its required conversation into the provider context limit.',
                                 'Inspect retained progress and context diagnostics; this is not a correctness verdict or elapsed-time result.'),
    'observed': ('A tool observation was recorded; this status alone does not establish correctness.',
                 'Read the observation classification and evidence.'),
    'orientation': ('The verifier is examining the task and environment before candidate verification.',
                    'Use the orientation findings to prepare the candidate; this stage does not certify a candidate.'),
    'candidate_verification': ('The verifier is evaluating the submitted candidate against the task requirements.',
                              'Read the completed verification report and retained execution evidence.'),
    'CANDIDATE_VERIFICATION': ('The verifier is evaluating the submitted candidate against the task requirements.',
                              'Read the completed verification report and retained execution evidence.'),
    'met': ('The verifier reports that this requirement was satisfied by the cited evidence.',
            'Read the cited evidence for the scope of the result.'),
    'violated': ('The verifier found evidence that this requirement was not satisfied.',
                 'Read the requirement and cited failure evidence before repairing the candidate.'),
    'could-not-confirm': ('The verifier did not obtain enough evidence to confirm this requirement.',
                          'Read the missing checks or evidence; uncertainty is not a passing result.'),
    'done': ('The agent segment finished; its report determines acceptance.',
             'Read the verification report rather than inferring correctness from segment completion.'),
    'stalled': ('The agent segment stopped making useful progress.', 'Read the segment diagnostic and unmet requirements before resuming.'),
    'stalled_quiescent': ('The agent segment became inactive without completing the required work.',
                          'Inspect the retained segment state and unmet requirements before resuming.'),
    'infra': ('The agent segment failed because of infrastructure or provider execution.',
              'Inspect diagnostics and restore the environment before continuing the existing work.'),
    'budget': ('The agent segment exhausted its allocation before completion.',
               'Read the allocation diagnostic and retained progress; this does not establish correctness.'),
    'RUNNING': _DESCRIPTIONS['running'],
    'BUDGET_EXHAUSTED': _DESCRIPTIONS['budget-exhausted'],
    'infrastructure-failed': _DESCRIPTIONS['infrastructure'],
    'infrastructure-failure': _DESCRIPTIONS['infrastructure'],
    'TIME_BUDGET_EXHAUSTED': _DESCRIPTIONS['budget-exhausted'],
    'waiting-approval': _DESCRIPTIONS['awaiting-approval'],
    'TARGET_REACHED': ('The configured workload target was reached; individual outcomes retain their own verification results.',
                       'Inspect the completed outcomes and their evidence.'),
    'PROJECT_LIMIT_REACHED': ('The configured project workload limit was reached.',
                              'Inspect completed and failed project outcomes before planning more work.'),
    'SATURATED': ('The curriculum stopped because its configured saturation condition was met.',
                  'Read the stopping reason and project outcomes; this does not prove all possible improvements are exhausted.'),
    'STALLED': ('The workflow stopped making progress under its configured progress checks.',
                'Inspect the stopping reason and incomplete requirements before retrying.'),
    'proposed': ('A curriculum proposal is available but has not yet been approved to run.',
                 'Review its projects and approve or reject the proposal.'),
    'adopted': ('An existing campaign has been registered for continuation.',
                'Inspect its retained progress and approval state before running further work.'),
})

_FIELDS = {'status', 'classification', 'lifecycle', 'outcome', 'category', 'stage'}


def explain_status(code, *, field='status', context=None):
    """Return a non-authoritative description with stable code and next action.

    Unknown codes explicitly remain unexplained: prettifying a token must not
    invent its semantics. ``context`` carries the containing record for callers.
    """
    if not isinstance(code, str):
        raise TypeError('status code must be a string')
    description, action = _DESCRIPTIONS.get(code, (
        f'The {field} code is {code!r}; no specific explanation is registered.',
        'Read the accompanying detail, reason, error, and evidence before interpreting this code.'))
    if field == 'lifecycle' and code == 'failed':
        description = 'The original failure has not yet been confirmed fixed by an accepted retry.'
        action = 'Inspect the investigation state and original retry evidence.'
    if code == 'pending' and context and context.get('approval_mode'):
        description, action = _DESCRIPTIONS['awaiting-approval']
    return {'code': code, 'description': description, 'next_action': action}


def describe_statuses(payload):
    """Copy a presentation tree, adding explanations without altering raw fields.

    Only named status fields and missing requirements are interpreted. Arbitrary
    prose, source, artifact payloads, and immutable evidence remain untouched.
    """
    if isinstance(payload, (list, tuple)):
        return [describe_statuses(item) for item in payload]
    if not isinstance(payload, Mapping):
        return payload
    # Exact artifact payloads stay raw and reachable; explain their surrounding
    # observation instead. This prevents display additions being mistaken for facts.
    result = {key: value if key in {'payload', 'source', 'submission'} else describe_statuses(value)
              for key, value in payload.items() if key != 'status_explanations'}
    explanations = []
    for field in sorted(_FIELDS):
        code = payload.get(field)
        if isinstance(code, str):
            explanations.append({'field': field, **explain_status(code, field=field, context=payload)})
    missing = payload.get('missing')
    if isinstance(missing, (list, tuple)):
        for code in missing:
            if isinstance(code, str):
                explanation = explain_status(code, field='missing', context=payload)
                if code.startswith('sync-performance requires '):
                    explanation.update(description='The project lacks measurements of two distinct verified source variants.',
                                       next_action=code)
                explanations.append({'field': 'missing', **explanation})
    if explanations:
        result['status_explanations'] = explanations
    return result
