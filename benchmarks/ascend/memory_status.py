"""Host-owned explanation of review progress, separate from model decisions.

These fields are diagnostic only. They never authorize publication, retries or
repair scheduling, and may be derived for legacy records without rewriting them.
"""


def assessment(status, stage, reason_code):
    return dict(status=status, stage=stage, reason_code=reason_code)


def evaluation_assessment(result):
    decision = result.get('decision')
    if decision in {'accept', 'rewrite', 'reject'}:
        return assessment({'accept': 'accepted', 'rewrite': 'needs-repair',
                           'reject': 'rejected'}[decision], 'usefulness', 'judge-' + decision)
    for arm in result.get('arms', {}).values():
        observations = (arm or {}).get('observations') or {}
        evidence = observations.get('evidence') or {}
        fields = [observations.get(key) for key in
                  ('classification', 'status', 'error_type', 'interruption_reason')]
        if isinstance(evidence, dict):
            fields += [evidence.get(key) for key in ('classification', 'actor_status')]
        if any('infrastructure' in str(value).lower() or value in
               {'transport-unresolved', 'provider-failure'} for value in fields):
            return assessment('infrastructure-blocked', 'transfer', 'arm-infrastructure')
    if result.get('validity') != {'control': True, 'treatment': True}:
        return assessment('benefit-unproven', 'transfer', 'pair-incomplete')
    if result.get('error_type'):
        return assessment('audit-incomplete', 'usefulness', 'judge-incomplete')
    return assessment('benefit-unproven', 'usefulness', 'judge-inconclusive')


def admission_assessment(job):
    status = job['status']
    if status == 'accepted':
        return assessment('accepted', 'publication', 'memory-published')
    if status == 'rejected':
        if job.get('project_repair'):
            return assessment('needs-repair', 'source-audit', 'source-repair-required')
        if job.get('rewrite_reviews') and job['rewrite_reviews'][-1].get('memory') is None:
            return assessment('no-proposal', 'source-audit', 'no-memory-proposed')
        return assessment('rejected', 'usefulness', 'rewrite-opportunities-exhausted')
    if job.get('error_type'):
        return assessment('audit-incomplete', 'admission', 'worker-interrupted')
    if status in {'pending', 'evaluating'}:
        return assessment(status, 'admission', 'review-' + status)
    evaluations = job.get('evaluations') or []
    return evaluation_assessment(evaluations[-1] if evaluations else {})


def project_assessment(record):
    review = record.get('review')
    if review is None:
        return record.get('assessment') or assessment('pending', 'source-audit', 'review-pending')
    if review['decision'] == 'repair':
        return assessment('needs-repair', 'source-audit', 'source-repair-required')
    if review.get('memory') is None:
        return assessment('no-proposal', 'source-audit', 'no-memory-proposed')
    return assessment('eligible', 'source-audit', 'source-audit-passed')
