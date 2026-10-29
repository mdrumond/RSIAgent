"""Host-owned explanation of review progress, separate from model decisions.

These fields are diagnostic only. They never authorize publication, retries or
repair scheduling, and may be derived for legacy records without rewriting them.
"""


def assessment(status, stage, reason_code):
    return dict(status=status, stage=stage, reason_code=reason_code)


def project_assessment(record):
    review = record.get('review')
    if review is None:
        return record.get('assessment') or assessment('pending', 'source-audit', 'review-pending')
    if review['decision'] == 'repair':
        return assessment('needs-repair', 'source-audit', 'source-repair-required')
    if review.get('memory') is None:
        return assessment('no-proposal', 'source-audit', 'no-memory-proposed')
    return assessment('eligible', 'source-audit', 'source-audit-passed')


def review_report(state):
    """Enrich legacy source records consistently for live and offline reports."""
    projects = {key: dict(row, assessment=project_assessment(row))
                for key, row in state.get('projects', {}).items()}
    return dict(state, projects=projects)
