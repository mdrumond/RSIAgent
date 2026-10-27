"""Page aggregate review evidence without hiding admission controls or raising limits."""
from coding.pi.context import ContextLimitError, ContextRecords, encoded


class ReviewContextRecords(ContextRecords):
    # These are presentation/evidence sections, not the authoritative controls.
    # Project/source identity, host flags, required reads, and allowed evidence
    # stay directly visible. Every replacement points to the exact original data.
    PAGEABLE = (
        ('host', 'project', 'rewrite_feedback', 'prior_draft'),
        ('host', 'project', 'rewrite_feedback', 'retained_lessons'),
        ('host', 'project', 'rewrite_feedback', 'previous_proposal'),
        ('host', 'project', 'rewrite_feedback', 'usefulness', 'arms'),
        ('host', 'project', 'rewrite_feedback', 'usefulness', 'evidence'),
        ('host', 'tool_observations'),
        ('host', 'validation_feedback', 'allowed_citations'),
        ('host', 'validation_feedback', 'allowed_evidence'),
        ('host', 'artifact_pages'),
        ('host', 'review_state', 'citations'),
        ('host', 'review_state', 'observed_versions'),
        ('host', 'project', 'project', 'host_facts'),
        ('host', 'project', 'project', 'objective'),
        ('host', 'project', 'project', 'artifacts'),
    )

    def snapshot(self, context, *, target=24576):
        result = self.compact(context)
        if len(encoded(result)) <= target:
            return result
        candidates = []
        for path in self.PAGEABLE:
            original, parent = context, result
            try:
                for key in path[:-1]:
                    original, parent = original[key], parent[key]
                key = path[-1]
                value = original[key]
                reference = dict(self.admit(value), section='.'.join(path[1:]))
                saving = len(encoded(parent[key])) - len(encoded(reference))
                if saving > 0:
                    candidates.append((saving, parent, key, reference))
            except (KeyError, TypeError):
                continue
        # Several individually small dictionaries can exceed the aggregate cap.
        # Page the largest useful sections first; never truncate their contents.
        for _, parent, key, reference in sorted(candidates, key=lambda row: row[0], reverse=True):
            if len(encoded(result)) <= target:
                break
            parent[key] = reference
        if len(encoded(result)) > target:
            raise ContextLimitError('mandatory review controls exceed target byte budget')
        return result
