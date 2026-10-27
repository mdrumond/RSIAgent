"""Project whole provider requests, retaining complete call/result exchanges."""
import hashlib
import json

from coding.pi.context import encoded


def exchanges(rows, responses=False):
    groups, index = [], 0
    while index < len(rows):
        row = rows[index]
        calls = (row.get('tool_calls', []) if not responses else
                 [row] if row.get('type') == 'function_call' else [])
        if calls:
            group = [row]
            index += 1
            if responses:
                while index < len(rows) and rows[index].get('type') == 'function_call':
                    calls.append(rows[index]); group.append(rows[index]); index += 1
            identities = {call.get('call_id') if responses else call['id'] for call in calls}
            observed = set()
            while index < len(rows):
                candidate = rows[index]
                identity = candidate.get('call_id') if responses else candidate.get('tool_call_id')
                is_result = candidate.get('type') == 'function_call_output' if responses else candidate.get('role') == 'tool'
                if not is_result:
                    break
                if identity not in identities:
                    raise ValueError('orphan review tool result')
                observed.add(identity); group.append(candidate); index += 1
            if observed != identities:
                raise ValueError('incomplete review tool exchange')
            groups.append((group, True))
        else:
            if row.get('role') == 'tool' or row.get('type') == 'function_call_output':
                raise ValueError('orphan review tool result')
            groups.append(([row], False)); index += 1
    return groups


class ReviewRequestProjector:
    def __init__(self, records, view, record):
        self.records, self.view, self.record = records, view, record

    def __call__(self, request, *, responses=False, hard_bytes=49152):
        self.record('pi-raw-provider-request', json.loads(encoded(request)))
        key = 'input' if responses else 'messages'
        pinned = [row for row in request[key] if row.get('role') in {'system', 'developer'}]
        rows = [row for row in request[key] if row.get('role') not in {'system', 'developer'}]
        groups = exchanges(rows, responses)
        protected = {len(groups)-1}
        batches = [i for i, (_, batch) in enumerate(groups) if batch]
        if batches:
            protected.add(batches[-1])
        kept = list(range(len(groups)))
        core = self.records.snapshot(self.view(), target=min(12000, hard_bytes//3))
        while True:
            removed = [row for i, (group, _) in enumerate(groups) if i not in kept for row in group]
            payload = dict(review_context=core)
            if removed:
                raw = encoded(removed)
                payload['older_exchanges'] = dict(record=hashlib.sha256(raw).hexdigest(), bytes=len(raw),
                    preview=raw[:384].decode(errors='ignore'), read_with='read_context')
            projected = dict(request, **{key: pinned + [dict(role='user', content=json.dumps(payload))] +
                                       [row for i in kept for row in groups[i][0]]})
            if len(encoded(projected)) <= hard_bytes:
                if removed:
                    self.records.admit(removed)
                self.record('pi-request-projection', dict(raw_bytes=len(encoded(request)),
                    forwarded_bytes=len(encoded(projected)), archived_messages=len(removed),
                    retained_batches=sum(groups[i][1] for i in kept)))
                return projected
            removable = next((i for i in kept if i not in protected), None)
            if removable is None:
                raise ValueError('provider request mandatory review state/latest exchange exceeds hard byte budget')
            kept.remove(removable)
