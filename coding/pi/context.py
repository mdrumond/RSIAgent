"""Deterministic admission and bounded, addressable context; no paid compaction."""
import hashlib
import json
import re


class ContextLimitError(ValueError):
    pass


def encoded(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode()


class ContextRecords:
    def __init__(self):
        self.records = {}

    def admit(self, value):
        raw = value.encode() if isinstance(value, str) else encoded(value)
        digest = hashlib.sha256(raw).hexdigest()
        self.records[digest] = raw
        return {"record": digest, "bytes": len(raw),
                "preview": raw[:384].decode(errors="ignore"),
                "read_with": "read_context"}

    def _pointer(self, value):
        return {key: value for key, value in self.admit(value).items() if key != 'preview'}

    def retrieval_rows(self, rows, *, budget=1800):
        """Stable decision indices and useful facts, with individually paged rows."""
        entries = []
        def text(value, size=256):
            return value.encode()[:size].decode(errors='ignore') if isinstance(value, str) else None
        for index, row in enumerate(rows):
            value = row if isinstance(row, dict) else {}
            memory = value.get('memory', value)
            proposal = memory.get('proposal', {})
            observed = value.get('observations') or {}
            scope = observed.get('applicability', {})
            facts = {key: shown for key, item in dict(
                memory_id=memory.get('memory_id'), project_id=proposal.get('project_id', value.get('project_id')),
                kind=proposal.get('kind'), summary=proposal.get('summary', value.get('reason', value.get('text'))),
                outcome=observed.get('outcome', value.get('outcome')), goal=scope.get('goal'),
                source_hash=scope.get('final_source_hash')).items() if (shown := text(item)) is not None}
            commands = observed.get('commands', {}).get('items', [])
            facts['commands'] = [dict(command=text(r.get('command'), 64),
                result=text(r.get('classification'), 80), final_source_scope=r.get('final_source_scope'),
                stderr=text(r.get('stderr', {}).get('text'), 160)) for r in commands[-3:]]
            entries.append(dict(index=index, facts=facts, full=self._pointer(row)))
        visible = []
        for entry in entries:
            if len(encoded([*visible, entry])) > budget:
                break
            visible.append(entry)
        return dict(count=len(rows), items=visible, omitted=len(rows) - len(visible),
                    index=self._pointer(entries), all_records=self._pointer(rows))

    def resource(self, value, *, threshold=3072, limit=6144):
        compact = self.compact(value, threshold)
        if len(encoded(compact)) <= limit:
            return compact
        result = {'full_observation': self.admit(value)}
        if not isinstance(value, dict):
            return result
        for key in ('reference', 'receipt_id', 'action', 'executed', 'status',
                    'path', 'sha256', 'project_hash', 'project_id', 'goal', 'contract'):
            if key in value:
                result[key] = value[key] if len(encoded(value[key])) <= 512 else self.admit(value[key])
        if 'summary' in value:
            result['summary'] = value['summary'] if len(encoded(value['summary'])) <= 2048 else self.admit(value['summary'])
        evidence = value.get('evidence', {})
        response = evidence.get('payload', {}).get('response', {}) if isinstance(evidence, dict) else {}
        if all(isinstance(response.get(key), list) for key in ('results', 'negative_records')):
            result['retrieval'] = {key: self.retrieval_rows(response[key], budget=1400)
                                   for key in ('results', 'negative_records')}
            if len(encoded(result)) > limit and 'summary' in value:
                result['summary'] = self._pointer(value['summary'])
        return result

    def compact(self, value, threshold=1024):
        if isinstance(value, str):
            # Host outcomes and optimizer previews carry JSON strings. Preserve
            # the bounded knowledge source list through that transport envelope.
            if len(value.encode()) <= 4096 and value.startswith('{'):
                try:
                    parsed = json.loads(value)
                except ValueError:
                    parsed = None
                if isinstance(parsed, dict):
                    observation = parsed.get('host', parsed)
                    if (isinstance(observation, dict)
                            and observation.get('event') == 'knowledge-query'
                            and 'results_index_sha256' in observation
                            and len(encoded(observation)) <= 3500):
                        return self.compact(parsed, threshold)
            return self.admit(value) if len(value.encode()) > threshold else value
        if isinstance(value, list):
            if len(encoded(value)) > 4096:
                return self.admit(value)
            return [self.compact(item, threshold) for item in value]
        if isinstance(value, dict):
            result = {}
            for key, item in value.items():
                if key == 'negative_records' and isinstance(item, list) and len(encoded(item)) > 4096:
                    result[key] = self.retrieval_rows(item)
                    continue
                if key in {"diagnostic", "compile_diagnostic", "detail"} and isinstance(item, str) and len(item.encode()) > threshold:
                    # Keep compiler errors visible even after a long warning preamble.
                    errors = [line for line in item.splitlines()
                              if re.search(r"(?:fatal )?error:|undefined reference", line)]
                    if errors:
                        result[key] = {"errors": "\n".join(errors).encode()[:2048].decode(errors="ignore"),
                                       "full": self.admit(item)}
                        continue
                result[key] = self.compact(item, threshold)
            return result
        return value

    def snapshot(self, context, *, target=24576):
        result = self.compact(context)
        if len(encoded(result)) > target:
            raise ContextLimitError("mandatory context exceeds target byte budget")
        return result

    def read(self, digest, offset=0, limit=4096):
        if digest not in self.records:
            raise ValueError("context record is not admitted")
        if type(offset) is not int or offset < 0 or type(limit) is not int or not 1 <= limit <= 8192:
            raise ValueError("invalid page bounds")
        raw = self.records[digest]
        if offset > len(raw) or offset and raw[offset:offset+1] and raw[offset] & 0xC0 == 0x80:
            raise ValueError("offset must be a returned UTF-8 page boundary")
        end = min(offset + limit, len(raw))
        while end > offset and end < len(raw) and raw[end] & 0xC0 == 0x80:
            end -= 1
        if end == offset and offset < len(raw):
            raise ValueError("page limit is too small for the next character")
        return {"record": digest, "offset": offset, "text": raw[offset:end].decode(),
                "next_offset": end if end < len(raw) else None, "bytes": len(raw)}
