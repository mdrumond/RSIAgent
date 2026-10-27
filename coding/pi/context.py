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
