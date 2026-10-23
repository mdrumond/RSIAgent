"""Bounded source excerpts with independently readable, exact knowledge artifacts."""
from __future__ import annotations

import json
import re
from collections.abc import Callable, Mapping


def _encoded(value):
    return json.dumps(value, ensure_ascii=True, separators=(',', ':')).encode('ascii')


def _excerpt(text: str, query: str) -> str:
    # Select a source window, not an inferred answer or an LLM paraphrase.
    terms = set(re.findall(r'\w+', query.casefold()))
    lines = list(re.finditer(r'[^\n]+', text))
    if not lines:
        return ''
    selected = max(lines, key=lambda line:
                   len(terms & set(re.findall(r'\w+', line.group().casefold()))))
    line = selected.group()
    match = next((word for word in re.finditer(r'\w+', line)
                  if word.group().casefold() in terms), None)
    offset = min(max(0, match.start() - 80), max(0, len(line) - 300)) if match else 0
    return line[offset:offset + 300]


def knowledge_presentation(query: str, rows: list[Mapping],
                           retain: Callable[[bytes], str]) -> dict:
    """Retain full chunks and a complete index; inline only a small source list.

    The index contains provenance, excerpts and full-text pointers, never another
    copy of the complete source text. ``retain`` must return its SHA256 identifier.
    """
    entries = []
    for row in rows:
        citation = dict(row['citation'])
        text = row['text']
        entries.append({
            'source': f"{citation['path']}:{citation['start_line']}-{citation['end_line']}",
            'excerpt': _excerpt(text, query),
            'citation': citation,
            'score': row.get('score'),
            'artifact_sha256': retain(text.encode('utf-8')),
        })
    index = {'summary_kind': 'query-focused source excerpts', 'results': entries}
    observation = {
        'event': 'knowledge-query', 'available': True,
        'summary_kind': index['summary_kind'], 'result_count': len(entries),
        'results_index_sha256': retain(_encoded(index)),
        'read_with': 'read-artifact',
        'read_help': 'Use sha256 and offset=0. artifact_sha256 reads exact source text; '
                     'results_index_sha256 reads all source excerpts, citations and pointers.',
        'results': entries.copy(), 'omitted_results': 0,
    }
    # Keep this below the host's 4096-byte observation compaction boundary.
    while len(_encoded(observation)) > 3500 and observation['results']:
        observation['results'].pop()
        observation['omitted_results'] += 1
    return observation


def compact_knowledge_result(query: str, result, retain: Callable[[bytes], str]):
    """Accept the existing list/observation callback shapes; preserve other replies."""
    if isinstance(result, dict) and result.get('available') is False:
        return result
    rows = result.get('results') if isinstance(result, dict) else result
    if (isinstance(rows, list) and all(isinstance(row, Mapping)
            and isinstance(row.get('text'), str) and isinstance(row.get('citation'), Mapping)
            and {'path', 'start_line', 'end_line'} <= row['citation'].keys()
            for row in rows)):
        return knowledge_presentation(query, rows, retain)
    return result
