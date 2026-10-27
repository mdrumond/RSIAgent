"""Standalone librarian answers grounded in reference and accepted-memory hits."""
import json
import math
import secrets
import time

from flask import abort, jsonify, render_template, request
from .control import ControlError


def answer_question(services, question, *, max_seconds=60, limit=5):
    if not isinstance(question, str) or not question.strip():
        raise ValueError('Enter a knowledge question.')
    if type(max_seconds) not in (int, float) or not math.isfinite(max_seconds) or max_seconds <= 0:
        raise ValueError('Elapsed allowance must be finite and positive.')
    deadline = time.monotonic() + max_seconds
    def remaining():
        seconds = deadline - time.monotonic()
        if seconds <= 0:
            raise TimeoutError('Knowledge question elapsed allowance exhausted.')
        return seconds
    result = services.knowledge.query(dict(query=question, limit=limit), remaining_seconds=remaining())
    sources = list(result['hits'])[:limit]
    collections = {hit['citation'].get('collection') for hit in sources}
    for memory in result.get('memories', []):
        if len(sources) >= limit:
            break
        if 'subject-memory-' + memory['version'] not in collections:
            sources.append(dict(text=json.dumps(memory['proposal'], sort_keys=True), citation=dict(
                memory_id=memory['memory_id'], version=memory['version'],
                source_sha256=memory['artifacts']['memory-document'])))
    if not sources:
        return dict(answer='No matching reference or accepted-memory evidence was found.', sources=[])
    excerpts = [dict(index=i + 1, citation=hit['citation'], text=hit['text'][:4096])
                for i, hit in enumerate(sources)]
    messages = [dict(role='system', content=(
        'Answer the user question using only supplied knowledge sources. Distinguish source facts from inference. '
        'If evidence is insufficient, say what is missing. Return JSON {"answer":"...","sources":[1,2]}; '
        'sources must identify the supplied excerpts supporting the answer. Do not invent citations.')),
        dict(role='user', content=json.dumps(dict(question=question, sources=excerpts), sort_keys=True))]
    while True:
        raw = services.complete('librarian', messages, remaining_seconds=remaining())
        remaining()
        try:
            value = json.loads(raw)
            if (not isinstance(value, dict) or set(value) != {'answer', 'sources'}
                    or not isinstance(value['answer'], str) or not value['answer'].strip()
                    or not isinstance(value['sources'], list) or not value['sources']
                    or any(type(i) is not int or not 1 <= i <= len(excerpts) for i in value['sources'])):
                raise ValueError('Return an answer and valid supporting source indices.')
            return dict(answer=value['answer'], sources=[excerpts[i - 1] for i in dict.fromkeys(value['sources'])])
        except (ValueError, TypeError) as error:
            messages.extend((dict(role='assistant', content=raw), dict(role='user', content=str(error))))


def register_knowledge(app, config_path, csrf_token):
    controls = app.extensions['dashboard_controls']
    views = app.extensions['dashboard_subjects']
    @app.get('/knowledge')
    def knowledge_page():
        return render_template('knowledge.html', campaigns=views.campaigns.values(),
                               enabled=controls.enabled, csrf_token=csrf_token)
    def validate(payload):
        if payload.get('campaign_id') not in views.campaigns:
            raise ControlError('Unknown subject campaign.')
        if not isinstance(payload.get('question'), str) or not payload['question'].strip():
            raise ControlError('Enter a knowledge question.')
        allowance = payload.get('max_seconds', 60)
        if type(allowance) not in (int, float) or not math.isfinite(allowance) or allowance <= 0:
            raise ControlError('Elapsed allowance must be finite and positive.')
        return 'subject:' + payload['campaign_id']
    def execute(payload):
        from .subject_services import bind_subject
        allowance = payload.get('max_seconds', 60)
        started = time.monotonic()
        with bind_subject(config_path, payload['campaign_id'], max_seconds=allowance) as services:
            available = allowance - (time.monotonic() - started)
            if available <= 0:
                raise TimeoutError('Knowledge binding consumed its elapsed allowance.')
            return answer_question(services, payload['question'], max_seconds=available,
                                   limit=payload.get('limit', 5))
    controls.handlers['knowledge-ask'] = (validate, execute)
    @app.post('/api/knowledge/ask')
    def knowledge_question():
        if not controls.enabled or not secrets.compare_digest(request.headers.get('X-CSRF-Token', ''), csrf_token):
            abort(403)
        try:
            operation = controls.enqueue('knowledge-ask', request.get_json(silent=True) or {},
                                         request.headers.get('Idempotency-Key'))
            return jsonify(operation_id=operation), 202
        except (ValueError, KeyError, OSError, TypeError) as error:
            return jsonify(error=str(error)), 409
