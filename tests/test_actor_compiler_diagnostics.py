"""Compiler logs stay readable through an ordinary host-owned repair attempt."""
import hashlib
import json
import threading
from types import SimpleNamespace

import pytest

from benchmarks.a5kernels.artifacts import ContentAddressedArtifacts
from benchmarks.a5kernels.evidence import EvidenceLedger, canonical_digest
from benchmarks.a5kernels.phase1_adaptive_actor import (
    _ProjectActions, compiler_diagnostic_preview, disabled_knowledge_action,
    disabled_profile_action,
)
from benchmarks.a5kernels.phase1_project_executor import CompileObservation, ProjectExecutor
from benchmarks.a5kernels.trial import TrialAction
from coding.pi.adapters import A5_SCHEMAS
from coding.pi.session import PiCodingSession
from test_a5_phase1_adaptive_actor import SourceBackend, _request


def actions(tmp_path, backend):
    executor=ProjectExecutor(backend, EvidenceLedger(tmp_path/'evidence.jsonl'))
    request=_request(tmp_path, executor)
    return _ProjectActions(executor,request,executor.prepare(request.spec),
        ContentAddressedArtifacts(tmp_path/'artifacts'),max_actions=1,max_seconds=60,
        max_source_bytes=10000,query_knowledge=disabled_knowledge_action,
        request_profile=disabled_profile_action,request_timing=disabled_profile_action)


def host(actions, name, **payload):
    return json.loads(actions(TrialAction(name,payload)).observation)['host']


def editable_session(source, write_action=None):
    pi=PiCodingSession.__new__(PiCodingSession)
    pi.lock=threading.Lock()
    pi.allowed={'edit_source','write'}
    pi.schemas=A5_SCHEMAS | {'edit_source': {'type':'object','properties':{
        name:{'type':'string'} for name in ['sha256','old_text','new_text']},
        'required':['sha256','old_text','new_text'],'additionalProperties':False}}
    pi.mutations=0;pi.first_action_only=False;pi.pending=None
    pi.source=source;pi.write_action=write_action or (lambda s:('write',{'slot':'kernel','content':s}))
    return pi


def test_compiler_log_read_then_audited_edit_repair_and_submit(tmp_path):
    diagnostic=('compiler warning: template diagnostic\n'*700 +
                'candidate.cpp:91: error: use fixed source instead\n' +
                'compiler warning: trailing diagnostic\n'*300)
    details={'stage':'compile','stderr':diagnostic,'stdout':'compiler stdout'}
    class Backend(SourceBackend):
        def compile(self, request):
            if request.candidate_source=='compile failure':
                self.requests.append(request)
                return CompileObservation(False,canonical_digest(details),diagnostic,None,details)
            return super().compile(request)
    backend=Backend();current=actions(tmp_path,backend)
    host(current,'write',slot='kernel',content='compile failure')
    failed=host(current,'compile')
    assert failed['passed'] is False and failed['repeated_source'] is False
    assert failed['diagnostic'].startswith('candidate.cpp:91: error:')
    digest=failed['diagnostics_artifact']['sha256']
    assert current.admitted[digest]=='compile-diagnostics'
    text='';offset=0
    while offset is not None:
        page=host(current,'read-artifact',sha256=digest,offset=offset,limit=8192)
        text+=page['text'];offset=page['next_offset']
    assert text==diagnostic
    compiled=current.artifacts.read_json(failed['artifact']['sha256'])
    assert compiled['compiler_output']==details
    assert host(current,'compile')['repeated_source'] is True
    pi=editable_session('compile failure')
    receipt=pi._operation('edit_source',{'sha256':hashlib.sha256(b'compile failure').hexdigest(),
        'old_text':'compile failure','new_text':'fixed source'})
    assert receipt=={'status':'queued','executed':False,'action':'write'}
    name,payload=pi.pending
    host(current,name,**payload)
    assert current.source=='fixed source' and current.latest is None
    assert host(current,'compile')['repeated_source'] is False
    assert host(current,'run')['classification']=='host-verified'
    submitted=current(TrialAction('submit',{}))
    assert submitted.terminal and submitted.status=='host-verified'
    assert len(backend.run_requests)==1


def test_backend_compile_evidence_must_match_digest(tmp_path):
    class Backend(SourceBackend):
        def compile(self, request):
            return CompileObservation(False,'0'*64,'actual error',None,{'stderr':'different bytes'})
    current=actions(tmp_path,Backend())
    host(current,'write',slot='kernel',content='compile failure')
    with pytest.raises(ValueError,match='compiler details differ'):
        host(current,'compile')


def test_native_edit_cannot_queue_an_unavailable_host_route():
    pi=editable_session('broken',lambda s:('compile',{}))
    with pytest.raises(ValueError,match='unavailable host write'):
        pi._operation('edit_source',{'sha256':hashlib.sha256(b'broken').hexdigest(),
            'old_text':'broken','new_text':'fixed'})
    assert pi.pending is None


def test_large_unicode_diagnostic_preview_is_bounded_and_unchanged_when_small():
    text='warning: é\n'*2000+'kernel.cpp:4: fatal error: missing header\n'+'detail: é\n'*2000
    preview=compiler_diagnostic_preview(text)
    assert preview.startswith('kernel.cpp:4: fatal error:') and len(preview.encode())<=2048
    assert compiler_diagnostic_preview('error: expected semicolon')=='error: expected semicolon'
