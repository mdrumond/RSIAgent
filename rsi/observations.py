"""Bounded, source-scoped host observations for accepted SubjectMemory records.

Call only after the owning run's scope/source/outcome receipts are verified.
Selected strategies remain model claims; host command outcomes establish only
observations on the frozen verifier/input scope, never universal correctness.
"""
import json


def capture(runner, artifacts, outcome):
    refs = set()
    def retained(data):
        reference = artifacts.put_bytes(data)
        refs.add(reference)
        return reference
    def preview(data, limit):
        text = data.decode('utf-8', errors='replace')
        return dict(reference=retained(data), text=text[:limit], bytes=len(data),
                    text_chars=len(text), truncated=len(text) > limit,
                    decoding='utf-8-with-replacement')
    spec = runner.spec
    actual = runner.backend.project_hash()
    files, total_chars = [], 0
    for name in runner.config.editable_files[:8]:
        path = runner.backend._file(name)
        row = dict(path=name, present=path.is_file())
        if row['present']:
            data = path.read_bytes()
            shown = preview(data, max(0, min(3000, 12000 - total_chars)))
            total_chars += len(shown['text'])
            row.update(shown, sha256=shown['reference'])
        files.append(row)
    command_rows = []
    operations = list(runner.workflow.state.get('execution_reservations', {}))
    for operation in operations[-12:]:
        receipt = runner.backend.receipt(operation)  # Complete authoritative backend receipt.
        reference = artifacts.put_json(receipt)
        refs.add(reference)
        command_rows.append(dict(receipt_ref=reference, command=receipt['command'],
            operation_id=operation, argv=receipt['argv'], source_hash=receipt['source_hash'],
            final_source_scope=receipt['source_hash'] == actual,
            classification=receipt['classification'], returncode=receipt['returncode'],
            numeric=receipt['numeric'], profiling=receipt['profiling'],
            elapsed_seconds=receipt['elapsed'], clock_domain=receipt['clock_domain'],
            stdout=preview(runner.store.read_bytes(receipt['stdout']), 750),
            stderr=preview(runner.store.read_bytes(receipt['stderr']), 750)))
    selections = [e for e in runner.events if e['type'] == 'project-tool' and e['name'] == 'select_strategy']
    claims = []
    for event in selections[-4:]:
        reference = event['reference']
        record = runner.store.read_json(reference)
        if record['context']['workflow'] != runner.contract_ref:
            raise ValueError('strategy observation belongs to another workflow')
        retained(runner.store.read_bytes(reference))
        proposal = record['payload']['proposal']
        claims.append(dict(reference=reference, authority='model-claim',
            host_admission='scope-and-evidence-coverage-only', stage=event['stage'],
            source_hash=record['context']['source'], proposal=proposal[:1800],
            proposal_chars=len(proposal), truncated=len(proposal) > 1800))
    validation = runner.latest('validation-results')
    if validation and validation['source_hash'] != actual:
        validation = None
    verifier = preview(runner.backend.verifier.read_bytes(), 3000)
    refs.update((runner.contract_ref, runner.backend.manifest['contract']))
    observation = dict(schema='rsi-project-observations-v1', authority='host-observed-project-facts',
        outcome=outcome, applicability=dict(project_id=spec.project_id, goal=spec.goal,
            project_contract_ref=runner.contract_ref, execution_contract_ref=runner.backend.manifest['contract'],
            final_source_hash=actual, case_scope=dict(verifier=verifier,
                execution_inputs=runner.backend.contract.get('execution_inputs', []),
                commands=runner.backend.contract['commands']),
            performance_goal=spec.performance_goal, performance_seconds=runner.config.performance_seconds),
        final_sources=files, source_files_total=len(runner.config.editable_files),
        source_files_truncated=len(runner.config.editable_files) > len(files),
        strategies=dict(items=claims, total=len(selections), truncated=len(selections) > len(claims)),
        commands=dict(items=command_rows, total=len(operations), truncated=len(operations) > len(command_rows)),
        validation=validation, actor_seconds=runner.workflow.accounting.seconds(),
        limitations=['Only the frozen project, inputs and verifier were observed.',
                     'Strategy proposals are model claims; command receipts are host observations.',
                     'A passing finite case suite does not establish universal correctness.'],
        artifact_refs=sorted(refs))
    # A useful bounded writer summary even when older Learning only aggregates evidence summaries.
    source = next((row for row in files if row.get('present')), {})
    summary = json.dumps(dict(outcome=outcome, goal=spec.goal[:100],
        host_commands=[dict(command=r['command'], result=r['classification'], elapsed=r['elapsed_seconds'])
                       for r in command_rows if r['final_source_scope']][-3:],
        source=dict(path=source.get('path'), text=source.get('text', '')[:160])), sort_keys=True)
    return observation, summary
