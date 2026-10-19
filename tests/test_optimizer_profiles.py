from optimizer.profiles import normalize_csv, discover, compact_reports, profile


def test_units_zero_missing_and_no_timeline(tmp_path):
    report = tmp_path / 'op_summary.csv'
    report.write_text('Op Name,aic_mte2_ratio,Read Bandwidth(GB/s),Time(us),Count\nk,0,N/A,2.5,3\n')
    rows = normalize_csv(report)
    assert [(r['name'], r['value'], r['unit']) for r in rows] == [
        ('aic_mte2_ratio', 0, None), ('Read Bandwidth', None, 'GB/s'),
        ('Time', 2.5, 'us'), ('Count', 3, None)]


def test_discover_records_help_and_enum_capabilities():
    class Metrics:
        PipeUtilization = 1
        Memory = 2
        PipeTimeline = 3
    result = discover(metrics=Metrics, help_text='--aic-metrics=PipeUtilization,Memory')
    assert result['groups'] == ['Memory', 'PipeUtilization']
    assert '--aic-metrics' in result['help']


def test_compact_report_manifest_excludes_timeline(tmp_path):
    (tmp_path / 'op_summary_0.csv').write_text('Op Name,aiv_vec_ratio\nx,0.5\n')
    (tmp_path / 'timeline.csv').write_text('secret large data')
    (tmp_path / 'trace.json').write_text('secret timeline')
    result = compact_reports(tmp_path, tmp_path / 'compact')
    assert len(result) == 1
    assert result[0]['metrics'][0]['value'] == .5
    assert len(result[0]['source']['sha256']) == 64
    assert 'secret' not in str(result)


def test_unsupported_and_failed_captures_remain_unavailable(tmp_path, monkeypatch):
    monkeypatch.setattr('optimizer.profiles.discover', lambda: {'groups': ['Memory']})
    def failure(*args, **kwargs):
        raise RuntimeError('counter unavailable')
    monkeypatch.setattr('optimizer.profiles.capture', failure)
    result = profile(lambda: None, tmp_path / 'profiles', ['Memory', 'PipeTimeline'])
    assert not any(g['available'] for g in result['groups'])
    assert result['groups'][0]['reason'] == 'RuntimeError: counter unavailable'
    assert result['groups'][1]['reason'] == 'unsupported non-timeline metric group'
