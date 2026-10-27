import pytest

from optimizer.profiles import GROUPS, normalize_csv, discover, compact_reports, profile


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


@pytest.mark.parametrize('group', GROUPS)
@pytest.mark.parametrize('columns,values', [
    ('Task Duration(us),Stream ID', '2,0'),
    ('aicore_time(us),aic_total_cycles,aiv_time(us),aiv_total_cycles', '1,100,2,200'),
])
def test_generic_columns_do_not_establish_group_availability(tmp_path, monkeypatch, group, columns, values):
    monkeypatch.setattr('optimizer.profiles.discover', lambda: {'groups': [group]})
    def fake_capture(fn, root, *args, **kwargs):
        root.mkdir(parents=True)
        (root / 'op_summary_0.csv').write_text(f'Op Name,{columns}\nk,{values}\n')
        return root
    monkeypatch.setattr('optimizer.profiles.capture', fake_capture)
    result = profile(lambda: None, tmp_path / 'profiles', [group])['groups'][0]
    assert result['reports']
    assert result['available'] is False
    assert result['reason'] == 'no group counters exported'


@pytest.mark.parametrize('column,value', [
    ('aic_main_mem_read_bw(GB/s)', 'N/A'),
    ('aiv_main_mem_write_bw(GB/s)', 'nan'),
    ('aic_mte2_ratio', '0.5'),
])
def test_missing_or_wrong_group_counters_are_unavailable(tmp_path, monkeypatch, column, value):
    monkeypatch.setattr('optimizer.profiles.discover', lambda: {'groups': ['Memory']})
    def fake_capture(fn, root, *args, **kwargs):
        root.mkdir(parents=True)
        (root / 'op_summary_0.csv').write_text(f'Op Name,{column}\nk,{value}\n')
        return root
    monkeypatch.setattr('optimizer.profiles.capture', fake_capture)
    result = profile(lambda: None, tmp_path / 'profiles', ['Memory'])['groups'][0]
    assert result['available'] is False
    assert result['counter_columns'] == []
    assert result['reason'] == 'no group counters exported'


@pytest.mark.parametrize('group,column', [
    ('PipeUtilization', 'aic_mac_ratio'),
    ('Memory', 'aic_mte2_main_mem_read_bandwidth(GB/s)'),
    ('Memory', 'aic_mte2_read_main_memory(GB/s)'),
    ('MemoryL0', 'aic_mte1_l0a_bandwidth(GB/s)'),
    ('MemoryUB', 'aiv_mte2_ub_bandwidth(GB/s)'),
    ('L2Cache', 'aic_l2cache_read_hit_rate'),
    ('ArithmeticUtilization', 'aiv_vec_fp32_ratio'),
    ('ResourceConflictRatio', 'aiv_vec_bankgroup_cflt_ratio'),
    ('MemoryAccess', 'aic_mte2_read_main_memory'),
    ('Memory', 'aic_l1_read_bw(GB/s)'),
    ('Memory', 'aiv_main_mem_write_bw(GB/s)'),
    ('MemoryL0', 'aic_l0c_write_bw_cube(GB/s)'),
    ('MemoryUB', 'aiv_ub_read_bw_vector(GB/s)'),
    ('L2Cache', 'aic_write_cache_hit'),
    ('L2Cache', 'aiv_r1_read_cache_miss_allocate'),
    ('ArithmeticUtilization', 'aiv_vector_fops'),
    ('MemoryAccess', 'aic_read_main_memory_datas(KB)'),
    ('MemoryAccess', 'GM_to_UB_datas(KB)'),
])
def test_group_counters_accept_measured_zero(tmp_path, monkeypatch, group, column):
    monkeypatch.setattr('optimizer.profiles.discover', lambda: {'groups': [group]})
    def fake_capture(fn, root, *args, **kwargs):
        root.mkdir(parents=True)
        (root / 'op_summary_0.csv').write_text(f'Op Name,{column}\nk,0\n')
        return root
    monkeypatch.setattr('optimizer.profiles.capture', fake_capture)
    result = profile(lambda: None, tmp_path / 'profiles', [group])['groups'][0]
    assert result['available'] is True
    assert result['counter_columns'] == [column]


@pytest.mark.parametrize('value,available', [('NA', False), ('0', True)])
def test_candidate_availability_ignores_other_kernel_counters(tmp_path, monkeypatch, value, available):
    monkeypatch.setattr('optimizer.profiles.discover', lambda: {'groups': ['PipeUtilization']})
    def fake_capture(fn, root, *args, **kwargs):
        root.mkdir(parents=True)
        (root / 'op_summary_0.csv').write_text(
            f'Op Name,aiv_vec_ratio\nother_kernel,90\ncandidate_kernel,{value}\n')
        return root
    monkeypatch.setattr('optimizer.profiles.capture', fake_capture)
    result = profile(lambda: None, tmp_path / 'profiles', ['PipeUtilization'],
                     identity={'kernel_symbol': 'candidate_kernel'})['groups'][0]
    assert result['available'] is available
    if not available:
        assert result['reason'] == 'no requested counters for candidate kernel symbol'
