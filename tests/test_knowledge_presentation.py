import pytest

from benchmarks.a5kernels.artifacts import ContentAddressedArtifacts
from benchmarks.ascend.agent_presentation import artifact_page_presentation
from coding.pi.context import ContextRecords, encoded
from core.knowledge_presentation import compact_knowledge_result, knowledge_presentation


def source_rows(count=20):
    return [dict(citation=dict(collection='official', path=f'docs/tool-{i}.md',
        start_line=1, end_line=900, chunk_hash=f'{i:064x}', repository='https://example.org/tool',
        revision='a' * 40, source_sha256='b' * 64),
        text='Unrelated setup details.\n' * 700 +
             f'racecheck reports synchronization hazards between copy and compute {i}.\n' +
             'Remaining information.\n' * 700, score=1.0) for i in range(count)]


def test_large_results_remain_visible_and_exact_sources_survive_later_queries(tmp_path):
    store = ContentAddressedArtifacts(tmp_path)
    rows = source_rows()
    observation = knowledge_presentation('racecheck synchronization hazards', rows, store.put_bytes)
    assert len(encoded(observation)) < 4096
    visible = ContextRecords().snapshot({'latest': observation})['latest']
    assert isinstance(visible['results'], list)
    assert 0 < len(visible['results']) < len(rows)
    assert visible['result_count'] == len(rows)
    assert visible['omitted_results'] + len(visible['results']) == len(rows)
    index = store.read_json(observation['results_index_sha256'])
    assert len(index['results']) == len(rows)
    knowledge_presentation('other query', source_rows(1), store.put_bytes)
    # Recover through the existing host read action, with a fresh store instance.
    reopened = ContentAddressedArtifacts(tmp_path)
    for entry, original in zip(index['results'], rows):
        assert entry['citation'] == original['citation']
        assert entry['excerpt'].startswith('racecheck reports synchronization hazards')
        assert 'text' not in entry
        chunks, offset = [], 0
        while offset is not None:
            page = artifact_page_presentation(dict(sha256=entry['artifact_sha256'],
                                                  offset=offset, limit=8192), reopened)
            chunks.append(page['text'])
            offset = page['next_offset']
        assert ''.join(chunks) == original['text']


@pytest.mark.parametrize('result', [{'status': 'unavailable'}, {'available': False, 'results': []},
                                  {'results': ['custom']}, None])
def test_unstructured_callback_responses_are_preserved(result):
    def retain(_):
        raise AssertionError('must not retain non-knowledge responses')
    assert compact_knowledge_result('query', result, retain) is result


def test_empty_search_has_readable_complete_index(tmp_path):
    store = ContentAddressedArtifacts(tmp_path)
    result = knowledge_presentation('nothing', [], store.put_bytes)
    assert result['results'] == []
    assert result['result_count'] == 0
    assert store.read_json(result['results_index_sha256'])['results'] == []


def test_multibyte_source_excerpts_respect_inline_bound_and_retain_original(tmp_path):
    store = ContentAddressedArtifacts(tmp_path)
    rows = source_rows(5)
    for row in rows:
        row['text'] = '同步 racecheck ' * 500
    result = knowledge_presentation('racecheck', rows, store.put_bytes)
    assert len(encoded(result)) < 4096
    assert isinstance(ContextRecords().snapshot(result)['results'], list)
    index = store.read_json(result['results_index_sha256'])
    for entry, row in zip(index['results'], rows):
        assert store.read_bytes(entry['artifact_sha256']).decode() == row['text']
