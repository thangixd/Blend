import json

import duckdb
import pytest

from evaluation.nlseeker.build import populate_tables_registry, write_metadata_csv


def test_write_metadata_csv_maps_pneuma_ids_to_tableids(tmp_path):
    contexts = tmp_path / 'contexts.jsonl'
    contexts.write_text('\n'.join([
        json.dumps({'table': 'table_41', 'context': 'about candidates'}),
        json.dumps({'table': 'table_41', 'context': 'second row'}),
        json.dumps({'table': 'table_7', 'context': 'currency rates'}),
    ]))
    assignment = [(0, '/lake/CurrencyRate_SEP_table_7.csv'), (1, '/lake/JobCandidate_SEP_table_41.csv')]
    out = write_metadata_csv(contexts, assignment, tmp_path / 'metadata.csv')

    import pandas as pd
    frame = pd.read_csv(out)
    assert list(frame.columns) == ['table_id', 'value']
    assert frame['table_id'].tolist() == [1, 1, 0]
    assert frame['value'][2] == 'currency rates'


def test_write_metadata_csv_unknown_table_raises(tmp_path):
    contexts = tmp_path / 'contexts.jsonl'
    contexts.write_text(json.dumps({'table': 'table_99', 'context': 'x'}))
    with pytest.raises(KeyError):
        write_metadata_csv(contexts, [(0, '/lake/A_SEP_table_1.csv')], tmp_path / 'm.csv')


def test_populate_tables_registry_creates_sentinel_and_mapping(tmp_path):
    con = duckdb.connect(str(tmp_path / 'eval.duckdb'))
    populate_tables_registry(con, 'mini', [(0, '/lake/A_SEP_table_1.csv'), (1, '/lake/B_SEP_table_2.csv')])
    assert con.execute('SELECT count(*) FROM "mini"').fetchone()[0] == 0
    rows = con.execute('SELECT tableid, filename FROM "mini_tables" ORDER BY tableid').fetchall()
    assert rows == [(0, 'A_SEP_table_1.csv'), (1, 'B_SEP_table_2.csv')]
    con.close()
