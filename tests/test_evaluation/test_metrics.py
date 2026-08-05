from evaluation.nlseeker.metrics import hit, latency_stats, recall, reciprocal_rank


def test_hit():
    assert hit(['table_1', 'table_2'], {'table_2'}) is True
    assert hit(['table_1'], {'table_3'}) is False
    assert hit([], {'table_1'}) is False


def test_recall_deduplicates_window():
    assert recall(['table_1', 'table_1', 'table_2'], {'table_1', 'table_3'}) == 0.5
    assert recall([], {'table_1'}) == 0.0


def test_reciprocal_rank():
    assert reciprocal_rank(['table_9', 'table_1'], {'table_1'}) == 0.5
    assert reciprocal_rank(['table_9'], {'table_1'}) == 0.0


def test_latency_stats():
    mean, p50, p95 = latency_stats([10.0, 20.0, 30.0])
    assert mean == 20.0
    assert p50 == 20.0
    assert latency_stats([]) == (0.0, 0.0, 0.0)
