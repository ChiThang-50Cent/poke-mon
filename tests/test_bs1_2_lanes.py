from pokeai.env.runner import RunConfig, _config_dict, lane_indices


def test_lanes_partition_the_schedule_in_whole_swapped_pairs():
    a, b = lane_indices(26, [0, 2]), lane_indices(26, [1, 2])
    assert sorted(a + b) == list(range(26)) and not set(a) & set(b)
    for lane in (a, b):
        assert all((i + 1 in lane) for i in lane if i % 2 == 0)  # pair stays in one lane
    assert lane_indices(6, None) == list(range(6))


def test_pair_windows_slice_the_same_schedule_and_blocks_cover_it_once():
    blocks = [[0, 12], [12, 25], [25, 37], [37, 50]]
    seen = []
    for w in blocks:
        a, b = lane_indices(100, [0, 2], w), lane_indices(100, [1, 2], w)
        assert not set(a) & set(b)
        seen += sorted(a + b)
    assert seen == list(range(100))


def test_lane_key_is_absent_from_the_config_dict_when_unused():
    assert "lane" not in _config_dict(RunConfig(battles=2))
    assert _config_dict(RunConfig(battles=4, lane=[1, 2]))["lane"] == [1, 2]
