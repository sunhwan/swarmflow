"""
Tests for `extract._parse_frames`.

Pure function over user-supplied --frames specs. Lots of small surface
area: `last`, `first`, `all`, single index, slice (`a:b` with empty
endpoints), comma list. Slice end-points clamp to n_frames.
"""

import pytest

from swarmflow.extract import _parse_frames


def test_last():
    assert _parse_frames('last', 10) == [9]


def test_first():
    assert _parse_frames('first', 10) == [0]


def test_all():
    assert _parse_frames('all', 3) == [0, 1, 2]


def test_single_index():
    assert _parse_frames('5', 10) == [5]


def test_slice_explicit():
    assert _parse_frames('2:5', 10) == [2, 3, 4]


def test_slice_empty_start_means_zero():
    assert _parse_frames(':5', 10) == [0, 1, 2, 3, 4]


def test_slice_empty_end_means_n_frames():
    assert _parse_frames('7:', 10) == [7, 8, 9]


def test_slice_end_clamps_to_n_frames():
    """Out-of-range end shouldn't IndexError later — clamp to n_frames."""
    assert _parse_frames('0:1000', 10) == list(range(10))


def test_comma_list():
    assert _parse_frames('1,3,5', 10) == [1, 3, 5]


def test_comma_list_with_whitespace_and_trailing_comma():
    assert _parse_frames('1, 3, 5,', 10) == [1, 3, 5]


@pytest.mark.parametrize('spec', ['last', 'first', 'all', '0', ':1', '0:1'])
def test_returns_list_of_ints(spec):
    out = _parse_frames(spec, 5)
    assert isinstance(out, list)
    assert all(isinstance(i, int) for i in out)
