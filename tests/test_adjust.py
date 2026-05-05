"""
Tests for `adjust._propose_anchor_radii`.

Decides where to insert anchors based on per-anchor crossing stats. The
correctness conditions worth pinning:
  - balanced + sufficient crossings → no inserts.
  - bottleneck (low total): split the wider neighbor gap.
  - imbalance: split the side with FEW crossings (so it picks up more).
  - never insert closer than `2 * min_gap` (the 1e-9 tolerance line in
    adjust.py was added because radii=[0.9, 1.0] evaluate to 0.0999...8).
  - never insert duplicates.
  - 'NO DATA' rows are passed through without proposing splits.
"""

from swarmflow._config import C, load_config
from swarmflow.adjust import _propose_anchor_radii


def _row(idx, radius, total, inner, outer, flag='OK'):
    return {'anchor': idx, 'radius': radius, 'total': total,
            'inner': inner, 'outer': outer,
            'sim_time_ps': 1000.0, 'rate': total / 1000.0,
            'n_members': 4, 'flag': flag}


def test_all_balanced_no_inserts():
    load_config(None, {})
    radii = [0.1, 0.3, 0.6, 1.0]
    rows = [_row(i, r, total=200, inner=100, outer=100)
            for i, r in enumerate(radii)]
    new_radii, decisions = _propose_anchor_radii(
        rows, radii, imbalance_ratio=5.0, min_gap=0.05)
    assert new_radii == radii
    assert all(new_r is None for _, _, new_r in decisions)


def test_bottleneck_splits_wider_neighbor_gap():
    load_config(None, {'bottleneck_threshold': 30})
    # Anchor 1 is bottlenecked (total < threshold). Inner gap = 0.2,
    # outer gap = 0.4 — outer is wider, so split the outer gap.
    radii = [0.1, 0.3, 0.7, 1.0]
    rows = [
        _row(0, 0.1, total=200, inner=100, outer=100),
        _row(1, 0.3, total=10,  inner=5,   outer=5),    # bottleneck, balanced
        _row(2, 0.7, total=200, inner=100, outer=100),
        _row(3, 1.0, total=200, inner=100, outer=100),
    ]
    new_radii, decisions = _propose_anchor_radii(
        rows, radii, imbalance_ratio=5.0, min_gap=0.05)
    # Midpoint of (0.3, 0.7) = 0.5
    assert 0.5 in new_radii
    assert new_radii == sorted(set(radii + [0.5]))


def test_imbalance_outer_gt_inner_splits_inner_gap():
    """outer >> inner means few INNER crossings → split inward (inner gap)."""
    load_config(None, {'bottleneck_threshold': 30})
    radii = [0.1, 0.3, 0.7, 1.0]
    rows = [
        _row(0, 0.1, total=200, inner=100, outer=100),
        _row(1, 0.3, total=120, inner=10,  outer=110),  # outer >> inner
        _row(2, 0.7, total=200, inner=100, outer=100),
        _row(3, 1.0, total=200, inner=100, outer=100),
    ]
    new_radii, _ = _propose_anchor_radii(
        rows, radii, imbalance_ratio=5.0, min_gap=0.05)
    # Inner gap of anchor 1 is (0.1, 0.3); midpoint = 0.2
    assert 0.2 in new_radii


def test_imbalance_inner_gt_outer_splits_outer_gap():
    """inner >> outer means few OUTER crossings → split outward (outer gap)."""
    load_config(None, {'bottleneck_threshold': 30})
    radii = [0.1, 0.3, 0.7, 1.0]
    rows = [
        _row(0, 0.1, total=200, inner=100, outer=100),
        _row(1, 0.3, total=120, inner=110, outer=10),  # inner >> outer
        _row(2, 0.7, total=200, inner=100, outer=100),
        _row(3, 1.0, total=200, inner=100, outer=100),
    ]
    new_radii, _ = _propose_anchor_radii(
        rows, radii, imbalance_ratio=5.0, min_gap=0.05)
    # Outer gap of anchor 1 is (0.3, 0.7); midpoint = 0.5
    assert 0.5 in new_radii


def test_min_gap_blocks_insert_when_neighbors_too_close():
    """If both adjacent gaps are below `2 * min_gap`, no insert is possible.
    This protects against runaway anchor proliferation."""
    load_config(None, {'bottleneck_threshold': 30})
    radii = [0.10, 0.12, 0.14, 1.0]
    rows = [
        _row(0, 0.10, total=200, inner=100, outer=100),
        _row(1, 0.12, total=10,  inner=5,   outer=5),    # bottleneck
        _row(2, 0.14, total=200, inner=100, outer=100),
        _row(3, 1.00, total=200, inner=100, outer=100),
    ]
    new_radii, decisions = _propose_anchor_radii(
        rows, radii, imbalance_ratio=5.0, min_gap=0.05)
    # Gaps either side of anchor 1 are 0.02; 2*min_gap = 0.10 → no room.
    # Radii past anchor 2 (0.86 gap) are not its neighbors and can't be split.
    assert new_radii == radii
    # Decision text should say "no room"
    anchor1_decision = next(d for i, d, _ in decisions if i == 1)
    assert 'no room' in anchor1_decision


def test_no_data_rows_pass_through():
    load_config(None, {})
    radii = [0.1, 0.3, 1.0]
    rows = [
        _row(0, 0.1, total=200, inner=100, outer=100),
        _row(1, 0.3, total=0,   inner=0,   outer=0, flag='NO DATA'),
        _row(2, 1.0, total=200, inner=100, outer=100),
    ]
    new_radii, decisions = _propose_anchor_radii(
        rows, radii, imbalance_ratio=5.0, min_gap=0.05)
    assert new_radii == radii
    # Decision for the NO DATA row says 'no data'
    assert decisions[1] == (1, 'no data', None)


def test_no_duplicate_inserts():
    """Two adjacent flagged anchors must not both propose the same midpoint."""
    load_config(None, {'bottleneck_threshold': 30})
    radii = [0.1, 0.3, 0.5, 0.7]
    rows = [
        _row(0, 0.1, total=200, inner=100, outer=100),
        _row(1, 0.3, total=10,  inner=5,   outer=5),    # bottleneck
        _row(2, 0.5, total=10,  inner=5,   outer=5),    # bottleneck
        _row(3, 0.7, total=200, inner=100, outer=100),
    ]
    new_radii, _ = _propose_anchor_radii(
        rows, radii, imbalance_ratio=5.0, min_gap=0.05)
    # No duplicate radii and result is sorted
    assert new_radii == sorted(set(new_radii))


def test_min_gap_float_tolerance_for_uniform_grid():
    """Uniform 0.1 grid: 1.0-0.9 evaluates to 0.0999...8 in float, which
    would falsely fail `gap < 2 * min_gap` at min_gap=0.05 without the
    1e-9 tolerance in adjust.py."""
    load_config(None, {'bottleneck_threshold': 30})
    radii = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    rows = [_row(i, r, total=200, inner=100, outer=100)
            for i, r in enumerate(radii)]
    rows[5] = _row(5, 0.6, total=10, inner=5, outer=5)   # bottleneck mid-grid
    new_radii, _ = _propose_anchor_radii(
        rows, radii, imbalance_ratio=5.0, min_gap=0.05)
    # 0.55 or 0.65 should appear (split inner or outer gap of anchor 5).
    assert any(r in new_radii for r in (0.55, 0.65))
