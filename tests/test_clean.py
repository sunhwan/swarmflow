"""
Tests for `tools.clean._build_targets`.

This module deletes files. The risk is one mode quietly listing files
that belong to another stage. Each test sets up a fake work tree under
tmp_path and asserts mode-specific containment / non-containment.
"""

import pytest

from swarmflow._config import C, load_config
from swarmflow.tools.clean import _build_targets


@pytest.fixture
def fake_work(tmp_path):
    """Build a representative populated work tree and load C to point at it."""
    work = tmp_path / 'work_X'
    work.mkdir()

    # Stage outputs
    (work / 'solvated.prmtop').write_text('')
    (work / 'solvated.rst7').write_text('')
    (work / 'solvated.pdb').write_text('')
    (work / 'complex-equil.pdb').write_text('')
    (work / 'complex-equil.rst7').write_text('')
    (work / 'complex-equil-orient.pdb').write_text('')
    (work / 'complex-equil-side.pdb').write_text('')
    (work / 'complex-equil-both.pdb').write_text('')
    (work / 'seekrflow_X.json').write_text('{}')

    # Param + complex
    (work / 'complex').mkdir()
    (work / 'complex' / 'vac.prmtop').write_text('')

    # All four roots
    for sub in ('root', 'root_orient', 'root_side', 'root_both'):
        d = work / sub
        d.mkdir()
        (d / 'model.xml').write_text('')
        (d / 'anchor_0').mkdir()
        (d / 'anchor_0' / 'building').mkdir()
        (d / 'anchor_0' / 'prod').mkdir()

    # Swarm outputs under root/
    bld = work / 'root' / 'anchor_0' / 'building'
    prd = work / 'root' / 'anchor_0' / 'prod'
    (bld / 'state.swarm_0.xml').write_text('')
    (bld / 'state.swarm_1.xml').write_text('')
    (prd / 'mmvt.restart1.swarm_0.out').write_text('')
    (prd / 'mmvt1.swarm_0.dcd').write_text('')
    (prd / 'backup.checkpoint.swarm_0').write_text('')

    # Post-setup artifacts
    (work / 'benchmark_report.png').write_text('')
    (work / 'kinetics_pmf.csv').write_text('')

    load_config(None, {'name': 'X', 'work_dir': str(work)})
    return work


def _to_str_set(targets):
    return {str(t) for t in targets}


def test_swarm_mode_only_targets_swarm_outputs(fake_work):
    """`clean swarm` must NOT touch HIDR roots, equil files, or solvated.*."""
    targets = _to_str_set(_build_targets('swarm', fake_work))

    # Must include
    assert str(fake_work / 'root' / 'anchor_0' / 'building'
               / 'state.swarm_0.xml') in targets
    assert str(fake_work / 'root' / 'anchor_0' / 'prod'
               / 'mmvt.restart1.swarm_0.out') in targets
    assert str(fake_work / 'root' / 'anchor_0' / 'prod'
               / 'mmvt1.swarm_0.dcd') in targets

    # Must NOT include — these belong to earlier stages
    assert str(fake_work / 'root') not in targets
    assert str(fake_work / 'root_orient') not in targets
    assert str(fake_work / 'complex-equil.pdb') not in targets
    assert str(fake_work / 'solvated.prmtop') not in targets
    assert str(fake_work / 'seekrflow_X.json') not in targets


def test_hidr_alts_keeps_root_main(fake_work):
    """`clean hidr-alts` removes campaigns B/C/D, keeps campaign A's root/."""
    targets = _to_str_set(_build_targets('hidr-alts', fake_work))

    assert str(fake_work / 'root_orient') in targets
    assert str(fake_work / 'root_side') in targets
    assert str(fake_work / 'root_both') in targets
    # The crucial guarantee: root/ itself is NOT touched
    assert str(fake_work / 'root') not in targets
    # Per-campaign equil PDBs are wiped
    assert str(fake_work / 'complex-equil-orient.pdb') in targets
    # Main equil PDB is preserved
    assert str(fake_work / 'complex-equil.pdb') not in targets


def test_hidr_mode_targets_all_four_roots(fake_work):
    targets = _to_str_set(_build_targets('hidr', fake_work))
    for sub in ('root', 'root_orient', 'root_side', 'root_both'):
        assert str(fake_work / sub) in targets
    # Solvated & equil are kept
    assert str(fake_work / 'solvated.prmtop') not in targets
    assert str(fake_work / 'complex-equil.pdb') not in targets


def test_post_equil_keeps_solvated_and_param(fake_work):
    """`clean post-equil` should remove HIDR + equil, keep upstream stages."""
    targets = _to_str_set(_build_targets('post-equil', fake_work))

    assert str(fake_work / 'complex-equil.pdb') in targets
    assert str(fake_work / 'root') in targets
    # Upstream survivors:
    assert str(fake_work / 'solvated.prmtop') not in targets
    assert str(fake_work / 'complex' / 'vac.prmtop') not in targets


def test_post_solvate_keeps_param_only(fake_work):
    targets = _to_str_set(_build_targets('post-solvate', fake_work))

    assert str(fake_work / 'solvated.prmtop') in targets
    assert str(fake_work / 'complex-equil.pdb') in targets
    assert str(fake_work / 'root') in targets
    # complex/ from param stage stays
    assert str(fake_work / 'complex' / 'vac.prmtop') not in targets
    assert str(fake_work / 'complex') not in targets


def test_all_mode_targets_only_work_dir(fake_work):
    targets = _build_targets('all', fake_work)
    assert len(targets) == 1
    assert str(targets[0]) == str(fake_work)


def test_no_targets_when_work_is_empty(tmp_path):
    work = tmp_path / 'work_empty'
    work.mkdir()
    load_config(None, {'name': 'empty', 'work_dir': str(work)})
    # No mode should produce any targets when nothing exists
    for mode in ('swarm', 'hidr', 'hidr-alts', 'post-equil',
                 'post-solvate', 'views'):
        assert _build_targets(mode, work) == [], f'{mode}: expected []'


def test_targets_are_unique(fake_work):
    targets = _build_targets('post-solvate', fake_work)
    assert len(targets) == len({str(t) for t in targets}), \
        'duplicates in target list — _build_targets dedupe broken'
