"""
Tests for `_config.load_config`.

The two invariants worth pinning:

1. Derived `production_timestep_ps` and `production_steps_per_anchor` track
   `hmr_enabled` × `production_ns_per_anchor` exactly. CLAUDE.md flags this:
   "Stages must read those — never recompute step counts from ns themselves,
   or HMR toggling will silently halve simulation time."

2. `C` is mutated in place, never reassigned, so a reference grabbed before
   `load_config` reflects post-load values. Modules across the package do
   `from ._config import C` at import time and rely on this.
"""

import pytest

from swarmflow._config import C, DEFAULTS, load_config


def test_default_timestep_and_steps_no_hmr():
    load_config(None, {})
    assert C.production_timestep_ps == 0.002
    # ns=1.0 default, dt=2 fs → 500_000 steps
    assert C.production_steps_per_anchor == 500_000


def test_hmr_doubles_timestep_and_halves_steps():
    load_config(None, {'hmr_enabled': True})
    assert C.production_timestep_ps == 0.004
    # ns=1.0, dt=4 fs → 250_000 steps
    assert C.production_steps_per_anchor == 250_000


@pytest.mark.parametrize('hmr,ns,expected_dt,expected_steps', [
    (False, 0.1,  0.002,  50_000),
    (False, 1.0,  0.002, 500_000),
    (False, 2.5,  0.002, 1_250_000),
    (True,  0.1,  0.004,  25_000),
    (True,  1.0,  0.004, 250_000),
    (True,  2.5,  0.004, 625_000),
])
def test_steps_derived_from_ns_and_hmr(hmr, ns, expected_dt, expected_steps):
    load_config(None, {'hmr_enabled': hmr, 'production_ns_per_anchor': ns})
    assert C.production_timestep_ps == expected_dt
    assert C.production_steps_per_anchor == expected_steps


def test_C_is_mutated_in_place_not_reassigned():
    """Module-level `from ._config import C` references must stay valid
    after load_config. Reassigning C would orphan every importing module."""
    load_config(None, {'name': 'first'})
    ref = C
    ref_id = id(C)

    load_config(None, {'name': 'second'})
    assert id(C) == ref_id
    assert ref is C
    assert ref.name == 'second'


def test_cli_overrides_beat_defaults():
    load_config(None, {'name': 'override-name', 'box_buffer_ang': 20.0})
    assert C.name == 'override-name'
    assert C.box_buffer_ang == 20.0


def test_yaml_overrides_defaults_then_cli_overrides_yaml(tmp_path):
    cfg = tmp_path / 'config.yml'
    cfg.write_text('name: from-yaml\nbox_buffer_ang: 15.0\n')

    load_config(str(cfg), {})
    assert C.name == 'from-yaml'
    assert C.box_buffer_ang == 15.0

    load_config(str(cfg), {'box_buffer_ang': 25.0})
    assert C.name == 'from-yaml'        # CLI didn't touch this; YAML wins over default
    assert C.box_buffer_ang == 25.0     # CLI wins over YAML


def test_none_cli_overrides_do_not_clobber_yaml(tmp_path):
    """argparse populates absent flags as None. Those Nones must not
    overwrite YAML values — only explicitly-set CLI flags should override."""
    cfg = tmp_path / 'config.yml'
    cfg.write_text('name: from-yaml\n')

    load_config(str(cfg), {'name': None, 'box_buffer_ang': None})
    assert C.name == 'from-yaml'
    assert C.box_buffer_ang == DEFAULTS['box_buffer_ang']


def test_cuda_gpus_string_is_split_to_list():
    load_config(None, {'cuda_gpus': '0,1,2'})
    assert C.cuda_gpus == ['0', '1', '2']


def test_cuda_gpus_string_strips_whitespace_and_drops_empties():
    load_config(None, {'cuda_gpus': ' 0 , , 1 '})
    assert C.cuda_gpus == ['0', '1']


def test_cuda_gpus_list_passes_through():
    load_config(None, {'cuda_gpus': ['3', '4']})
    assert C.cuda_gpus == ['3', '4']


def test_work_dir_defaults_from_name():
    load_config(None, {'name': 'foo'})
    assert C.work_dir == 'work_foo'


def test_work_dir_explicit_wins():
    load_config(None, {'name': 'foo', 'work_dir': '/tmp/custom'})
    assert C.work_dir == '/tmp/custom'


def test_missing_config_file_falls_back_to_defaults(tmp_path):
    load_config(str(tmp_path / 'does-not-exist.yml'), {})
    assert C.name == DEFAULTS['name']
