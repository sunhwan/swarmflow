"""
Structural invariant: every name in `STAGES` resolves to a callable in
`STAGE_FNS`, and the documented `hidr_alts` back-compat alias still maps
to `stage_hidr_smd`. Catches the kind of mistake where someone adds a
stage name to STAGES but forgets to wire it into STAGE_FNS.
"""

import importlib

from swarmflow import STAGES, STAGE_FNS


def test_every_stage_in_STAGES_has_a_callable_in_STAGE_FNS():
    for name in STAGES:
        assert name in STAGE_FNS, f'{name!r} in STAGES but not STAGE_FNS'
        assert callable(STAGE_FNS[name])


def test_hidr_alts_alias_points_to_hidr_smd():
    """Back-compat: `--stage hidr_alts` historically meant SMD-style HIDR.
    The alias is documented in CLAUDE.md."""
    assert 'hidr_alts' in STAGE_FNS
    assert STAGE_FNS['hidr_alts'] is STAGE_FNS['hidr_smd']


def test_each_stage_module_exposes_its_stage_function():
    """`__init__.py` wires `from .<stage> import stage_<stage>`. If the
    module name and exposed function don't match the canonical convention,
    new stages won't drop in cleanly."""
    for name in STAGES:
        mod = importlib.import_module(f'swarmflow.{name}')
        fn_name = f'stage_{name}'
        assert hasattr(mod, fn_name), \
            f'swarmflow.{name} is missing {fn_name}()'
        assert callable(getattr(mod, fn_name))
