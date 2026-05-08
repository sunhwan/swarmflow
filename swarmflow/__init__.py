"""
swarmflow — stage-based driver for seekr2 milestoning + 4-member MMVT swarm.

Each stage module implements `stage_<name>(args)` and reads from `_config.C`,
operating on the `_config.paths()` directory tree.

Public API:
    STAGES      - canonical ordered list of stage names
    STAGE_FNS   - dict mapping stage name → callable
    DEFAULTS    - default config dict
    load_config - populate C from YAML + CLI overrides
    paths       - derived Path namespace from current C
"""

from ._config import C, DEFAULTS, STAGES, load_config, paths

from .param      import stage_param
from .solvate    import stage_solvate
from .equil      import stage_equil
from .check      import stage_check
from .setup      import stage_setup
from .report     import stage_report
from .adjust     import stage_adjust
from .hidr_smd   import stage_hidr_smd
from .hidr_metad import stage_hidr_metad
from .bd         import stage_bd
from .swarm      import stage_swarm
from .kinetics   import stage_kinetics
from .extract    import stage_extract
from .analyze    import stage_analyze

STAGE_FNS = {
    'param':      stage_param,
    'solvate':    stage_solvate,
    'equil':      stage_equil,
    'check':      stage_check,
    'setup':      stage_setup,
    'report':     stage_report,
    'adjust':     stage_adjust,
    'hidr_smd':   stage_hidr_smd,
    'hidr_metad': stage_hidr_metad,
    # Back-compat alias — old --stage hidr_alts still works
    'hidr_alts':  stage_hidr_smd,
    'bd':         stage_bd,
    'swarm':      stage_swarm,
    'kinetics':   stage_kinetics,
    'extract':    stage_extract,
    'analyze':    stage_analyze,
}

__all__ = ['STAGES', 'STAGE_FNS', 'DEFAULTS', 'C', 'load_config', 'paths']
