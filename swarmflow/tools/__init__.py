"""
Helper subcommands for inspecting / cleaning a swarmflow project. Each
module exposes `def cmd_<name>(args): ...` and is wired into cli.py.

Heavy imports (parmed, MDAnalysis) are deferred into the command bodies so
`swarmflow --help` / `status` / `stages` don't pay for them.
"""

from .clean      import cmd_clean
from .diagnose   import cmd_diagnose
from .merge_hidr import cmd_merge_hidr
from .verify     import cmd_verify_bound

__all__ = ['cmd_clean', 'cmd_diagnose', 'cmd_merge_hidr', 'cmd_verify_bound']
