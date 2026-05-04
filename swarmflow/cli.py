"""
swarmflow command-line entry point.

Usage:
    swarmflow <stage> [options]              # run a single stage
    swarmflow run [--from X] [--to Y]        # run a range of stages
    swarmflow status                         # show pipeline progress
    swarmflow stages                         # list canonical stage order
    swarmflow init <dir>                     # scaffold a new project

Stage-specific flags live on the stage's own subparser — try
`swarmflow kinetics --help` for stage X to see only its options.

Examples:
    swarmflow setup
    swarmflow swarm --gpu 0,1 --mps-per-gpu 2
    swarmflow kinetics --num-error-samples 500 --skip-sliding-window
    swarmflow extract --frames last --every 1
    swarmflow run --from param --to setup
"""

import argparse
import warnings
from importlib.resources import files
from pathlib import Path

from swarmflow import STAGE_FNS, STAGES, load_config, paths
from swarmflow.tools import (cmd_clean, cmd_diagnose, cmd_merge_hidr,
                             cmd_verify_bound)
from swarmflow.tools.clean import MODES as _CLEAN_MODES

warnings.filterwarnings('ignore')


# ── Stage-specific argument adders ──────────────────────────────────────
# Each adder gets called on the subparser for its stage AND on the
# `run` subparser, so range invocations carry the stage's options too.

def _kinetics_args(p):
    g = p.add_argument_group('kinetics options')
    g.add_argument('--num-error-samples', dest='num_error_samples',
                   type=int, default=1000,
                   help='Bootstrap error samples (default: 1000)')
    g.add_argument('--n-blocks', dest='n_blocks', type=int, default=4,
                   help='Block-average count (default: 4 quarters)')
    g.add_argument('--n-windows', dest='n_windows', type=int, default=30,
                   help='Sliding-window count (default: 30)')
    g.add_argument('--skip-block-average', dest='skip_block_average',
                   action='store_true',
                   help='Skip the block-average convergence test')
    g.add_argument('--skip-sliding-window', dest='skip_sliding_window',
                   action='store_true',
                   help='Skip the sliding-window k_off plot')


def _extract_args(p):
    g = p.add_argument_group('extract options')
    g.add_argument('--frames', default='last',
                   help='last|first|all|N|N:M|N,M,... (default: last)')
    g.add_argument('--every', type=int, default=1,
                   help='Subsample every Nth frame (default: 1)')
    g.add_argument('--full', action='store_true',
                   help='Include solvent + ions (default: host+guest only)')


_STAGE_ARGS = {
    'kinetics': _kinetics_args,
    'extract':  _extract_args,
}


# ── Global options (config + overrides), added to every stage subparser ─
# Args here are partitioned at run-time: keys in _CLI_ONLY are CLI scaffolding
# and never become C overrides; everything else (e.g. --name, --gpu) is
# applied as a config override after YAML load.

_CLI_ONLY = {
    'config', 'command', 'from_', 'to',
    'frames', 'every', 'full',
    'num_error_samples', 'n_blocks', 'n_windows',
    'skip_block_average', 'skip_sliding_window',
    'pmf_bulk_ref_anchors',
    'project_dir', 'force',
    # Helper-tool flags (clean, diagnose, merge_hidr, verify_bound)
    'mode', 'yes', 'campaigns', 'by_anchor', 'out',
}


def _add_global_options(p):
    g = p.add_argument_group('config + overrides')
    g.add_argument('--config', default='config.yml',
                   help='YAML config file (default: config.yml)')
    g.add_argument('--name',      help='System name')
    g.add_argument('--input-pdb', dest='input_pdb', help='Input PDB')
    g.add_argument('--host-sdf',  dest='host_sdf',  help='Host SDF')
    g.add_argument('--guest-sdf', dest='guest_sdf', help='Guest SDF')
    g.add_argument('--work-dir',  dest='work_dir',  help='Work directory')
    g.add_argument('--gpu', dest='cuda_gpus',
                   help='Comma-separated GPU indices, e.g. 0,1')
    g.add_argument('--mps-per-gpu', dest='mps_per_gpu', type=int,
                   help='Anchors per GPU via MPS')


# ── Parser ──────────────────────────────────────────────────────────────
def _build_parser():
    p = argparse.ArgumentParser(
        prog='swarmflow',
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='command', required=True, metavar='COMMAND')

    # One subparser per stage
    for name in STAGES:
        sp = sub.add_parser(name, help=f'Run stage: {name}')
        _add_global_options(sp)
        if name in _STAGE_ARGS:
            _STAGE_ARGS[name](sp)

    # `run` — range execution; carries every stage's options so any
    # range can supply them.
    run_p = sub.add_parser(
        'run', help='Run a range of stages',
        description='Run a contiguous range of stages. Defaults: '
                    '--from = first stage, --to = last stage.')
    run_p.add_argument('--from', dest='from_', choices=STAGES,
                       help='First stage to run (inclusive)')
    run_p.add_argument('--to',   dest='to',   choices=STAGES,
                       help='Last stage to run (inclusive)')
    _add_global_options(run_p)
    for adder in _STAGE_ARGS.values():
        adder(run_p)

    # `status` — show which stages have produced their primary output
    st_p = sub.add_parser('status', help='Show pipeline progress')
    _add_global_options(st_p)

    # `stages` — list canonical stage order
    sub.add_parser('stages', help='List stages in canonical order')

    # `init` — scaffold a new project directory
    in_p = sub.add_parser('init', help='Scaffold a new project directory')
    in_p.add_argument('project_dir', help='Directory to create')
    in_p.add_argument('--name', help='System name (default: directory name)')
    in_p.add_argument('--force', action='store_true',
                      help='Overwrite existing config.yml')

    # ── Helper subcommands ────────────────────────────────────────────────
    cl_p = sub.add_parser('clean',
                          help='Remove specific stage outputs to allow re-runs')
    cl_p.add_argument('mode', choices=_CLEAN_MODES,
                      help='What to clean (see --help for descriptions)')
    cl_p.add_argument('-y', '--yes', action='store_true',
                      help='Skip the confirmation prompt')
    _add_global_options(cl_p)

    dg_p = sub.add_parser('diagnose',
                          help='Per-anchor HIDR campaign quality table')
    _add_global_options(dg_p)

    mg_p = sub.add_parser('merge_hidr',
                          help='Merge HIDR PDBs into multi-model views')
    mg_p.add_argument('--campaigns', default='all',
                      help='Comma-separated tags from {A,B,C,D} or "all" (default)')
    mg_p.add_argument('--by-anchor', dest='by_anchor', action='store_true',
                      help='Order models per-anchor (each anchor\'s 4 poses '
                           'side-by-side) instead of per-campaign')
    mg_p.add_argument('--out', help='Output PDB path (default: derived from selection)')
    _add_global_options(mg_p)

    vb_p = sub.add_parser('verify_bound',
                          help='Geometry summary of the 4 equilibrated bound states')
    _add_global_options(vb_p)

    return p


# ── Helpers ─────────────────────────────────────────────────────────────
def _config_overrides(args):
    return {k: v for k, v in vars(args).items()
            if k not in _CLI_ONLY and v is not None}


def _load_C(args):
    C = load_config(args.config, _config_overrides(args))
    Path(C.work_dir).mkdir(exist_ok=True)
    return C


# ── Command handlers ────────────────────────────────────────────────────
def _cmd_stage(stage_name, args):
    _load_C(args)
    STAGE_FNS[stage_name](args)


def _cmd_run(args):
    _load_C(args)
    start = STAGES.index(args.from_) if args.from_ else 0
    end   = STAGES.index(args.to) + 1 if args.to else len(STAGES)
    if start >= end:
        raise SystemExit(
            f'--from {args.from_!r} comes at or after --to {args.to!r} '
            f'in the stage order')
    for stage_name in STAGES[start:end]:
        print(f'\n── Stage: {stage_name} ──')
        STAGE_FNS[stage_name](args)


def _cmd_status(args):
    _load_C(args)
    P = paths()

    # Each entry: (stage, marker, description). marker is either a
    # concrete Path, or a callable returning bool, or None for stages
    # with no persistent output.
    has_root = lambda: P.root.exists()
    has_mmvt = lambda: (has_root() and any(P.root.glob(
        'anchor_*/prod/mmvt*.out')))
    has_hidr = lambda: (has_root() and any(P.root.glob(
        'anchor_*/building/hidr_metadyn_at_*_0.pdb')))
    has_orient = lambda: any(P.work.glob('orientation_*.csv'))

    rows = [
        ('param',      P.complex_dir / 'vac.prmtop',  'vacuum complex'),
        ('solvate',    P.solvated_top,                'solvated.prmtop'),
        ('equil',      P.equil_pdb,                   'complex-equil.pdb'),
        ('check',      None,                          '(no persistent output)'),
        ('setup',      P.root / 'model.xml',          'root/model.xml'),
        ('report',     P.work / 'benchmark_report.png', 'benchmark_report.png'),
        ('adjust',     None,                          '(prints to stdout)'),
        ('hidr_smd',   has_hidr,                      'anchor_*/building/hidr_*'),
        ('hidr_metad', has_hidr,                      'anchor_*/building/hidr_*'),
        ('swarm',      has_mmvt,                      'anchor_*/prod/mmvt*.out'),
        ('kinetics',   P.work / 'kinetics_pmf.csv',   'kinetics_pmf.csv'),
        ('extract',    None,                          '(user-specified frames)'),
        ('analyze',    has_orient,                    'orientation_*.csv'),
    ]

    print(f'work dir: {P.work}')
    print(f'{"stage":<12s}  {"status":<6s}  marker')
    print(f'{"-"*12}  {"-"*6}  {"-"*40}')
    for name, marker, descr in rows:
        if marker is None:
            ok = '—'
        elif callable(marker):
            ok = '✓' if marker() else '·'
        else:
            ok = '✓' if marker.exists() else '·'
        print(f'{name:<12s}  {ok:<6s}  {descr}')


def _cmd_stages(args):
    print('canonical stage order:')
    for i, name in enumerate(STAGES):
        print(f'  {i:2d}. {name}')


def _cmd_init(args):
    target = Path(args.project_dir)
    target.mkdir(parents=True, exist_ok=True)
    cfg_path = target / 'config.yml'
    if cfg_path.exists() and not args.force:
        raise SystemExit(f'{cfg_path} already exists; use --force to overwrite')
    name = args.name or target.resolve().name
    template = files('swarmflow').joinpath('templates/config.yml.template').read_text()
    cfg_path.write_text(template.replace('{name}', name))
    print(f'wrote {cfg_path}')
    print(f'next steps:')
    print(f'  cd {target}')
    print(f'  # drop input.pdb, host.sdf, guest.sdf into this dir, then:')
    print(f'  swarmflow run --to setup')


# ── Entry point ─────────────────────────────────────────────────────────
def main():
    args = _build_parser().parse_args()

    if args.command in STAGES:
        _cmd_stage(args.command, args)
    elif args.command == 'run':
        _cmd_run(args)
    elif args.command == 'status':
        _cmd_status(args)
    elif args.command == 'stages':
        _cmd_stages(args)
    elif args.command == 'init':
        _cmd_init(args)
    elif args.command == 'clean':
        _load_C(args)
        cmd_clean(args)
    elif args.command == 'diagnose':
        _load_C(args)
        cmd_diagnose(args)
    elif args.command == 'merge_hidr':
        _load_C(args)
        cmd_merge_hidr(args)
    elif args.command == 'verify_bound':
        _load_C(args)
        cmd_verify_bound(args)
    else:
        raise SystemExit(f'unknown command: {args.command!r}')

    print('\nDone.')


if __name__ == '__main__':
    main()
