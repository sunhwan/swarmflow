"""
Clean specific outputs of a swarmflow project so a stage can be re-run.

System-agnostic via `paths()` and `C.name` — no hardcoded work_dir or
filenames. Lists targets first; asks confirmation unless --yes.

Modes:
  from-setup    Wipe everything anchor-dependent (seekrflow JSON, all roots,
                per-campaign equil PDBs, plots). Keep param/, complex/,
                solvated.*, complex-equil.*. Use after editing anchor_radii.
  hidr          Remove all 4 HIDR campaigns + per-campaign equil files.
                Keep solvated, equil. Re-run hidr_smd or hidr_metad next.
  hidr-alts     Remove only the alt campaigns (root_orient, root_side,
                root_both, legacy root_down). Keep root/. Re-run hidr_smd.
  swarm         Remove cached swarm states + MMVT outputs. Keep HIDR.
  post-equil    Remove everything past the equil stage. Keep param/,
                complex/, solvated.*, complex-equil.*.
  post-solvate  Remove everything past the solvate stage. Keep param/,
                complex/, solvated.*.
  views         Remove view_*.pdb visualization artifacts only.
  all           Wipe the entire work directory.
"""

import shutil
import sys
from pathlib import Path

from .._config import C, paths


MODES = ('from-setup', 'hidr', 'hidr-alts', 'swarm', 'post-equil',
         'post-solvate', 'views', 'all')

_ALT_TAGS = ('orient', 'side', 'both', 'down')   # 'down' = legacy of 'both'
_ALT_ROOTS = ('root_orient', 'root_side', 'root_both', 'root_down')
_ALL_ROOTS = ('root',) + _ALT_ROOTS


def _exists(p: Path) -> bool:
    return p.exists() or p.is_symlink()


def _glob(work: Path, pattern: str):
    return [p for p in work.glob(pattern) if _exists(p)]


def _add(targets: list, *paths_):
    for p in paths_:
        if _exists(p):
            targets.append(p)


def _hidr_alt_files(work: Path) -> list:
    """Per-campaign equil PDBs/rst7s + (legacy) per-campaign JSONs."""
    out = []
    for tag in _ALT_TAGS:
        out += _glob(work, f'complex-equil-{tag}.pdb')
        out += _glob(work, f'complex-equil-{tag}.rst7')
        # Legacy seekrflow-era per-campaign JSON, if a stale install left one
        out += _glob(work, f'seekrflow_{C.name}_{tag}.json')
    return out


def _post_setup_artifacts(work: Path) -> list:
    """Aggregate plots / CSVs / logs produced by stages downstream of setup."""
    return (
        _glob(work, 'benchmark_report.png')
        + _glob(work, 'anchor_radii_adjusted.yml')
        + _glob(work, 'kinetics_pmf.csv')
        + _glob(work, 'kinetics_pmf.png')
        + _glob(work, 'kinetics_k_off_windows.csv')
        + _glob(work, 'kinetics_k_off_windows.png')
        + _glob(work, 'orientation_*.csv')
        + _glob(work, 'orientation_*.png')
        + _glob(work, 'min_distance_*.png')
        + _glob(work, 'view_*.pdb')
        + _glob(work, 'hidr_smd_*.log')
    )


def _swarm_outputs(work: Path) -> list:
    out = []
    out += list(work.glob('root/anchor_*/building/state.swarm_*.xml'))
    out += list(work.glob('root/anchor_*/prod/mmvt.restart1.swarm_*.out'))
    out += list(work.glob('root/anchor_*/prod/mmvt1.swarm_*.dcd'))
    out += list(work.glob('root/anchor_*/prod/backup.checkpoint.swarm_*'))
    return out


def _build_targets(mode: str, work: Path) -> list:
    targets: list = []

    if mode == 'from-setup':
        _add(targets, paths().json)
        for sub in _ALL_ROOTS:
            _add(targets, work / sub)
        targets += _hidr_alt_files(work)
        targets += _post_setup_artifacts(work)

    elif mode == 'hidr':
        for sub in _ALL_ROOTS:
            _add(targets, work / sub)
        targets += _hidr_alt_files(work)
        targets += _glob(work, 'hidr_smd_*.log')

    elif mode == 'hidr-alts':
        for sub in _ALT_ROOTS:
            _add(targets, work / sub)
        targets += _hidr_alt_files(work)
        targets += _glob(work, 'hidr_smd_*.log')
        # Cached swarm states reference the now-stale HIDR; clear them too
        targets += list(work.glob('root/anchor_*/building/state.swarm_*.xml'))

    elif mode == 'swarm':
        targets += _swarm_outputs(work)
        _add(targets, work / '.swarm_logs')

    elif mode == 'post-equil':
        _add(targets, work / 'complex-equil.pdb', work / 'complex-equil.rst7')
        for sub in _ALL_ROOTS:
            _add(targets, work / sub)
        targets += _hidr_alt_files(work)
        _add(targets, paths().json)
        targets += _post_setup_artifacts(work)

    elif mode == 'post-solvate':
        _add(targets,
             work / 'solvated.prmtop', work / 'solvated.rst7',
             work / 'solvated.pdb',
             work / 'complex-equil.pdb', work / 'complex-equil.rst7',
             work / 'leap.log')
        for sub in _ALL_ROOTS:
            _add(targets, work / sub)
        targets += _hidr_alt_files(work)
        _add(targets, paths().json)
        targets += _glob(work, 'solvate*.tleap.in')
        targets += _post_setup_artifacts(work)

    elif mode == 'views':
        targets += _glob(work, 'view_*.pdb')

    elif mode == 'all':
        _add(targets, work)

    else:
        sys.exit(f'[clean] unknown mode: {mode}')

    # De-dupe while preserving order
    seen = set()
    out = []
    for t in targets:
        key = str(t)
        if key not in seen:
            seen.add(key)
            out.append(t)
    return out


def _human_size(p: Path) -> str:
    try:
        if p.is_dir():
            total = sum(f.stat().st_size for f in p.rglob('*') if f.is_file())
        else:
            total = p.stat().st_size
    except OSError:
        return '?'
    for unit in ('B', 'K', 'M', 'G', 'T'):
        if total < 1024:
            return f'{total:.0f}{unit}'
        total /= 1024
    return f'{total:.0f}P'


def cmd_clean(args):
    work = paths().work
    if not work.exists():
        sys.exit(f'[clean] work dir not found: {work}')

    targets = _build_targets(args.mode, work)
    if not targets:
        print(f"[clean] mode='{args.mode}' on {work} — nothing to delete.")
        return

    print(f"[clean] mode='{args.mode}' on {work} — would delete:")
    for t in targets:
        suffix = '/' if t.is_dir() else ''
        print(f'  {t}{suffix}   ({_human_size(t)})')

    if not args.yes:
        try:
            ans = input('Proceed? [y/N] ').strip().lower()
        except EOFError:
            ans = ''
        if ans not in ('y', 'yes'):
            print('[clean] aborted.')
            return

    for t in targets:
        if t.is_dir() and not t.is_symlink():
            shutil.rmtree(t, ignore_errors=True)
        else:
            try:
                t.unlink()
            except FileNotFoundError:
                pass
    print('[clean] done.')
