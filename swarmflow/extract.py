"""Stage: dump anchor trajectory snapshots as PDB."""

from ._config import C, paths


def _parse_frames(spec: str, n_frames: int) -> list:
    """Resolve a --frames spec (last|first|all|N|N:M|N,M,...) to indices."""
    if spec == 'last':
        return [n_frames - 1]
    if spec == 'first':
        return [0]
    if spec == 'all':
        return list(range(n_frames))
    if ':' in spec:
        a, b = spec.split(':', 1)
        a = int(a) if a else 0
        b = int(b) if b else n_frames
        return list(range(a, min(b, n_frames)))
    if ',' in spec:
        return [int(x) for x in spec.split(',') if x.strip()]
    return [int(spec)]


def stage_extract(args):
    import MDAnalysis as mda

    P = paths()
    out_dir = P.work / 'snapshots'
    out_dir.mkdir(exist_ok=True)

    frames_spec = getattr(args, 'frames', 'last') or 'last'
    every       = getattr(args, 'every', 1) or 1
    keep_full   = bool(getattr(args, 'full', False))

    if keep_full:
        sel_str = 'all'
        tag = 'full'
    else:
        sel_str = f'resname {C.host_resname} or resname {C.guest_resname}'
        tag = 'hostguest'

    print(f'[extract] frames={frames_spec}, every={every}, atoms={tag}')

    n_total = 0
    for anchor in range(len(C.anchor_radii)):
        dcd = P.root / f'anchor_{anchor}' / 'prod' / 'mmvt1.dcd'
        top = P.root / f'anchor_{anchor}' / 'building' / 'solvated.prmtop'
        if not dcd.exists():
            print(f'[extract] anchor_{anchor}: no DCD, skipping')
            continue

        u = mda.Universe(str(top), str(dcd))
        sel = u.select_atoms(sel_str)
        n_frames = len(u.trajectory)
        indices = _parse_frames(frames_spec, n_frames)
        indices = indices[::every] if every > 1 else indices

        for fi in indices:
            u.trajectory[fi]
            out_pdb = out_dir / f'anchor_{anchor}_frame_{fi:06d}_{tag}.pdb'
            sel.write(str(out_pdb))
            n_total += 1

        print(f'[extract] anchor_{anchor}: {len(indices)} snapshot(s) '
              f'({n_frames} total frames)')

    print(f'[extract] wrote {n_total} PDB(s) -> {out_dir}')
