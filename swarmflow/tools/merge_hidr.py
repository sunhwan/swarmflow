"""
Merge HIDR-deposited anchor PDBs into multi-model PDBs for visual inspection
in PyMOL/VMD/ChimeraX. Per anchor: one model per campaign × anchor pose,
host+guest only, sorted either by campaign (default) or by anchor.

Default ordering (`--by campaign`): all of A's anchors, then all of B's, etc.
Useful for scrubbing through one campaign at a time.

Per-anchor ordering (`--by anchor`): for each anchor α, the available poses
across A,B,C,D side-by-side. This is the new replacement for the legacy
merge_swarm_pdbs.py preview — it shows the actual 4 starting poses the
swarm stage will use, instead of the obsolete synthetic-flip version.
"""

import re
import sys
from pathlib import Path

from .._config import paths


CAMPAIGNS = {
    'A': ('root',         None),
    'B': ('root_orient',  None),
    'C': ('root_side',    None),
    'D': ('root_both',    'root_down'),   # legacy fallback
}


def _collect_anchor_pdbs(root_dir: Path):
    if root_dir is None or not root_dir.exists():
        return []
    out = []
    for p in root_dir.glob('anchor_*/building/hidr_*.pdb'):
        m_idx = re.search(r'anchor_(\d+)', p.parent.parent.name)
        m_rad = re.search(r'at_([0-9.]+)_', p.name)
        if not m_idx:
            continue
        out.append((int(m_idx.group(1)),
                    m_rad.group(1) if m_rad else '?',
                    p))
    return sorted(out, key=lambda t: t[0])


def _resolve_root(work: Path, primary: str, fallback):
    p = work / primary
    if p.exists():
        return p
    if fallback:
        f = work / fallback
        if f.exists():
            print(f'[merge] using legacy {fallback}/ for primary {primary}/')
            return f
    return None


def _write_merged(sources, out_pdb: Path, by_anchor: bool):
    """sources: list of (tag, [(idx, rad, pdb), ...])."""
    import MDAnalysis as mda

    if not sources:
        sys.exit('[merge] nothing to merge — run hidr_smd or hidr_metad first')

    first_pdb = sources[0][1][0][2]
    u = mda.Universe(str(first_pdb))
    print(f'[merge] {u.atoms.n_atoms} atoms (from {first_pdb})')

    # Assemble (tag, idx, rad, pdb) flat list in the chosen order
    flat = []
    if by_anchor:
        all_idx = sorted({idx for _, anchors in sources for idx, _, _ in anchors})
        per_tag = {tag: {idx: (rad, pdb) for idx, rad, pdb in anchors}
                   for tag, anchors in sources}
        for idx in all_idx:
            for tag, _ in sources:
                if idx in per_tag[tag]:
                    rad, pdb = per_tag[tag][idx]
                    flat.append((tag, idx, rad, pdb))
    else:
        for tag, anchors in sources:
            for idx, rad, pdb in anchors:
                flat.append((tag, idx, rad, pdb))

    n = 0
    with mda.Writer(str(out_pdb), u.atoms.n_atoms, multiframe=True,
                    bonds=None, reindex=True) as W:
        for tag, idx, rad, pdb in flat:
            u_i = mda.Universe(str(pdb))
            if u_i.atoms.n_atoms != u.atoms.n_atoms:
                print(f'[merge] SKIP {pdb}: atom count mismatch')
                continue
            u.atoms.positions = u_i.atoms.positions
            W.write(u.atoms)
            n += 1
            print(f'  model {n:3d}  {tag}  anchor_{idx:02d}  r={rad}')
    print(f'[merge] wrote {n} models -> {out_pdb}')


def cmd_merge_hidr(args):
    P = paths()
    if args.campaigns.lower() in ('all', '*'):
        wanted = list(CAMPAIGNS)
    else:
        wanted = [s.strip().upper() for s in args.campaigns.split(',') if s.strip()]
        for w in wanted:
            if w not in CAMPAIGNS:
                sys.exit(f'[merge] unknown campaign tag: {w} (choose from A,B,C,D)')

    sources = []
    for tag in wanted:
        primary, fallback = CAMPAIGNS[tag]
        root_dir = _resolve_root(P.work, primary, fallback)
        anchors = _collect_anchor_pdbs(root_dir)
        if anchors:
            sources.append((tag, anchors))
            print(f'[merge] campaign {tag}: {len(anchors)} anchors in {root_dir}')
        else:
            print(f'[merge] campaign {tag}: no PDBs in {primary}/'
                  + (f' or {fallback}/' if fallback else ''))

    if not sources:
        sys.exit('[merge] nothing to merge — run hidr_smd or hidr_metad first')

    if args.out:
        out_pdb = Path(args.out)
    else:
        order_tag = 'byanchor' if args.by_anchor else 'bycampaign'
        tags_str = ''.join(t for t, _ in sources).lower()
        out_pdb = P.work / f'view_anchors_{tags_str}_{order_tag}.pdb'

    _write_merged(sources, out_pdb, by_anchor=args.by_anchor)
    print(f'\nOpen:  pymol {out_pdb}     # or  vmd {out_pdb}')
