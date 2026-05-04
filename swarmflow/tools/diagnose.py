"""
Per-anchor table of HIDR campaign quality.

For each campaign that exists (A=root, B=root_orient, C=root_side,
D=root_both), prints:

  - signed projection s = (g_com − h_com) · host_axis  in Å  (+ = primary side)
  - actual COM-COM distance |r|                         in Å
  - whether |r| lies inside the anchor's Voronoi cell (✓ / ✗)

✗ marks anchors whose deposited HIDR structure won't seed swarm correctly
(swarm.py rejects with the same 20 mÅ tolerance).
"""

import json
import re
import sys
from pathlib import Path

from .._config import C, paths


CAMPAIGNS = {
    'A': 'root',
    'B': 'root_orient',
    'C': 'root_side',
    'D': 'root_both',
}


def _collect_anchor_pdbs(root_dir: Path):
    pdbs = {}
    for p in root_dir.glob('anchor_*/building/hidr_*.pdb'):
        m_idx = re.search(r'anchor_(\d+)', p.parent.parent.name)
        m_rad = re.search(r'at_([0-9.]+)_', p.name)
        if not m_idx:
            continue
        idx = int(m_idx.group(1))
        rad = float(m_rad.group(1)) if m_rad else float('nan')
        pdbs[idx] = (rad, p)
    return pdbs


def _signed_side(prmtop, pdb, host_resname, guest_resname, O2_idx, O6_idx):
    import numpy as np
    import parmed as pmd
    struct = pmd.load_file(str(prmtop), str(pdb))
    coords = np.array([[a.xx, a.xy, a.xz] for a in struct.atoms])
    host_idx  = [i for i, a in enumerate(struct.atoms) if a.residue.name == host_resname]
    guest_idx = [i for i, a in enumerate(struct.atoms) if a.residue.name == guest_resname]
    O2_cen = coords[O2_idx].mean(0)
    O6_cen = coords[O6_idx].mean(0)
    host_axis = O6_cen - O2_cen
    host_axis /= np.linalg.norm(host_axis)
    h_com = coords[host_idx].mean(0)
    g_com = coords[guest_idx].mean(0)
    com_vec = g_com - h_com
    return float(com_vec @ host_axis), float(np.linalg.norm(com_vec))


def _cell_bounds_nm(idx: int, anchor_radii_nm):
    n = len(anchor_radii_nm)
    r_inner = 0.5 * (anchor_radii_nm[idx-1] + anchor_radii_nm[idx]) if idx > 0 else 0.0
    if idx + 1 < n:
        r_outer = 0.5 * (anchor_radii_nm[idx] + anchor_radii_nm[idx+1])
    else:
        last_step = (anchor_radii_nm[-1] - anchor_radii_nm[-2]
                     if n >= 2 else anchor_radii_nm[-1])
        r_outer = anchor_radii_nm[-1] + 0.5 * last_step
    return r_inner, r_outer


def cmd_diagnose(args):
    P = paths()
    if not P.json.exists():
        sys.exit(f'[diagnose] {P.json} not found — run setup first')
    cfg = json.loads(P.json.read_text())
    meta = cfg['_orientation_meta']
    O2_idx, O6_idx = meta['host_O2_indices'], meta['host_O6_indices']
    anchor_radii_nm = list(C.anchor_radii)

    campaign_pdbs = {}
    for tag, sub in CAMPAIGNS.items():
        d = P.work / sub
        if not d.exists():
            print(f'[diag] {tag} ({sub}/): not found, skipping')
            continue
        pdbs = _collect_anchor_pdbs(d)
        if not pdbs:
            print(f'[diag] {tag} ({sub}/): no PDBs')
            continue
        campaign_pdbs[tag] = pdbs
        print(f'[diag] {tag} ({sub}/): {len(pdbs)} anchors')

    if not campaign_pdbs:
        sys.exit('[diag] no HIDR campaigns found — run hidr_smd or hidr_metad first')

    all_idx = sorted(set().union(*[p.keys() for p in campaign_pdbs.values()]))
    active = [t for t in CAMPAIGNS if t in campaign_pdbs]

    print()
    print('  s = signed projection on host O2→O6 axis (Å):  − = secondary, + = primary')
    print('  |r| = COM-COM distance (Å), with ✓/✗ for cell containment vs anchor cell')
    print()

    hdr_top = f'  {"":>3}  {"":>5}  {"":>14}'
    hdr_sub = f'  {"α":>3}  {"r_α":>5}  {"cell (nm)":>14}'
    for tag in active:
        hdr_top += f'  {tag:^14}'
        hdr_sub += f'  {"s":>6} {"|r|":>5} {"✓":>1}'
    print(hdr_top)
    print(hdr_sub)
    print('  ' + '─' * (3 + 2 + 5 + 2 + 14 + 16 * len(active)))

    contained = {t: 0 for t in active}
    visited   = {t: 0 for t in active}
    rows = []

    for idx in all_idx:
        if idx >= len(anchor_radii_nm):
            continue
        r_alpha = anchor_radii_nm[idx]
        r_in, r_out = _cell_bounds_nm(idx, anchor_radii_nm)
        cell_str = f'[{r_in:.3f}, {r_out:.3f}]'
        line = f'  {idx:>3}  {r_alpha:>5.2f}  {cell_str:>14}'
        cells = []
        for tag in active:
            if idx not in campaign_pdbs[tag]:
                line += f'  {"--":^14}'
                cells.append((tag, None, None))
                continue
            _, pdb = campaign_pdbs[tag][idx]
            proj, dist = _signed_side(P.solvated_top, pdb,
                                      C.host_resname, C.guest_resname,
                                      O2_idx, O6_idx)
            visited[tag] += 1
            r_nm = dist / 10.0
            tol = 0.020   # matches swarm.py accept/reject tolerance
            inside = (r_in - tol <= r_nm <= r_out + tol)
            if inside:
                contained[tag] += 1
            mark = '✓' if inside else '✗'
            line += f'  {proj:+6.2f} {dist:>5.2f} {mark}'
            cells.append((tag, proj, dist))
        print(line)
        rows.append((idx, r_alpha, cells))

    print()
    print('  ── Cell containment ──')
    for tag in active:
        n_in, n_total = contained[tag], visited[tag]
        pct = (100.0 * n_in / n_total) if n_total else 0.0
        warn = '' if n_in == n_total else '  ⚠ some structures will be rejected by swarm'
        print(f'  {tag}: {n_in}/{n_total} anchors inside cell ({pct:.0f}%){warn}')

    if 'A' in campaign_pdbs:
        print()
        print('  ── Side direction (vs A) ──')
        for tag in active:
            if tag == 'A':
                continue
            same = opp = 0
            for _, _, cells in rows:
                a_proj = next((p for t, p, _ in cells if t == 'A'), None)
                t_proj = next((p for t2, p, _ in cells if t2 == tag), None)
                if a_proj is None or t_proj is None:
                    continue
                if (a_proj > 0) == (t_proj > 0):
                    same += 1
                else:
                    opp += 1
            print(f'  {tag} vs A:  {same} same-side, {opp} opposite')
