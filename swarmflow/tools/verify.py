"""
Geometry summary of the equilibrated bound states for all 4 HIDR campaigns:

  A: complex-equil.pdb           (no transform; from equil)
  B: complex-equil-orient.pdb    (orientation flipped)
  C: complex-equil-side.pdb      (side flipped, orientation preserved)
  D: complex-equil-both.pdb      (both flipped)
                                 (legacy: complex-equil-down.pdb → D)

For each that exists, prints COM-COM distance, signed projection on the
host O2→O6 axis, perpendicular component, guest-axis vs host-axis angle,
and minimum heavy-atom contact. Also writes a host+guest-only PDB per
campaign for visual comparison in PyMOL.
"""

import json
import sys
from pathlib import Path

from .._config import C, paths


def _summarize(label, prmtop, pdb, json_path, host_resname, guest_resname):
    import numpy as np
    import parmed as pmd
    from scipy.spatial.distance import cdist

    cfg  = json.loads(Path(json_path).read_text())
    meta = cfg['_orientation_meta']
    O2_idx = meta['host_O2_indices']
    O6_idx = meta['host_O6_indices']
    g_axis_idx = meta['guest_axis_indices']

    struct = pmd.load_file(str(prmtop), str(pdb))
    coords = np.array([[a.xx, a.xy, a.xz] for a in struct.atoms])

    host_idx  = [i for i, a in enumerate(struct.atoms) if a.residue.name == host_resname]
    guest_idx = [i for i, a in enumerate(struct.atoms) if a.residue.name == guest_resname]
    heavy_h   = [i for i in host_idx  if struct.atoms[i].element != 1]
    heavy_g   = [i for i in guest_idx if struct.atoms[i].element != 1]

    O2_cen = coords[O2_idx].mean(0)
    O6_cen = coords[O6_idx].mean(0)
    host_axis = O6_cen - O2_cen
    host_axis /= np.linalg.norm(host_axis)

    h_com = coords[host_idx].mean(0)
    g_com = coords[guest_idx].mean(0)
    com_vec = g_com - h_com
    com_dist = float(np.linalg.norm(com_vec))
    proj = float(com_vec @ host_axis)
    perp = float(np.linalg.norm(com_vec - proj * host_axis))

    g_axis = coords[g_axis_idx[1]] - coords[g_axis_idx[0]]
    g_axis /= np.linalg.norm(g_axis)
    angle = float(np.degrees(np.arccos(np.clip(host_axis @ g_axis, -1, 1))))

    min_d = float(cdist(coords[heavy_h], coords[heavy_g]).min())
    side = 'O6 (primary)' if proj > 0 else 'O2 (secondary)'

    print(f'\n  {label}:')
    print(f'    COM-COM distance         : {com_dist:6.2f} Å')
    print(f'    along O2→O6 axis (signed): {proj:+6.2f} Å   [side: {side}]')
    print(f'    perpendicular            : {perp:6.2f} Å')
    print(f'    guest-axis vs host-axis  : {angle:6.1f}°')
    print(f'    min heavy-atom contact   : {min_d:6.2f} Å')
    return struct


def _write_hostguest(struct, host_resname, guest_resname, out: Path):
    sub = struct[f':{host_resname},{guest_resname}']
    sub.save(str(out), overwrite=True)
    print(f'    -> {out}')


def cmd_verify_bound(args):
    P = paths()
    if not P.json.exists():
        sys.exit(f'[verify] {P.json} not found — run setup first')

    work = P.work
    candidates = [
        ('A: original',                 work / 'complex-equil.pdb',
         work / 'view_A_hostguest.pdb'),
        ('B: orient flipped',           work / 'complex-equil-orient.pdb',
         work / 'view_B_hostguest.pdb'),
        ('C: side flipped (orient kept)', work / 'complex-equil-side.pdb',
         work / 'view_C_hostguest.pdb'),
        ('D: both flipped',             work / 'complex-equil-both.pdb',
         work / 'view_D_hostguest.pdb'),
    ]

    legacy = work / 'complex-equil-down.pdb'
    if legacy.exists() and not (work / 'complex-equil-both.pdb').exists():
        candidates[3] = ('D: both flipped (legacy: complex-equil-down.pdb)',
                         legacy, work / 'view_D_hostguest.pdb')

    print(f'[verify] work_dir = {work.resolve()}')
    print(f'[verify] using SMARTS metadata from {P.json.name}')

    found = False
    for label, pdb_path, view_out in candidates:
        if not pdb_path.exists():
            print(f'\n  {label}: {pdb_path.name} not found — skipping')
            continue
        struct = _summarize(label, P.solvated_top, pdb_path, P.json,
                            C.host_resname, C.guest_resname)
        _write_hostguest(struct, C.host_resname, C.guest_resname, view_out)
        found = True

    if not found:
        sys.exit('\n[verify] no equilibrated bound states found')

    files = ' '.join(str(t[2]) for t in candidates if t[1].exists())
    print(f'\nOpen all in PyMOL:\n  pymol {files}')
