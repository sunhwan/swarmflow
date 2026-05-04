"""Stage: verify the solvated box is large enough for the outermost anchor."""

import numpy as np

from ._config import C, paths


def _min_image_distance(box_vectors_ang: np.ndarray) -> float:
    """
    Smallest distance between two periodic images for an arbitrary
    triclinic box. For each pair of basis vectors, the perpendicular
    distance to the opposite face equals |volume| / |a_i × a_j|.
    """
    a, b, c = box_vectors_ang
    vol = abs(np.dot(a, np.cross(b, c)))
    d_a = vol / np.linalg.norm(np.cross(b, c))
    d_b = vol / np.linalg.norm(np.cross(a, c))
    d_c = vol / np.linalg.norm(np.cross(a, b))
    return min(d_a, d_b, d_c)


def stage_check(args):
    import parmed as pmd

    P = paths()
    assert P.solvated_top.exists(), 'Run solvate stage first'
    src = P.equil_pdb if P.equil_pdb.exists() else P.solvated_pdb
    print(f'[check] using {src}')
    struct = pmd.load_file(str(P.solvated_top), str(src))

    if struct.box is None:
        print('[check] FAIL: no periodic box info on structure')
        return
    a, b, c, alpha, beta, gamma = struct.box
    bv = np.array([
        [a, 0, 0],
        [b * np.cos(np.radians(gamma)),
         b * np.sin(np.radians(gamma)), 0],
        [c * np.cos(np.radians(beta)),
         c * (np.cos(np.radians(alpha)) - np.cos(np.radians(beta)) *
              np.cos(np.radians(gamma))) / np.sin(np.radians(gamma)),
         0],
    ])
    bv[2, 2] = np.sqrt(c ** 2 - bv[2, 0] ** 2 - bv[2, 1] ** 2)

    L_min = _min_image_distance(bv)

    coords = np.array([[a.xx, a.xy, a.xz] for a in struct.atoms])
    host_mask  = np.array([a.residue.name == C.host_resname  for a in struct.atoms])
    guest_mask = np.array([a.residue.name == C.guest_resname for a in struct.atoms])
    host_xyz, guest_xyz = coords[host_mask], coords[guest_mask]
    host_com, guest_com = host_xyz.mean(0), guest_xyz.mean(0)
    host_radius  = float(np.linalg.norm(host_xyz  - host_com,  axis=1).max())
    guest_radius = float(np.linalg.norm(guest_xyz - guest_com, axis=1).max())

    cutoff_ang = 9.0  # OpenMM nonbonded_cutoff (0.9 nm)
    R_max_ang  = max(C.anchor_radii) * 10.0

    required = host_radius + R_max_ang + guest_radius + cutoff_ang
    margin   = L_min - required

    print(f'[check] box vectors (Å): a={np.linalg.norm(bv[0]):.2f}, '
          f'b={np.linalg.norm(bv[1]):.2f}, c={np.linalg.norm(bv[2]):.2f}')
    print(f'[check] min image distance:        {L_min:>7.2f} Å')
    print(f'[check] host radius (max from COM): {host_radius:>7.2f} Å')
    print(f'[check] guest radius (max from COM):{guest_radius:>7.2f} Å')
    print(f'[check] outermost anchor radius:    {R_max_ang:>7.2f} Å '
          f'({max(C.anchor_radii)} nm)')
    print(f'[check] nonbonded cutoff:           {cutoff_ang:>7.2f} Å')
    print(f"[check] required ≥ {required:.2f} Å  "
          f"(host + R_max + guest + cutoff)")
    if margin >= 0:
        print(f'[check] OK — margin {margin:+.2f} Å')
    else:
        deficit = -margin
        suggested = C.box_buffer_ang + deficit / 2 + 1.0
        print(f'[check] FAIL — short by {deficit:.2f} Å')
        print(f'[check] try box_buffer_ang ≥ {suggested:.1f} '
              f'(currently {C.box_buffer_ang}) and re-run from --stage solvate')
