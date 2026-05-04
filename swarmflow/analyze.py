"""Stage: host-axis/guest-axis orientation + min host-guest distance from trajectories."""

import csv
import json

import numpy as np

from ._config import C, paths


def stage_analyze(args):
    import MDAnalysis as mda
    from MDAnalysis.analysis.distances import distance_array
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    P = paths()
    assert P.json.exists(), 'Run setup stage first'
    cfg  = json.loads(P.json.read_text())
    meta = cfg['_orientation_meta']
    O2_idx  = meta['host_O2_indices']
    O6_idx  = meta['host_O6_indices']
    guest_axis_idx = meta['guest_axis_indices']    # [head, tail] — directed
    lig_idx = cfg['workflow']['ligand_indices']
    rec_idx = cfg['workflow']['receptor_indices']

    rows = []
    for anchor in range(len(C.anchor_radii)):
        prod = P.root / f'anchor_{anchor}' / 'prod'
        top = P.root / f'anchor_{anchor}' / 'building' / 'solvated.prmtop'
        # Filter out empty DCDs. seekr2 sometimes creates a restart-N DCD on
        # prepare() but doesn't write frames if the run was interrupted before
        # any reporting interval — leaving 0-byte files that crash MDAnalysis
        # with "premature EOF" on the header.
        dcds = [d for d in sorted(prod.glob('mmvt*.dcd')) if d.stat().st_size > 0]
        if not dcds:
            print(f'[analyze] anchor_{anchor}: no DCD, skipping')
            continue
        print(f'[analyze] anchor_{anchor}: {len(dcds)} DCD(s)')

        global_frame = 0
        for dcd in dcds:
            source = dcd.stem
            try:
                u = mda.Universe(str(top), str(dcd))
            except (OSError, IOError) as e:
                print(f'[analyze]   {dcd.name}: skipping ({e})')
                continue
            host_heavy  = u.select_atoms(f'resname {C.host_resname} and not name H*')
            guest_heavy = u.select_atoms(f'resname {C.guest_resname} and not name H*')

            for ts in u.trajectory:
                pos = u.atoms.positions
                O2_cen = pos[O2_idx].mean(0)
                O6_cen = pos[O6_idx].mean(0)
                host_ax = O6_cen - O2_cen
                host_ax /= np.linalg.norm(host_ax)

                lig_cen = pos[lig_idx].mean(0)
                rec_cen = pos[rec_idx].mean(0)
                com_dist_nm = np.linalg.norm(lig_cen - rec_cen) / 10.0

                guest_ax = pos[guest_axis_idx[1]] - pos[guest_axis_idx[0]]
                guest_ax /= np.linalg.norm(guest_ax)

                cos_a = np.clip(np.dot(host_ax, guest_ax), -1, 1)
                angle = np.degrees(np.arccos(cos_a))

                d_arr = distance_array(host_heavy.positions, guest_heavy.positions)
                min_dist_ang = float(d_arr.min())

                rows.append({'anchor': anchor, 'radius_nm': C.anchor_radii[anchor],
                             'source': source,
                             'frame': global_frame,
                             'com_dist_nm':  round(com_dist_nm, 4),
                             'angle_deg':    round(angle, 2),
                             'min_dist_ang': round(min_dist_ang, 3)})
                global_frame += 1

    if not rows:
        print('[analyze] no trajectory data found')
        return

    out_csv = P.work / 'orientation_analysis.csv'
    with open(out_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f'[analyze] {len(rows)} frames -> {out_csv}')

    dists  = [r['com_dist_nm'] for r in rows]
    angles = [r['angle_deg']   for r in rows]

    fig, ax = plt.subplots(figsize=(7, 5))
    cmap = matplotlib.colormaps['viridis'].copy()
    cmap.set_under('white')   # empty bins (count < 1) render as white
    h = ax.hist2d(dists, angles, bins=[30, 36],
                  range=[[min(dists), max(dists)], [0, 180]],
                  cmap=cmap, vmin=1)
    plt.colorbar(h[3], ax=ax, label='frame count')
    ax.set_xlabel('COM-COM distance (nm)')
    ax.set_ylabel('Host(O2→O6) vs Guest axis angle (°)')
    ax.axhline(90, color='black', ls=':', lw=0.8, alpha=0.5)
    ax.set_yticks([0, 45, 90, 135, 180])
    ax.set_title(f'{C.name}: orientation vs distance')
    fig.tight_layout()
    out_hm = P.work / 'orientation_heatmap.png'
    fig.savefig(out_hm, dpi=150)
    plt.close(fig)
    print(f'[analyze] heatmap -> {out_hm}')

    fig, ax = plt.subplots(figsize=(10, 4))
    for anchor in sorted(set(r['anchor'] for r in rows)):
        sub = [r for r in rows if r['anchor'] == anchor]
        ax.plot([r['frame'] for r in sub], [r['angle_deg'] for r in sub],
                alpha=0.6, lw=0.8,
                label=f"a{anchor}({C.anchor_radii[anchor]}nm)")
    ax.set_xlabel('Frame')
    ax.set_ylabel('Axis angle (°)')
    ax.set_ylim(0, 180)
    ax.set_yticks([0, 45, 90, 135, 180])
    ax.axhline(90, color='gray', ls=':', lw=0.8, alpha=0.6)
    ax.set_title('Host-guest orientation per anchor')
    ax.legend(fontsize=6, ncol=4)
    fig.tight_layout()
    out_ang = P.work / 'orientation_per_anchor.png'
    fig.savefig(out_ang, dpi=150)
    plt.close(fig)
    print(f'[analyze] per-anchor -> {out_ang}')

    anchors_sorted = sorted(set(r['anchor'] for r in rows))
    print(f"\n  {'Anchor':>6}  {'Radius':>7}  {'Min':>6}  {'Mean':>6}  {'Median':>6}  (Å)")
    box_data, box_labels = [], []
    for anchor in anchors_sorted:
        d_anchor = np.array([r['min_dist_ang'] for r in rows if r['anchor'] == anchor])
        box_data.append(d_anchor)
        box_labels.append(str(anchor))
        print(f"  {anchor:>6}  {C.anchor_radii[anchor]:>7.2f}  "
              f"{d_anchor.min():>6.2f}  {d_anchor.mean():>6.2f}  "
              f"{np.median(d_anchor):>6.2f}")

    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(10, 6))
    ax0.boxplot(box_data, tick_labels=box_labels, showfliers=False)
    ax0.set_xlabel('Anchor')
    ax0.set_ylabel('Min host-guest dist (Å)')
    ax0.set_title('Closest heavy-atom contact per anchor')
    ax0.grid(axis='y', alpha=0.3)

    for anchor in anchors_sorted:
        sub = [r for r in rows if r['anchor'] == anchor]
        ax1.plot([r['frame'] for r in sub], [r['min_dist_ang'] for r in sub],
                 alpha=0.6, lw=0.8,
                 label=f"a{anchor}({C.anchor_radii[anchor]}nm)")
    ax1.set_xlabel('Frame')
    ax1.set_ylabel('Min dist (Å)')
    ax1.legend(fontsize=6, ncol=4)
    fig.tight_layout()
    out_mind = P.work / 'min_distance_per_anchor.png'
    fig.savefig(out_mind, dpi=150)
    plt.close(fig)
    print(f'[analyze] min-distance -> {out_mind}')
