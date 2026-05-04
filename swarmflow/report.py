"""Stage: per-anchor crossing statistics + bottleneck flagging."""

import numpy as np

from ._config import C, paths


def read_crossings(out_file):
    """Parse a seekr2 mmvt.restart1.out -> list of (boundary_id, time_ps)."""
    crossings = []
    with open(out_file) as f:
        for line in f:
            line = line.strip()
            if line.startswith('#') or line.startswith('CHECKPOINT') or not line:
                continue
            parts = line.split(',')
            if len(parts) == 3:
                crossings.append((int(parts[0]), float(parts[2])))
    return crossings


def _member_id(stem):
    """Extract swarm member id from .out file stem; '' for non-swarm runs."""
    if '.swarm_' in stem:
        for part in stem.split('.'):
            if part.startswith('swarm_'):
                return part
    return 'single'


def collect_crossing_stats():
    """Aggregate per-anchor crossings across ALL swarm members + restarts.

    Per anchor, walks every `mmvt*restart*.out` file under prod/, groups by
    swarm member, dedupes restart overlap within a member, and sums per-member
    cumulative sim times (members ran in parallel).
    """
    P = paths()
    rows = []
    for anchor in range(len(C.anchor_radii)):
        prod = P.root / f'anchor_{anchor}' / 'prod'
        empty = {'anchor': anchor, 'radius': C.anchor_radii[anchor],
                 'total': 0, 'inner': 0, 'outer': 0,
                 'sim_time_ps': 0.0, 'rate': 0.0,
                 'n_members': 0, 'flag': 'NO DATA'}
        if not prod.exists():
            rows.append(empty); continue

        member_files = {}
        for f in sorted(prod.glob('mmvt*restart*.out')):
            member_files.setdefault(_member_id(f.stem), []).append(f)
        if not member_files:
            rows.append(empty); continue

        total = inner = outer = 0
        total_time = 0.0
        n_members_with_data = 0
        for member, files in member_files.items():
            cx = []
            for f in files:
                cx.extend(read_crossings(f))
            if not cx:
                continue
            cx = sorted(set(cx))   # dedupe restart-overlap (boundary, time exact match)
            total += len(cx)
            inner += sum(1 for bid, _ in cx if bid == 1)
            outer += sum(1 for bid, _ in cx if bid == 2)
            total_time += cx[-1][1]   # cumulative time for this member
            n_members_with_data += 1

        if total == 0:
            rows.append(empty); continue
        rate = total / total_time if total_time > 0 else 0.0
        flag = 'BOTTLENECK' if total < C.bottleneck_threshold else 'ok'
        rows.append({'anchor': anchor, 'radius': C.anchor_radii[anchor],
                     'total': total, 'inner': inner, 'outer': outer,
                     'sim_time_ps': total_time, 'rate': rate,
                     'n_members': n_members_with_data, 'flag': flag})
    return rows


def stage_report(args):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    P = paths()
    rows = collect_crossing_stats()

    header = (f"{'Anchor':>6}  {'Radius':>7}  {'N':>3}  {'Total':>6}  "
              f"{'Inner':>6}  {'Outer':>6}  {'Time(ps)':>9}  {'Rate(/ps)':>9}  Status")
    print('\n' + '─' * len(header))
    print(header)
    print('─' * len(header))
    for r in rows:
        suffix = '  *** BOTTLENECK ***' if r['flag'] == 'BOTTLENECK' else ''
        print(f"{r['anchor']:>6}  {r['radius']:>7.2f}  {r['n_members']:>3}  "
              f"{r['total']:>6}  {r['inner']:>6}  {r['outer']:>6}  "
              f"{r['sim_time_ps']:>9.1f}  {r['rate']:>9.3f}  {r['flag']}{suffix}")
    print('─' * len(header))

    bottlenecks = [r for r in rows if r['flag'] == 'BOTTLENECK']
    if bottlenecks:
        print(f"\n  {len(bottlenecks)} bottleneck anchor(s): "
              + ', '.join(f"anchor_{r['anchor']} ({r['radius']}nm)" for r in bottlenecks))
    else:
        print(f"\n  All anchors ≥ {C.bottleneck_threshold} crossings.")

    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    anchors = [r['anchor'] for r in rows]
    colors  = ['tab:red' if r['flag'] == 'BOTTLENECK' else 'tab:blue' for r in rows]

    bars = ax0.bar(anchors, [r['total'] for r in rows], color=colors)
    ax0.axhline(C.bottleneck_threshold, color='red', ls='--', lw=1,
                label=f'threshold ({C.bottleneck_threshold})')
    ax0.set_ylabel('Total crossings')
    ax0.set_title('MMVT crossing statistics per anchor')
    ax0.legend(fontsize=8)
    for bar, r in zip(bars, rows):
        ax0.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                 f"{r['radius']:.2f}", ha='center', va='bottom', fontsize=6, rotation=90)

    x = np.array(anchors)
    ax1.bar(x - 0.2, [r['inner'] for r in rows], width=0.4,
            label='inner (boundary 1)', color='tab:blue', alpha=0.7)
    ax1.bar(x + 0.2, [r['outer'] for r in rows], width=0.4,
            label='outer (boundary 2)', color='tab:orange', alpha=0.7)
    ax1.set_xlabel('Anchor index')
    ax1.set_ylabel('Crossings')
    ax1.set_title('Inner vs outer milestone crossings')
    ax1.legend(fontsize=8)

    fig.tight_layout()
    out_png = P.work / 'benchmark_report.png'
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f'\n[report] chart -> {out_png}')
