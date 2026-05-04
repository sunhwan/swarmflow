"""
Stage: ΔG_bind from k_on/k_off, Jacobian-corrected anchor PMF, block-average
+ sliding-window k_off convergence.

Uses ONE seekr2.analyze.Analysis object across all sub-analyses
(headline, blocks, sliding window). The first extract_data() call reads
all .out files; subsequent calls hit the cached path
(analyze.py:348 `files_already_read=True`) and only re-run the per-line
min_time/max_time filter + small-matrix kinetics solve. Two orders of
magnitude faster than constructing a new Analysis for each block/window.
"""

import os
import time as _time

import numpy as np

from ._config import C, paths


def _run_block(analysis, t_min_ps, t_max_ps):
    """Re-run cached extract_data + fill_out + process for one time window.
    Returns the new k_off (or None on failure)."""
    try:
        analysis.extract_data(min_time=t_min_ps, max_time=t_max_ps)
        analysis.fill_out_data_samples()
        analysis.process_data_samples()
        return float(analysis.k_off)
    except Exception:
        return None


def stage_kinetics(args):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import seekr2.modules.common_base as common_base
    import seekr2.analyze as seekr2_analyze

    P = paths()
    model_xml = P.root / 'model.xml'
    assert model_xml.exists(), 'Run setup + production stages first (no model.xml)'

    # Resolve outputs to absolute before chdir — seekr2 reads relative paths
    # from model.xml so we must chdir(P.root); plot/csv writes go elsewhere.
    out_csv = (P.work / 'kinetics_pmf.csv').resolve()
    pmf_png = (P.work / 'kinetics_pmf.png').resolve()
    win_png = (P.work / 'kinetics_k_off_windows.png').resolve()
    win_csv = (P.work / 'kinetics_k_off_windows.csv').resolve()

    print(f'[kinetics] loading {model_xml}')
    curdir = os.getcwd()
    os.chdir(P.root)
    model = common_base.load_model('model.xml')
    n_err = int(getattr(args, 'num_error_samples', 1000) or 1000)
    print(f'[kinetics] priming Analysis (num_error_samples={n_err})...')
    t0 = _time.time()
    analysis = seekr2_analyze.Analysis(model, force_warning=False,
                                       num_error_samples=n_err)
    # First call reads all .out files and caches lines on each anchor_stats.
    analysis.extract_data(min_time=0.0, max_time=None)
    analysis.fill_out_data_samples()
    analysis.process_data_samples()
    print(f'[kinetics] cache primed in {_time.time()-t0:.1f} s; '
          f'subsequent block/window calls reuse it')

    # ── Rate-based ΔG_bind ──────────────────────────────────────────────────
    R   = 1.987204259e-3   # kcal/mol/K
    T   = float(model.temperature)
    RT  = R * T

    k_off = analysis.k_off
    k_off_err = analysis.k_off_error
    k_ons = dict(analysis.k_ons or {})
    k_ons_err = dict(analysis.k_ons_error or {})

    print('\n  ── Rates ──')
    print(f'  k_off     : {k_off:.3e} s^-1' +
          (f'  ± {k_off_err:.2e}' if k_off_err else ''))
    if not k_ons:
        print('  k_on      : (no BD data — k_on not computed)')
        dG_bind = None
    else:
        for state_name, k_on in k_ons.items():
            err = k_ons_err.get(state_name, None)
            print(f'  k_on[{state_name}] : {k_on:.3e} M^-1 s^-1' +
                  (f'  ± {err:.2e}' if err else ''))
        print('\n  ── ΔG_bind (rate-based; standard state c° = 1 M) ──')
        dG_bind = {}
        for state_name, k_on in k_ons.items():
            if k_on <= 0 or k_off <= 0:
                continue
            K_eq = k_on / k_off
            dG   = -RT * np.log(K_eq)
            dG_bind[state_name] = dG
            print(f'  ΔG_bind[{state_name}] = {dG:+.2f} kcal/mol  '
                  f'(K_eq = {K_eq:.2e} M^-1)')

    # ── Jacobian-corrected anchor PMF ───────────────────────────────────────
    radii = list(C.anchor_radii)
    n = len(radii)
    bounds = [0.0]
    for i in range(n - 1):
        bounds.append(0.5 * (radii[i] + radii[i + 1]))
    last_step = radii[-1] - radii[-2] if n >= 2 else radii[-1]
    bounds.append(radii[-1] + 0.5 * last_step)

    pi_alpha = np.asarray(analysis.pi_alpha).flatten()
    pi_alpha_err = (np.asarray(analysis.pi_alpha_error).flatten()
                    if analysis.pi_alpha_error is not None
                    else np.zeros_like(pi_alpha))
    fe_uncorr = np.asarray(analysis.free_energy_anchors).flatten()

    cell_vol = np.array([(4.0 / 3.0) * np.pi *
                         (bounds[i + 1] ** 3 - bounds[i] ** 3)
                         for i in range(n)])

    rho = pi_alpha[:n] / cell_vol
    rho_pos = np.where(rho > 0, rho, np.nan)
    fe_corr = -RT * np.log(rho_pos / np.nanmax(rho_pos))
    fe_corr = fe_corr - np.nanmin(fe_corr)

    # Per-anchor PMF error bars via error propagation from pi_alpha_error.
    # W_3D = -RT * log(rho/rho_max) = -RT * log(pi_alpha) + const(V, pi_max)
    # Treat the const as exact (V is geometric, pi_max is dominated by one
    # cell). σ_W ≈ RT * σ_pi / pi  (relative error in pi → absolute error in W).
    with np.errstate(divide='ignore', invalid='ignore'):
        fe_corr_err = np.where(pi_alpha[:n] > 0,
                               RT * pi_alpha_err[:n] / pi_alpha[:n],
                               np.nan)

    print('\n  ── PMF along COM-COM distance ──')
    print(f"  {'idx':>3}  {'r(nm)':>6}  {'ΔV(nm³)':>8}  "
          f"{'W_uncorr':>9}  {'W_3D':>9}  {'± σ':>6}  (kcal/mol)")
    for i, r in enumerate(radii):
        w_u = float(fe_uncorr[i]) if i < len(fe_uncorr) else float('nan')
        w_c = float(fe_corr[i])  if i < len(fe_corr)  else float('nan')
        e_c = float(fe_corr_err[i]) if i < len(fe_corr_err) else float('nan')
        wu_s = '  --     ' if not np.isfinite(w_u) else f'{w_u:>9.2f}'
        wc_s = '  --     ' if not np.isfinite(w_c) else f'{w_c:>9.2f}'
        ec_s = '  -- ' if not np.isfinite(e_c) else f'{e_c:>6.2f}'
        print(f'  {i:>3}  {r:>6.2f}  {cell_vol[i]:>8.2f}  {wu_s}  {wc_s}  {ec_s}')

    with open(out_csv, 'w') as f:
        f.write('anchor,radius_nm,cell_vol_nm3,W_uncorrected_kcalmol,'
                'W_3D_kcalmol,W_3D_error_kcalmol\n')
        for i, r in enumerate(radii):
            w_u = float(fe_uncorr[i]) if i < len(fe_uncorr) else ''
            w_c = float(fe_corr[i])  if i < len(fe_corr)  else ''
            e_c = (float(fe_corr_err[i])
                   if (i < len(fe_corr_err) and np.isfinite(fe_corr_err[i]))
                   else '')
            f.write(f'{i},{r},{cell_vol[i]:.4f},{w_u},{w_c},{e_c}\n')

    # ── ΔG_bind from PMF integration ────────────────────────────────────────
    # Two routes, mathematically equivalent up to definition of "bulk":
    #   (1) volumetric integral of exp(-(W_3D - W_bulk)/RT) over bound region
    #   (2) sum_bound(π_α) · V_bulk / π_bulk / V°
    # Both use the SAME π_α and cell volumes, so they cross-check the
    # arithmetic and the choice of bound/bulk regions, not the underlying
    # data.
    V0 = 1.66054   # V° = 1/(N_A · 1 M) in nm³
    n_bulk_ref = int(getattr(args, 'pmf_bulk_ref_anchors', 3) or 3)
    finite_mask = np.isfinite(fe_corr)
    finite_idx = np.where(finite_mask)[0]
    if len(finite_idx) >= n_bulk_ref + 2:
        # Last n_bulk_ref MD anchors as the "bulk" plateau reference.
        bulk_ref_idx = finite_idx[-n_bulk_ref:]
        W_bulk = float(np.mean(fe_corr[bulk_ref_idx]))

        # Bound region: contiguous from anchor 0 up to (not including) the
        # first anchor whose W ≥ W_bulk - threshold. With a non-monotonic
        # PMF (rising past a barrier then dropping back to plateau), this
        # gives the inner well, NOT outer plateau cells.
        threshold = 0.3   # kcal/mol — half-RT margin below bulk
        bound_idx = []
        for i in finite_idx:
            if i in bulk_ref_idx:
                break
            if fe_corr[i] < W_bulk - threshold:
                bound_idx.append(int(i))
            elif bound_idx:
                # Left the well — barrier region begins.
                break
        bound_idx = np.array(bound_idx, dtype=int)

        if len(bound_idx) > 0:
            # Method 1: volumetric integral with W_3D
            integrand = cell_vol[bound_idx] * np.exp(
                (W_bulk - fe_corr[bound_idx]) / RT)
            K_eq_pmf = float(integrand.sum() / V0)
            dG_pmf = -RT * float(np.log(K_eq_pmf))

            # Method 2: π-based, π_bulk = sum over bulk-ref anchors so the
            # comparison uses the SAME bulk reference as W_bulk above.
            pi_bound = float(pi_alpha[bound_idx].sum())
            pi_bulk  = float(pi_alpha[bulk_ref_idx].sum())
            V_bulk_total = float(cell_vol[bulk_ref_idx].sum())
            if pi_bulk > 0:
                K_eq_pi = pi_bound * V_bulk_total / pi_bulk / V0
                dG_pi = -RT * float(np.log(K_eq_pi))
            else:
                K_eq_pi, dG_pi = float('nan'), float('nan')

            print('\n  ── ΔG_bind from PMF (volumetric integration) ──')
            print(f"  bulk plateau   : anchors {list(bulk_ref_idx)} "
                  f"(W_bulk = {W_bulk:.2f} kcal/mol)")
            print(f"  bound region   : anchors {list(bound_idx)} "
                  f"(r ≤ {radii[bound_idx[-1]]:.2f} nm)")
            print(f"  Method 1 (W_3D · ΔV integral):     "
                  f"K_eq = {K_eq_pmf:.2f} M⁻¹  →  ΔG = {dG_pmf:+.2f} kcal/mol")
            if not np.isnan(dG_pi):
                print(f"  Method 2 (π_bound · V_bulk / π_bulk / V°): "
                      f"K_eq = {K_eq_pi:.2f} M⁻¹  →  ΔG = {dG_pi:+.2f} kcal/mol")
            else:
                print(f"  Method 2: skipped (π_bulk = 0; bulk-ref anchors unsampled)")
        else:
            print('\n  ── ΔG_bind from PMF: no anchor below W_bulk found (PMF not bound-shaped?)')
    else:
        print('\n  ── ΔG_bind from PMF: insufficient finite anchors for plateau detection')

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(radii, fe_uncorr, 'o-', color='tab:gray', alpha=0.6,
            label='seekr2 anchor FE (no Jacobian)')
    # PMF error bars from bootstrap-propagated pi_alpha_error.
    err_arr = np.array([float(fe_corr_err[i]) if i < len(fe_corr_err)
                        and np.isfinite(fe_corr_err[i]) else 0.0
                        for i in range(len(radii))])
    ax.errorbar(radii, fe_corr, yerr=err_arr, fmt='s-', color='tab:blue',
                capsize=3, label='Jacobian-corrected W(r)')
    ax.set_xlabel('COM-COM distance (nm)')
    ax.set_ylabel('Free energy (kcal/mol)')
    if dG_bind:
        states = ', '.join(f'{k}: {v:+.2f}' for k, v in dG_bind.items())
        ax.set_title(f'{C.name} — ΔG_bind (rate-based) {states} kcal/mol')
    else:
        ax.set_title(f'{C.name} — anchor PMF (Jacobian-corrected)')
    ax.axhline(0, color='black', lw=0.5, alpha=0.5)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(pmf_png, dpi=150)
    plt.close(fig)

    print(f'\n[kinetics] CSV  -> {out_csv}')
    print(f'[kinetics] plot -> {pmf_png}')

    # ── Block-average convergence test ──────────────────────────────────────
    # Re-run analysis on N equal time slices. Default N = 4 quarters: gives
    # both a fine-grained drift profile (Q1→Q2→Q3→Q4) and the late-time test
    # (Q4 vs Q3). A monotonic decrease across quarters with Q4≈Q3 means the
    # rate has stabilized; large jumps at any boundary mean still drifting.
    # Uses seekr2's native min_time/max_time filtering.
    n_blocks = int(getattr(args, 'n_blocks', 4) or 4)

    # Determine T_max from the cached existing_lines (no extra file I/O).
    # The cache was populated by analysis.extract_data() above.
    t_max_ps = 0.0
    for stats in analysis.anchor_stats_list:
        for line in (stats.existing_lines or []):
            if isinstance(line, str):
                continue   # NEW_SWARM marker
            try:
                t = float(line[2])
                if t > t_max_ps:
                    t_max_ps = t
            except Exception:
                continue
    if t_max_ps <= 0.0:
        t_max_ps = (model.calculation_settings.num_production_steps
                    * model.get_timestep())

    if not getattr(args, 'skip_block_average', False):
        block_w = t_max_ps / n_blocks
        print(f'\n  ── Block-average convergence (T_max = {t_max_ps:.0f} ps, '
              f'{n_blocks} blocks of {block_w:.0f} ps) ──')

        n_err_block = max(100, n_err // 5)   # smaller bootstrap = faster
        analysis.num_error_samples = n_err_block
        block_results = []
        t_block_start = _time.time()
        for b in range(n_blocks):
            label = f'block {b+1}/{n_blocks}'
            t_min = b * block_w
            t_max = (b + 1) * block_w
            k_b = _run_block(analysis, t_min, t_max)
            if k_b is None:
                block_results.append({'label': label, 'error': 'analyze failed'})
            else:
                block_results.append({
                    'label':     label,
                    't_min':     t_min,
                    't_max':     t_max,
                    'k_off':     k_b,
                    'k_off_err': analysis.k_off_error,
                })
        print(f'  ({n_blocks} blocks in {_time.time()-t_block_start:.1f} s '
              f'using cached Analysis)')

        print(f'  full run                : k_off = {k_off:.3e} s^-1'
              + (f'  ± {k_off_err:.2e}' if k_off_err else ''))
        for r in block_results:
            if 'error' in r:
                print(f'  {r["label"]:24s}: FAILED ({r["error"]})')
            else:
                err_s = (f'  ± {r["k_off_err"]:.2e}'
                         if r["k_off_err"] else '')
                t_lab = f'[{r["t_min"]:.0f}–{r["t_max"]:.0f} ps]'
                print(f'  {r["label"]:10s} {t_lab:18s}: '
                      f'k_off = {r["k_off"]:.3e} s^-1{err_s}')

        valid = [r for r in block_results if 'k_off' in r and r['k_off'] > 0]
        if len(valid) >= 2:
            # Late-time stability: drift between the LAST TWO blocks. This is
            # the most relevant test — if the rate is still changing at the
            # end, more sampling is needed regardless of how the early blocks
            # compare.
            k_late = valid[-1]['k_off']
            k_prev = valid[-2]['k_off']
            late_drift = abs(k_late - k_prev) / max(k_late, k_prev)
            print(f'\n  late drift (block {len(valid)} vs {len(valid)-1}): '
                  f'{100*late_drift:.0f}%')

            # Adjacent-block drift summary across the whole timeline
            adj_drifts = []
            for i in range(1, len(valid)):
                k1, k2 = valid[i-1]['k_off'], valid[i]['k_off']
                adj_drifts.append(abs(k2 - k1) / max(k1, k2))
            if adj_drifts:
                print(f'  max adjacent-block drift  : '
                      f'{100*max(adj_drifts):.0f}%')

            if late_drift > 0.5:
                print('  ⚠ NOT CONVERGED — late blocks differ >50%; extend simulation')
            elif late_drift > 0.2:
                print('  ⚠ borderline — late drift 20-50%; consider extending')
            else:
                print('  ✓ late-time blocks agree within statistical noise')

        # ── Per-anchor block-drift summary ────────────────────────────────
        # Global k_off can hide problems where a few specific anchors are
        # still drifting while the average looks stable. Compute the late-
        # vs-prev rate per anchor (transitions/time in the latest block vs
        # the previous block) and flag those above 50%. Identifies WHICH
        # anchors to extend, not just whether to extend overall.
        try:
            import re as _re
            from pathlib import Path as _Path
            print(f'\n  ── Per-anchor late-block drift '
                  f'(last vs second-to-last quarter) ──')
            print(f'  {"anchor":>6}  {"r(nm)":>6}  '
                  f'{"prev rate":>10}  {"late rate":>10}  drift')
            n_flagged = 0
            for alpha in range(len(C.anchor_radii) - 1):  # skip bulk
                prod = P.root / f'anchor_{alpha}' / 'prod'
                if not prod.exists():
                    continue
                # Aggregate (boundary, time) tuples across all swarm members
                cx = []
                for f in sorted(prod.glob('mmvt*restart*.out')):
                    try:
                        for line in f.read_text().splitlines():
                            line = line.strip()
                            if not line or line.startswith('#') \
                                    or line.startswith('CHECKPOINT'):
                                continue
                            parts = line.split(',')
                            if len(parts) == 3:
                                cx.append(float(parts[2]))
                    except Exception:
                        pass
                if len(cx) < 20:
                    continue
                cx.sort()
                t_max_a = cx[-1]
                # Last and previous blocks (each block = T_max/n_blocks)
                # using the same t_max_ps boundary computed above globally.
                prev_lo, prev_hi = (n_blocks - 2) * block_w, (n_blocks - 1) * block_w
                late_lo, late_hi = (n_blocks - 1) * block_w, n_blocks * block_w
                n_prev = sum(1 for t in cx if prev_lo <= t < prev_hi)
                n_late = sum(1 for t in cx if late_lo <= t <= late_hi)
                rate_prev = n_prev / max(prev_hi - prev_lo, 1e-9)
                rate_late = n_late / max(late_hi - late_lo, 1e-9)
                if max(rate_prev, rate_late) <= 0:
                    continue
                drift_a = abs(rate_late - rate_prev) / max(rate_prev, rate_late)
                flag = ' ⚠' if drift_a > 0.5 else ''
                if flag:
                    n_flagged += 1
                print(f'  {alpha:>6}  {C.anchor_radii[alpha]:>6.2f}  '
                      f'{rate_prev:>10.3f}  {rate_late:>10.3f}  '
                      f'{100*drift_a:>4.0f}%{flag}')
            if n_flagged > 0:
                print(f'  {n_flagged} anchor(s) flagged with >50% late-block drift; '
                      f'extending swarm benefits these specifically')
        except Exception as e:
            print(f'  (per-anchor block analysis skipped: {e!r})')

    # ── Sliding-window k_off ────────────────────────────────────────────────
    # Fixed-width window walks across the full timeline; in each window
    # k_off is re-evaluated against the cached Analysis. Output: k_off vs
    # window-center on a semilog plot. Flat tail = converged. The window is
    # wide (T_max/2.5 = 40% of total) so each window sees enough crossings
    # for a stable rate estimate; the block-average pass above complements
    # this with non-overlapping slices that average out window-level noise.
    if not getattr(args, 'skip_sliding_window', False):
        n_windows = int(getattr(args, 'n_windows', 30) or 30)
        window_ps = t_max_ps / 2.5
        step_ps = max((t_max_ps - window_ps) / max(n_windows - 1, 1), 1.0)
        print(f'\n  ── Sliding-window k_off ({n_windows} × {window_ps:.0f} ps, '
              f'step {step_ps:.0f} ps) ──')

        analysis.num_error_samples = 0    # bootstrap off — too expensive per window
        t_window_start = _time.time()
        k_off_list, t_mid_list = [], []
        for w in range(n_windows):
            t_min = w * step_ps
            t_max = t_min + window_ps
            if t_max > t_max_ps:
                break
            k_w = _run_block(analysis, t_min, t_max)
            if k_w is not None:
                k_off_list.append(k_w)
                t_mid_list.append(0.5 * (t_min + t_max) * 1e-3)   # ns
        print(f'  {len(k_off_list)} windows in '
              f'{_time.time()-t_window_start:.1f} s')

        if k_off_list:
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.semilogy(t_mid_list, k_off_list, 'o-', color='tab:blue')
            ax.set_xlabel('window center (ns)')
            ax.set_ylabel(r'$k_{off}$ (s$^{-1}$)')
            ax.set_title(f'{C.name} — sliding-window $k_{{off}}$ '
                         f'({n_windows} × {window_ps:.0f} ps)')
            ax.grid(alpha=0.3, which='both')
            fig.tight_layout()
            fig.savefig(win_png, dpi=150)
            plt.close(fig)
            with open(win_csv, 'w') as f:
                f.write('window_center_ns,k_off_per_s\n')
                for t, k in zip(t_mid_list, k_off_list):
                    f.write(f'{t:.3f},{k:.6e}\n')
            print(f'  plot -> {win_png}')
            print(f'  csv  -> {win_csv}')
            print(f'  • flat tail = converged')
            print(f'  • monotonic drift in the late half = still equilibrating')
        else:
            print('  no successful windows — sampling too short?')

    os.chdir(curdir)
