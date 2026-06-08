"""
Stage: ΔG_bind from k_on/k_off, Jacobian-corrected anchor PMF, block-average
+ sliding-window convergence on BOTH k_off and PMF/ΔG_bind.

Uses ONE seekr2.analyze.Analysis object across all sub-analyses
(headline, blocks, sliding window). The first extract_data() call reads
all .out files; subsequent calls hit the cached path
(analyze.py:348 `files_already_read=True`) and only re-run the per-line
min_time/max_time filter + small-matrix kinetics solve. Two orders of
magnitude faster than constructing a new Analysis for each block/window.

Block/window PMF + ΔG_bind reuse the bulk-plateau and bound-region anchor
indices identified on the FULL run, so the time-resolved ΔG measures how
the population in those fixed regions evolves — not how the regions
themselves move (which would conflate two different sources of variance).
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


def _block_pmf_and_dG(analysis, n_anchors, cell_vol, RT,
                      bulk_ref_idx, bound_idx, V0):
    """Recompute Jacobian-corrected PMF + Method-2 ΔG_bind from the
    pi_alpha currently held by `analysis` (after a re-filtered
    process_data_samples). Returns (fe_corr_array, dG_kcalmol).
    Both regions are passed in pre-computed from the full run so the
    block-to-block comparison is on a fixed bound/bulk definition."""
    pi_alpha = np.asarray(analysis.pi_alpha).flatten()[:n_anchors]
    rho = np.where(cell_vol > 0, pi_alpha / cell_vol, 0.0)
    rho_pos = np.where(rho > 0, rho, np.nan)
    if not np.isfinite(np.nanmax(rho_pos)):
        return np.full(n_anchors, np.nan), float('nan')
    fe_corr = -RT * np.log(rho_pos / np.nanmax(rho_pos))
    fe_corr = fe_corr - np.nanmin(fe_corr)

    if bulk_ref_idx is None or bound_idx is None \
            or len(bulk_ref_idx) == 0 or len(bound_idx) == 0:
        return fe_corr, float('nan')

    pi_bound = float(pi_alpha[bound_idx].sum())
    pi_bulk  = float(pi_alpha[bulk_ref_idx].sum())
    V_bulk_total = float(cell_vol[bulk_ref_idx].sum())
    if pi_bulk <= 0 or pi_bound <= 0:
        return fe_corr, float('nan')
    K_eq = pi_bound * V_bulk_total / pi_bulk / V0
    if not (K_eq > 0):
        return fe_corr, float('nan')
    dG = -RT * float(np.log(K_eq))
    return fe_corr, dG


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
    pmf_blocks_png = (P.work / 'kinetics_pmf_blocks.png').resolve()
    pmf_blocks_csv = (P.work / 'kinetics_pmf_blocks.csv').resolve()
    dG_blocks_csv  = (P.work / 'kinetics_dG_blocks.csv').resolve()
    dG_win_png     = (P.work / 'kinetics_dG_windows.png').resolve()
    dG_win_csv     = (P.work / 'kinetics_dG_windows.csv').resolve()
    per_campaign_csv = (P.work / 'kinetics_per_campaign.csv').resolve()

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
    # seekr2 sometimes returns k_on / k_on_error as a 0-D or 1-element
    # numpy array rather than a Python scalar (depends on the analysis path).
    # `float(...)` works on both; `:.3e` formatting does not work on ndarrays.
    def _scalar(x):
        return float(x) if x is not None else None
    if not k_ons:
        print('  k_on      : (no BD data — k_on not computed)')
        dG_bind = None
    else:
        for state_name, k_on in k_ons.items():
            k_on_v = _scalar(k_on)
            err    = _scalar(k_ons_err.get(state_name, None))
            print(f'  k_on[{state_name}] : {k_on_v:.3e} M^-1 s^-1' +
                  (f'  ± {err:.2e}' if err else ''))
        print('\n  ── ΔG_bind (rate-based; standard state c° = 1 M) ──')
        dG_bind = {}
        for state_name, k_on in k_ons.items():
            k_on_v = _scalar(k_on)
            if k_on_v <= 0 or k_off <= 0:
                continue
            K_eq = k_on_v / k_off
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
    # Per-user override: skip these anchor indices when picking the bulk
    # plateau. Useful when one anchor is a clear W_3D outlier driven by an
    # undersampled rate-matrix entry rather than real plateau density.
    bulk_exclude_str = str(getattr(args, 'bulk_exclude', '') or '').strip()
    bulk_exclude = set()
    if bulk_exclude_str:
        for tok in bulk_exclude_str.split(','):
            tok = tok.strip()
            if tok:
                try:
                    bulk_exclude.add(int(tok))
                except ValueError:
                    print(f'  [warn] --bulk-exclude: ignoring non-integer {tok!r}')
    bulk_ref_idx = None        # exposed to block/window loops below;
    bound_idx    = None        # both stay None if region detection fails
    finite_mask = np.isfinite(fe_corr)
    finite_idx = np.where(finite_mask)[0]
    # Apply --bulk-exclude only to plateau selection, not to bound detection.
    plateau_candidates = np.array(
        [i for i in finite_idx if int(i) not in bulk_exclude], dtype=int)
    if len(plateau_candidates) >= n_bulk_ref + 2:
        # Last n_bulk_ref non-excluded MD anchors as the "bulk" plateau reference.
        bulk_ref_idx = plateau_candidates[-n_bulk_ref:]
        if bulk_exclude:
            print(f'  [bulk] excluding anchors {sorted(bulk_exclude)} from plateau')
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
                fe_b, dG_b = _block_pmf_and_dG(
                    analysis, n, cell_vol, RT,
                    bulk_ref_idx, bound_idx, V0)
                block_results.append({
                    'label':     label,
                    't_min':     t_min,
                    't_max':     t_max,
                    'k_off':     k_b,
                    'k_off_err': analysis.k_off_error,
                    'fe_corr':   fe_b,
                    'dG_pmf':    dG_b,
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
                dG_s = (f'  ΔG = {r["dG_pmf"]:+.2f} kcal/mol'
                        if np.isfinite(r.get('dG_pmf', float('nan')))
                        else '  ΔG = --')
                print(f'  {r["label"]:10s} {t_lab:18s}: '
                      f'k_off = {r["k_off"]:.3e} s^-1{err_s}{dG_s}')

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

        # ── ΔG_bind block drift ───────────────────────────────────────────
        # Absolute drift in kcal/mol. RT≈0.6 at 300 K, so 0.5 kcal/mol is
        # ~kT — anything larger is real movement, not noise.
        valid_dG = [r for r in block_results
                    if np.isfinite(r.get('dG_pmf', float('nan')))]
        if len(valid_dG) >= 2:
            dG_late = valid_dG[-1]['dG_pmf']
            dG_prev = valid_dG[-2]['dG_pmf']
            dG_drift = abs(dG_late - dG_prev)
            adj_dG_drifts = [abs(valid_dG[i]['dG_pmf']
                                 - valid_dG[i-1]['dG_pmf'])
                             for i in range(1, len(valid_dG))]
            print(f'\n  ΔG late drift (block {len(valid_dG)} vs '
                  f'{len(valid_dG)-1}): {dG_drift:.2f} kcal/mol')
            if adj_dG_drifts:
                print(f'  max adjacent-block ΔG drift: '
                      f'{max(adj_dG_drifts):.2f} kcal/mol')
            if dG_drift > 1.0:
                print('  ⚠ NOT CONVERGED — late ΔG drifts >1 kcal/mol')
            elif dG_drift > 0.5:
                print('  ⚠ borderline — late ΔG drifts 0.5-1.0 kcal/mol')
            else:
                print('  ✓ late ΔG agrees within ~kT')

        # ── Block PMF overlay plot + CSV ──────────────────────────────────
        # One W(r) curve per block. Visual check on whether the well shape
        # and barrier height are stable across quarters; if blocks disagree
        # on the shape (not just the absolute values), the milestoning is
        # not equilibrated.
        valid_blocks = [r for r in block_results if 'fe_corr' in r]
        if valid_blocks:
            fig, ax = plt.subplots(figsize=(8, 5))
            cmap = plt.get_cmap('viridis')
            for i, r in enumerate(valid_blocks):
                color = cmap(i / max(len(valid_blocks) - 1, 1))
                lab = (f'{r["label"]} '
                       f'[{r["t_min"]:.0f}–{r["t_max"]:.0f} ps]')
                if np.isfinite(r.get('dG_pmf', float('nan'))):
                    lab += f'  ΔG={r["dG_pmf"]:+.2f}'
                ax.plot(radii, r['fe_corr'], 'o-', color=color, label=lab)
            ax.plot(radii, fe_corr, 's--', color='black', alpha=0.7,
                    label='full run')
            ax.set_xlabel('COM-COM distance (nm)')
            ax.set_ylabel('W(r) (kcal/mol)')
            ax.set_title(f'{C.name} — block PMF overlay '
                         f'({len(valid_blocks)} blocks)')
            ax.axhline(0, color='black', lw=0.5, alpha=0.3)
            ax.grid(alpha=0.3)
            ax.legend(fontsize=8, loc='best')
            fig.tight_layout()
            fig.savefig(pmf_blocks_png, dpi=150)
            plt.close(fig)

            with open(pmf_blocks_csv, 'w') as f:
                cols = ['anchor', 'radius_nm', 'W_full_kcalmol']
                cols += [f'W_block{i+1}_kcalmol' for i in range(len(valid_blocks))]
                f.write(','.join(cols) + '\n')
                for ai, r_a in enumerate(radii):
                    row = [str(ai), f'{r_a:.3f}',
                           f'{fe_corr[ai]:.4f}' if np.isfinite(fe_corr[ai]) else '']
                    for r in valid_blocks:
                        v = r['fe_corr'][ai] if ai < len(r['fe_corr']) else float('nan')
                        row.append(f'{v:.4f}' if np.isfinite(v) else '')
                    f.write(','.join(row) + '\n')

            with open(dG_blocks_csv, 'w') as f:
                f.write('block,t_min_ps,t_max_ps,k_off_per_s,dG_pmf_kcalmol\n')
                for r in block_results:
                    if 'error' in r:
                        continue
                    dG = r.get('dG_pmf', float('nan'))
                    dG_s = f'{dG:.4f}' if np.isfinite(dG) else ''
                    f.write(f'{r["label"]},{r["t_min"]:.1f},{r["t_max"]:.1f},'
                            f'{r["k_off"]:.6e},{dG_s}\n')
            print(f'\n  block PMF overlay -> {pmf_blocks_png}')
            print(f'  block PMF csv     -> {pmf_blocks_csv}')
            print(f'  block ΔG csv      -> {dG_blocks_csv}')

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
        window_fraction = float(getattr(args, 'window_fraction', 2.5) or 2.5)
        window_ps = t_max_ps / window_fraction
        step_ps = max((t_max_ps - window_ps) / max(n_windows - 1, 1), 1.0)
        print(f'\n  ── Sliding-window k_off ({n_windows} × {window_ps:.0f} ps, '
              f'step {step_ps:.0f} ps) ──')

        analysis.num_error_samples = 0    # bootstrap off — too expensive per window
        t_window_start = _time.time()
        k_off_list, t_mid_list, dG_list = [], [], []
        for w in range(n_windows):
            t_min = w * step_ps
            t_max = t_min + window_ps
            if t_max > t_max_ps:
                break
            k_w = _run_block(analysis, t_min, t_max)
            if k_w is not None:
                k_off_list.append(k_w)
                t_mid_list.append(0.5 * (t_min + t_max) * 1e-3)   # ns
                _, dG_w = _block_pmf_and_dG(
                    analysis, n, cell_vol, RT,
                    bulk_ref_idx, bound_idx, V0)
                dG_list.append(dG_w)
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

            # ── Sliding-window ΔG_bind ───────────────────────────────────
            # Same windows, ΔG_bind via Method 2 (population-direct).
            # Plotted on a linear y axis since ΔG sits in a narrow range
            # (a couple of kcal/mol); easier to read drift than k_off's
            # log-scale spikes.
            valid_dG = [(t, g) for t, g in zip(t_mid_list, dG_list)
                        if np.isfinite(g)]
            if valid_dG:
                t_dG, vals_dG = zip(*valid_dG)
                fig, ax = plt.subplots(figsize=(8, 5))
                ax.plot(t_dG, vals_dG, 'o-', color='tab:green')
                ax.set_xlabel('window center (ns)')
                ax.set_ylabel(r'$\Delta G_{bind}$ (kcal/mol)')
                ax.set_title(f'{C.name} — sliding-window $\\Delta G_{{bind}}$ '
                             f'({n_windows} × {window_ps:.0f} ps)')
                ax.axhline(0, color='black', lw=0.5, alpha=0.3)
                ax.grid(alpha=0.3)
                fig.tight_layout()
                fig.savefig(dG_win_png, dpi=150)
                plt.close(fig)
                with open(dG_win_csv, 'w') as f:
                    f.write('window_center_ns,dG_pmf_kcalmol\n')
                    for t, g in zip(t_mid_list, dG_list):
                        f.write(f'{t:.3f},{g:.4f}\n' if np.isfinite(g)
                                else f'{t:.3f},\n')
                print(f'  ΔG plot -> {dG_win_png}')
                print(f'  ΔG csv  -> {dG_win_csv}')
            else:
                print('  ΔG sliding-window: bound/bulk regions undefined; skipped')
        else:
            print('  no successful windows — sampling too short?')

    # ── Per-campaign (swarm_K) analysis ─────────────────────────────────────
    # Re-run the kinetics independently on each of the 4 HIDR campaigns
    # (swarm_0..3), then Boltzmann-combine the 4 ΔG values. This is a
    # diagnostic for guests where the 4 orientational sub-states seeded by
    # the campaigns do not interconvert during production: for them, the
    # pooled rate matrix above is equal-sample-weighted across non-equilibrium
    # sub-states, which biases ΔG. The proper observable in that limit is
    #   K_eq^obs = Σ_K exp(-ΔG_K / kT)              (parallel binding modes)
    #   k_off^obs = 1 / Σ_K p_K / k_off^K           (slowest sub-state dominates)
    # where p_K is the Boltzmann population. If the per-campaign ΔG values
    # agree within ~kT, the campaigns sampled the same ensemble and pooling
    # was fine; spread >> kT means pooling is biased and the combined value
    # should replace the pooled one.
    if getattr(args, 'per_campaign', False):
        _per_campaign(
            args, model, n, cell_vol, RT, V0,
            bulk_ref_idx, bound_idx, per_campaign_csv, n_err)

    os.chdir(curdir)


def _per_campaign(args, model, n_anchors, cell_vol, RT, V0,
                  bulk_ref_idx, bound_idx, per_campaign_csv, n_err):
    """Run SEEKR2 Analysis 4×, once per HIDR campaign (swarm_K).
    Restricts each anchor's md_output_glob to mmvt.swarm_K*.out so the
    re-read picks up only that campaign's trajectories. Uses the bound/bulk
    region definitions from the pooled run to keep ΔG comparable across K.
    """
    import seekr2.analyze as seekr2_analyze
    import numpy as np

    if bulk_ref_idx is None or bound_idx is None \
            or len(bulk_ref_idx) == 0 or len(bound_idx) == 0:
        print('\n  ── Per-campaign analysis: bound/bulk regions undefined; skipped ──')
        return

    print('\n  ── Per-campaign (swarm_K) analysis ──')

    # Save original globs so we can restore them.
    md_anchors = [a for a in model.anchors if a.md and not a.bulkstate]
    orig_globs = [a.md_output_glob for a in md_anchors]

    n_err_pc = max(100, n_err // 4)   # bootstrap on, but lighter than full run

    rows = []
    failed_K = []

    for K in range(4):
        for a in md_anchors:
            a.md_output_glob = f'mmvt.swarm_{K}*.out'
        try:
            ana_K = seekr2_analyze.Analysis(
                model, force_warning=False, num_error_samples=n_err_pc)
            ana_K.extract_data(min_time=0.0, max_time=None)
            ana_K.fill_out_data_samples()
            ana_K.process_data_samples()
        except Exception as e:
            print(f'  swarm_{K}: analysis failed ({e!r})')
            failed_K.append(K)
            rows.append({'K': K, 'k_off': float('nan'),
                         'k_off_err': float('nan'),
                         'dG_pmf': float('nan')})
            continue

        k_off_K = float(ana_K.k_off) if ana_K.k_off else float('nan')
        k_off_err_K = (float(ana_K.k_off_error)
                      if ana_K.k_off_error else float('nan'))

        # k_on is shared across all K (comes from BD b-surface, not MMVT), but
        # K_eq = k_on/k_off uses the per-campaign k_off → per-campaign ΔG_rate.
        k_ons_K = dict(ana_K.k_ons or {})
        if k_ons_K and k_off_K > 0 and np.isfinite(k_off_K):
            k_on_first = list(k_ons_K.values())[0]
            k_on_K = float(k_on_first) if k_on_first is not None else float('nan')
        else:
            k_on_K = float('nan')

        if np.isfinite(k_on_K) and k_on_K > 0:
            K_eq_rate_K = k_on_K / k_off_K
            dG_rate_K = -RT * float(np.log(K_eq_rate_K))
        else:
            dG_rate_K = float('nan')

        # Recompute ΔG_PMF on this campaign's pi_alpha, holding bound/bulk
        # regions fixed at the values derived from the full pooled run.
        pi_alpha_K = np.asarray(ana_K.pi_alpha).flatten()[:n_anchors]
        pi_bound_K = float(pi_alpha_K[bound_idx].sum())
        pi_bulk_K  = float(pi_alpha_K[bulk_ref_idx].sum())
        V_bulk_total = float(cell_vol[bulk_ref_idx].sum())
        if pi_bulk_K > 0 and pi_bound_K > 0:
            K_eq_K = pi_bound_K * V_bulk_total / pi_bulk_K / V0
            dG_K = -RT * float(np.log(K_eq_K)) if K_eq_K > 0 else float('nan')
        else:
            dG_K = float('nan')

        err_s = (f' ± {k_off_err_K:.2e}'
                 if np.isfinite(k_off_err_K) else '')
        dG_s = (f'{dG_K:+.2f}' if np.isfinite(dG_K) else '   nan')
        dG_rate_s = (f'{dG_rate_K:+.2f}'
                     if np.isfinite(dG_rate_K) else '   nan')
        print(f'  swarm_{K}: k_off = {k_off_K:.3e} s^-1{err_s}  '
              f'ΔG_PMF = {dG_s}  ΔG_rate = {dG_rate_s} kcal/mol')

        rows.append({'K': K, 'k_off': k_off_K,
                     'k_off_err': k_off_err_K, 'dG_pmf': dG_K,
                     'k_on': k_on_K, 'dG_rate': dG_rate_K})

    # Restore original globs (in case the caller reuses model).
    for a, g in zip(md_anchors, orig_globs):
        a.md_output_glob = g

    # ── Combine via parallel-channel pose grouping ───────────────────────────
    # The 4 HIDR campaigns = 2 binding poses × 2 exit routes per pose:
    #   Pose A = {K=0 (exit +side, primary face), K=2 (exit −side, secondary face)}
    #       — guest's chemical face contacts the same host face in both; the two
    #         campaigns differ in which cylinder opening the guest exits through.
    #   Pose B = {K=1 (exit −side, secondary face), K=3 (exit +side, primary face)}
    #       — guest's other chemical face contacts the host.
    # The cylindrical host geometry prevents the guest from switching exit routes
    # on the production timescale (a bulky guest cannot traverse the cavity waist
    # to flip which face it exits from). Each campaign therefore captures one
    # parallel, non-exchanging exit channel. Parallel channels from the same bound
    # state contribute additively to the total K_eq and k_off:
    #     K_pose_A = K_{K=0} + K_{K=2}  = Σ_{K∈A} exp(-ΔG_K / RT)
    #     K_pose_B = K_{K=1} + K_{K=3}  = Σ_{K∈B} exp(-ΔG_K / RT)
    #     K_total  = K_pose_A + K_pose_B = Σ_K exp(-ΔG_K / RT)  (pure multi-mode)
    #     ΔG_paired = -RT · ln(K_total)
    #     k_off_A  = k_off_0 + k_off_2  (parallel exit routes)
    #     k_off_B  = k_off_1 + k_off_3
    #     k_off_obs = p_A·k_off_A + p_B·k_off_B  (Boltzmann-population-weighted)
    # This is 0.41 kcal/mol more bound than the old (1/2) paired formula.
    POSE_A_KS = (0, 2)   # A, side
    POSE_B_KS = (1, 3)   # orient, both

    valid = [r for r in rows
             if np.isfinite(r['dG_pmf']) and r['k_off'] > 0
             and np.isfinite(r['k_off'])]
    n_valid = len(valid)

    if n_valid == 0:
        print('\n  no valid per-campaign results; combined ΔG not computed')
        with open(per_campaign_csv, 'w') as f:
            f.write('campaign,k_off_per_s,k_off_err,dG_pmf_kcalmol,share\n')
            for r in rows:
                f.write(f'swarm_{r["K"]},{r["k_off"]},{r["k_off_err"]},'
                        f'{r["dG_pmf"]},\n')
        print(f'  csv -> {per_campaign_csv}')
        return

    K_to_row = {r['K']: r for r in rows}

    def _pose_keq(pose_Ks, dG_field):
        """Pose K_eq: (1/m) Σ exp(-ΔG_K/RT) over valid K in pose_Ks.
        ΔG is the same for all parallel exit channels of the same pose (thermodynamic
        identity), so campaigns within a pose are Boltzmann-averaged (1/m normalization).
        k_off combining uses _pose_koff_sum (rates add for parallel channels).
        Returns (dG_pose, K_eq_pose, n_members)."""
        members = [K_to_row[k][dG_field] for k in pose_Ks
                   if k in K_to_row
                   and np.isfinite(K_to_row[k].get(dG_field, float('nan')))]
        if not members:
            return float('nan'), 0.0, 0
        arr = np.array(members)
        x_ = -arr / RT
        x_max = x_.max()
        K_eq_pose = float(np.exp(x_max) * np.sum(np.exp(x_ - x_max))) / len(members)  # (1/m) Σ exp(-ΔG/RT)
        dG_pose = -RT * float(np.log(K_eq_pose)) if K_eq_pose > 0 else float('nan')
        return dG_pose, K_eq_pose, len(members)

    def _paired_total(pose_Ks_A, pose_Ks_B, dG_field):
        """ΔG_paired = -RT ln(K_pose_A + K_pose_B). Returns (dG, dG_A, dG_B, K_A, K_B)."""
        dG_A, K_A, n_A = _pose_keq(pose_Ks_A, dG_field)
        dG_B, K_B, n_B = _pose_keq(pose_Ks_B, dG_field)
        K_total = K_A + K_B
        if K_total <= 0:
            return float('nan'), dG_A, dG_B, K_A, K_B
        return -RT * float(np.log(K_total)), dG_A, dG_B, K_A, K_B

    # ΔG_PMF: pose K_eq sums + paired total
    dG_pmf_paired, dG_pmf_pose_A, dG_pmf_pose_B, K_pmf_A, K_pmf_B = _paired_total(
        POSE_A_KS, POSE_B_KS, 'dG_pmf')

    # ΔG_rate: same combination using per-campaign k_on/k_off_K ratios
    dG_rate_paired, dG_rate_pose_A, dG_rate_pose_B, _, _ = _paired_total(
        POSE_A_KS, POSE_B_KS, 'dG_rate')

    # k_off: sum within each pose (parallel exit channels that don't exchange
    # on the production timescale), then Boltzmann-population-weight across poses.
    def _pose_koff_sum(pose_Ks):
        vals = [K_to_row[k]['k_off'] for k in pose_Ks
                if k in K_to_row and K_to_row[k].get('k_off', 0) > 0
                and np.isfinite(K_to_row[k].get('k_off', float('nan')))]
        return float(sum(vals)) if vals else float('nan')

    koff_sum_A = _pose_koff_sum(POSE_A_KS)
    koff_sum_B = _pose_koff_sum(POSE_B_KS)
    K_pmf_total = K_pmf_A + K_pmf_B
    if K_pmf_total > 0 and np.isfinite(koff_sum_A) and np.isfinite(koff_sum_B):
        p_A = K_pmf_A / K_pmf_total
        p_B = K_pmf_B / K_pmf_total
        k_off_paired = p_A * koff_sum_A + p_B * koff_sum_B
    else:
        k_off_paired = float('nan')

    valid_koff_K = [r['k_off'] for r in valid if r['k_off'] > 0]
    k_off_arith = float(np.mean(valid_koff_K)) if valid_koff_K else float('nan')

    # Within-pose spread diagnostics (max |ΔG_a - ΔG_b| inside each pose).
    def _within_pose_spread(pose_Ks):
        vals = [K_to_row[k]['dG_pmf'] for k in pose_Ks
                if k in K_to_row and np.isfinite(K_to_row[k]['dG_pmf'])]
        return max(vals) - min(vals) if len(vals) >= 2 else float('nan')

    spread_A = _within_pose_spread(POSE_A_KS)
    spread_B = _within_pose_spread(POSE_B_KS)
    worst_within_pose = max(s for s in (spread_A, spread_B)
                            if np.isfinite(s)) if (np.isfinite(spread_A)
                                                    or np.isfinite(spread_B)) else float('nan')
    pose_gap = (abs(dG_pmf_pose_A - dG_pmf_pose_B)
                if np.isfinite(dG_pmf_pose_A) and np.isfinite(dG_pmf_pose_B)
                else float('nan'))

    # Total spread (legacy diagnostic — max-min across all K).
    dG_arr = np.array([r['dG_pmf'] for r in valid])
    dG_spread = float(dG_arr.max() - dG_arr.min())

    koff_A_s = (f'{koff_sum_A:.3e}' if np.isfinite(koff_sum_A) else 'nan')
    koff_B_s = (f'{koff_sum_B:.3e}' if np.isfinite(koff_sum_B) else 'nan')
    print(f'\n  ── Parallel-channel pose grouping ──')
    print(f'  Pose A = {{K=0 (+side exit), K=2 (−side exit)}}  '
          f'ΔG_PMF = {dG_pmf_pose_A:+.2f}  '
          f'ΔG_rate = {dG_rate_pose_A:+.2f}  '
          f'k_off_A = {koff_A_s}  spread = {spread_A:.2f} kcal/mol')
    print(f'  Pose B = {{K=1 (−side exit), K=3 (+side exit)}}  '
          f'ΔG_PMF = {dG_pmf_pose_B:+.2f}  '
          f'ΔG_rate = {dG_rate_pose_B:+.2f}  '
          f'k_off_B = {koff_B_s}  spread = {spread_B:.2f} kcal/mol')
    print(f'\n  ΔG_PMF_paired  = {dG_pmf_paired:+.2f} kcal/mol  '
          f'(−RT ln[K_pose_A + K_pose_B], parallel-channel sum)')
    if np.isfinite(dG_rate_paired):
        print(f'  ΔG_rate_paired = {dG_rate_paired:+.2f} kcal/mol  '
              f'(same formula on ΔG_rate_K)')
    print(f'  k_off_paired   = {k_off_paired:.3e} s^-1  '
          f'(p_A·k_off_A + p_B·k_off_B, Boltzmann population-weighted)')
    print(f'  k_off_arith    = {k_off_arith:.3e} s^-1  '
          f'(arithmetic mean across 4 — reference only)')
    print(f'  total spread across 4 campaigns: {dG_spread:.2f} kcal/mol  '
          f'(kT = {RT:.2f})')

    if np.isfinite(worst_within_pose) and np.isfinite(pose_gap):
        if worst_within_pose < pose_gap:
            print('  ✓ pose-pose gap > within-pose spread — pose decomposition valid')
        else:
            print('  ⚠ within-pose spread > pose-pose gap — exit-route ΔG spread '
                  'exceeds pose-pose gap; pose assignment ambiguous for this guest')

    K_pose_label = {0: 'A', 1: 'B', 2: 'A', 3: 'B'}

    def _fmt_f(v, spec='.6e'):
        return f'{v:{spec}}' if np.isfinite(v) else ''

    with open(per_campaign_csv, 'w') as f:
        f.write('campaign,pose,k_off_per_s,k_off_err,k_on_per_M_per_s,'
                'dG_pmf_kcalmol,dG_rate_kcalmol\n')
        for r in rows:
            f.write(
                f'swarm_{r["K"]},{K_pose_label.get(r["K"], "")},'
                f'{_fmt_f(r["k_off"])},'
                f'{_fmt_f(r["k_off_err"])},'
                f'{_fmt_f(r.get("k_on", float("nan")))},'
                f'{_fmt_f(r["dG_pmf"], ".4f")},'
                f'{_fmt_f(r.get("dG_rate", float("nan")), ".4f")}\n')
        f.write(f'pose_A,A,{_fmt_f(koff_sum_A)},,,{_fmt_f(dG_pmf_pose_A, ".4f")},'
                f'{_fmt_f(dG_rate_pose_A, ".4f")}\n')
        f.write(f'pose_B,B,{_fmt_f(koff_sum_B)},,,{_fmt_f(dG_pmf_pose_B, ".4f")},'
                f'{_fmt_f(dG_rate_pose_B, ".4f")}\n')
        # Paired (multi-mode sum across poses).
        f.write(f'paired,total,{_fmt_f(k_off_paired)},,,'
                f'{_fmt_f(dG_pmf_paired, ".4f")},'
                f'{_fmt_f(dG_rate_paired, ".4f")}\n')
    print(f'  csv -> {per_campaign_csv}')
