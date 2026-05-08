"""
Stage: 4-member swarm MMVT, all from real HIDR campaigns.

Four independent HIDR campaigns provide the 4 starting bound states
(2 orientations × 2 sides), each with its own anchor tree:

    root/         — campaign A (no transform; from `--stage run`)
    root_orient/  — campaign B (orientation flipped at bound state)
    root_side/    — campaign C (side flipped, orientation preserved)
    root_both/    — campaign D (both flipped at bound state)

Per anchor:
    swarm 0: HIDR-A structure
    swarm 1: HIDR-B structure
    swarm 2: HIDR-C structure
    swarm 3: HIDR-D structure

No synthetic transforms in this stage — every starting state is a real
HIDR-relaxed pose. We still run a short minimize + brief NVT with
restraints to harmonize box vectors / velocities into a saveable XML
state, but no positions are perturbed beyond what HIDR produced.

Output files in each anchor's prod/:
    mmvt.restart1.swarm_K.out
    mmvt1.swarm_K.dcd
    backup.checkpoint.swarm_K
seekr2.analyze aggregates them automatically.

Backward-compat: if root_both/ doesn't exist but root_down/ does (from
the deprecated --stage hidr_down), we use root_down/ as root_both/
without copying.
"""

import json
import multiprocessing
import os
import time
from pathlib import Path

import numpy as np

from ._config import C, paths


def _write_multiframe_pdb(src_pdb, n_frames, dst_pdb):
    """
    Write a multi-frame PDB by replicating the source PDB n_frames times.
    Used to satisfy seekr2's swarm-frame assertion: when --swarm_index=K,
    seekr2 reads frame K from anchor.amber_params.pdb_coordinates_filename
    (even though we then overwrite positions via load_state_file).
    """
    with open(src_pdb) as f:
        lines = f.readlines()
    cryst_lines = [l for l in lines if l.startswith('CRYST1')]
    atom_lines  = [l for l in lines
                   if l.startswith(('ATOM', 'HETATM', 'TER'))]
    with open(dst_pdb, 'w') as f:
        for cl in cryst_lines:
            f.write(cl)
        for k in range(n_frames):
            f.write(f'MODEL     {k+1:>4d}\n')
            for al in atom_lines:
                f.write(al)
            f.write('ENDMDL\n')
        f.write('END\n')


# ── Coordinate transforms ─────────────────────────────────────────────────────

def _orientation_flip(coords, guest_idx, guest_axis_idx, host_axis):
    """
    Rotate guest atoms 180° about an axis perpendicular to (C3→C17),
    through the guest COM. This swaps which end of cholesterol points
    toward the BCD primary face.

    The rotation axis is host_axis × guest_axis (perpendicular to both,
    falls back to a different vector if those are parallel).
    """
    out = coords.copy()
    g_axis = coords[guest_axis_idx[1]] - coords[guest_axis_idx[0]]
    g_axis /= np.linalg.norm(g_axis)

    perp = np.cross(host_axis, g_axis)
    if np.linalg.norm(perp) < 1e-6:
        # parallel — pick any vector not parallel to g_axis
        seed = np.array([1.0, 0.0, 0.0])
        if abs(seed @ g_axis) > 0.9:
            seed = np.array([0.0, 1.0, 0.0])
        perp = np.cross(g_axis, seed)
    perp /= np.linalg.norm(perp)

    # 180° rotation about unit vector n: R = 2 n n^T − I
    R = 2 * np.outer(perp, perp) - np.eye(3)

    g_com = out[guest_idx].mean(0)
    out[guest_idx] = (out[guest_idx] - g_com) @ R.T + g_com
    return out


# Orientation-flip helper kept for backward use only; current swarm
# stage doesn't apply any synthetic transforms (all 4 swarms come from
# real HIDR campaigns via --stage hidr_alts).


# ── State-file generation ────────────────────────────────────────────────────

def _relax_and_save_state(prmtop_file, coords_ang, struct_box, struct_atoms,
                          host_resname, guest_resname,
                          gpu_index, restraint_k, minimize_iter, relax_ps,
                          out_xml,
                          r_inner_nm=None, r_outer_nm=None,
                          rec_indices=None, lig_indices=None,
                          r_target_nm=None,
                          k_rad_kcal_per_mol_A2=500.0,
                          temperature_K=300.0):
    """
    Build a CUDA context with positional restraints on solute heavy atoms,
    set the transformed positions + box, minimize, run a brief NVT to let
    waters/ions repack around the new ligand position, then save the
    OpenMM XML state.

    If r_inner_nm/r_outer_nm and rec_indices/lig_indices are given, verify the
    post-relax |r| (using the seekr2 CV's atom indices) is strictly inside
    [r_inner_nm, r_outer_nm]; raise RuntimeError("OUTSIDE_CELL: ...") if not so
    the caller's retry/drop logic can handle it. Without this check, drift
    during the relax can land the saved state past a milestone, causing seekr2
    to bounce on step 0 with "trapped behind a boundary".
    """
    from openmm.app import AmberPrmtopFile, Simulation, PME, HBonds
    from openmm import (LangevinMiddleIntegrator, Platform,
                        CustomExternalForce, CustomCentroidBondForce)
    from openmm.app.internal.unitcell import computePeriodicBoxVectors
    import openmm.unit as unit

    prmtop = AmberPrmtopFile(prmtop_file)
    system = prmtop.createSystem(
        nonbondedMethod=PME, nonbondedCutoff=0.9 * unit.nanometer,
        constraints=HBonds, rigidWater=True)

    # Heavy-atom positional restraints on HOST atoms only — keeps host axis
    # stable during minimize. Guest is positioned via the CV-level radial
    # restraint below (so cholesterol's COM lands at r_target = anchor center
    # regardless of small offsets in the HIDR-deposited pose).
    restraint = CustomExternalForce("k_r*((x-x0)^2 + (y-y0)^2 + (z-z0)^2)")
    restraint.addGlobalParameter(
        "k_r", restraint_k * unit.kilocalories_per_mole / unit.angstrom**2)
    restraint.addPerParticleParameter("x0")
    restraint.addPerParticleParameter("y0")
    restraint.addPerParticleParameter("z0")

    n_restrained = 0
    for i, atom in enumerate(struct_atoms):
        if atom.residue.name == host_resname and atom.element != 1:
            x_nm = coords_ang[i, 0] * 0.1
            y_nm = coords_ang[i, 1] * 0.1
            z_nm = coords_ang[i, 2] * 0.1
            restraint.addParticle(i, [x_nm, y_nm, z_nm])
            n_restrained += 1
    system.addForce(restraint)

    # CV-level radial restraint pinning guest COM at r_target from host COM.
    # Same form as HIDR's k_rad: 0.5 * k_rad * (|COM-COM| - r_target)^2.
    # Without this, minimize over 2500 iters drifts cholesterol's COM ~90 mÅ
    # from the HIDR-deposited position — enough to land at the cell edge in
    # narrow cells (e.g., anchor 3 cell width is 100 mÅ).
    if (r_target_nm is not None and rec_indices is not None
            and lig_indices is not None):
        radial = CustomCentroidBondForce(
            2, "0.5*k_rad*(distance(g1,g2) - r_target)^2")
        radial.addGlobalParameter(
            "k_rad",
            k_rad_kcal_per_mol_A2 * unit.kilocalories_per_mole / unit.angstrom**2)
        radial.addGlobalParameter("r_target", r_target_nm * unit.nanometer)
        radial.addGroup(list(rec_indices))
        radial.addGroup(list(lig_indices))
        radial.addBond([0, 1], [])
        system.addForce(radial)

    # Detect HMR from H masses in the prmtop (avoids relying on C in this
    # subprocess where spawn re-imports the module without user config).
    # Mean H mass > 2 amu → HMR is active → 4 fs is safe; else 2 fs.
    # Phase 1 doesn't actually run MD (only minimize + setVelToTemperature),
    # but the integrator's timestep should be consistent with the prmtop
    # masses to avoid OpenMM warnings.
    h_masses = [a.mass for a in struct_atoms if a.atomic_number == 1]
    timestep_ps = 0.004 if (h_masses and sum(h_masses) / len(h_masses) > 2.0) else 0.002
    integrator = LangevinMiddleIntegrator(
        float(temperature_K) * unit.kelvin, 1.0 / unit.picosecond, timestep_ps * unit.picoseconds)
    platform = Platform.getPlatformByName('CUDA')
    simulation = Simulation(prmtop.topology, system, integrator, platform,
                            {'CudaDeviceIndex': str(gpu_index),
                             'CudaPrecision': 'mixed'})

    pos_q = unit.Quantity(coords_ang.tolist(), unit.angstrom)
    simulation.context.setPositions(pos_q)

    if struct_box is not None:
        a, b, c, alpha, beta, gamma = struct_box
        v1, v2, v3 = computePeriodicBoxVectors(
            a * unit.angstrom, b * unit.angstrom, c * unit.angstrom,
            alpha * unit.degree, beta * unit.degree, gamma * unit.degree)
        simulation.context.setPeriodicBoxVectors(v1, v2, v3)

    # Brief minimize repairs any small clashes from the HIDR-deposited
    # pose. No MD step — we want the saved state's COM to match the
    # PDB position (which we already verified is in-cell). MD relax would
    # let cholesterol drift toward a local minimum within the cell, and
    # at narrow cell widths the drift can land it past a milestone.
    simulation.minimizeEnergy(maxIterations=minimize_iter)
    simulation.context.setVelocitiesToTemperature(float(temperature_K) * unit.kelvin)

    # Save state WITHOUT global parameters. simulation.saveState() includes
    # all parameters by default — including our private `k_r` from the
    # CustomExternalForce restraint. seekr2's MMVT system doesn't have
    # `k_r`, so loadState() raises "invalid parameter name: k_r" when it
    # tries to apply the saved state. We just need positions, velocities,
    # and box vectors — strip everything else.
    import openmm
    state = simulation.context.getState(
        getPositions=True, getVelocities=True, enforcePeriodicBox=True)

    # Post-minimize cell-containment check using seekr2's exact CV
    # evaluator (mass-weighted COMs). If the minimize-relaxed state lands
    # outside the cell, raise OUTSIDE_CELL so the caller can retry with a
    # stronger radial restraint or drop the state — without this, seekr2
    # will bounce on step 0 with "trapped behind a boundary".
    if (r_inner_nm is not None and r_outer_nm is not None
            and rec_indices is not None and lig_indices is not None):
        import seekr2.modules.common_base as _seekr_base
        positions = state.getPositions()
        h_com = _seekr_base.get_openmm_center_of_mass_com(
            system, positions, list(rec_indices))
        g_com = _seekr_base.get_openmm_center_of_mass_com(
            system, positions, list(lig_indices))
        d_post = float(np.linalg.norm(g_com - h_com))
        if not (r_inner_nm <= d_post <= r_outer_nm):
            raise RuntimeError(
                f'OUTSIDE_CELL: post-minimize r={d_post:.4f} nm outside '
                f'cell [{r_inner_nm:.4f}, {r_outer_nm:.4f}]')

    with open(str(out_xml), 'w') as f:
        f.write(openmm.XmlSerializer.serialize(state))


# ── Per (anchor, swarm-member) worker (subprocess) ───────────────────────────

# Defaults for the bounce-rate pathology watchdog. Normal MMVT in this
# system runs ~1-5 bounces/ps; the pathology mode (trajectory parked just
# outside a milestone, integrator ringing every step) hits ~200+ bounces/ps.
# 50/ps splits the gap. Grace window of 100 ps avoids flagging the
# transient at the very start of a run.
BOUNCE_RATE_THRESHOLD_PER_PS = 50.0
BOUNCE_RATE_GRACE_SIM_PS     = 100.0
BOUNCE_RATE_POLL_SEC         = 30.0


def _bounce_rate_watchdog(prod_dir, swarm_idx,
                          threshold_per_ps=BOUNCE_RATE_THRESHOLD_PER_PS,
                          grace_sim_ps=BOUNCE_RATE_GRACE_SIM_PS,
                          poll_sec=BOUNCE_RATE_POLL_SEC):
    """Daemon thread that watches the most recent
    `mmvt.swarm_K.restart*.out` for this (anchor, swarm_K) and aborts the
    subprocess if the cumulative bounce-rate exceeds threshold after the
    grace window. Symptom of a velocity-flip MMVT trajectory parked just
    outside a milestone — recording millions of integrator-induced
    crossings while the configuration is effectively frozen. See
    diagnostic notes in the BCD/cholesterol project log.

    On detection: removes `backup.checkpoint.swarm_K` to prevent a future
    `swarmflow swarm` invocation from resuming the same bad state, writes
    a `PATHOLOGY.swarm_K` sentinel describing the failure, and calls
    os._exit(2). The corrupt .out/.dcd files are left in place so the user
    can inspect them; they should be moved or deleted before re-running.
    """
    import os, glob, re, time
    while True:
        time.sleep(poll_sec)
        try:
            files = glob.glob(
                os.path.join(prod_dir, f'mmvt.swarm_{swarm_idx}.restart*.out'))
            if not files:
                continue
            files.sort(key=lambda f: int(re.search(r'restart(\d+)', f).group(1)))
            latest = files[-1]
            cnt = 0
            last_t = 0.0
            with open(latest) as fh:
                for ln in fh:
                    ln = ln.strip()
                    if not ln or ln.startswith('#') \
                            or ln.startswith('CHECKPOINT'):
                        continue
                    parts = ln.split(',')
                    if len(parts) != 3:
                        continue
                    cnt += 1
                    try:
                        t = float(parts[2])
                        if t > last_t:
                            last_t = t
                    except Exception:
                        pass
            if last_t < grace_sim_ps:
                continue
            rate = cnt / max(last_t, 1e-9)
            if rate <= threshold_per_ps:
                continue
            # Pathology detected. Remove the corrupt checkpoint so a future
            # resume doesn't re-trigger the same state, write a sentinel,
            # and exit hard. daemon=True watchdog can't be cleanly joined,
            # so os._exit terminates the whole subprocess.
            msg = (f'[watchdog] PATHOLOGY DETECTED on swarm_{swarm_idx}: '
                   f'{cnt:,} bounces in {last_t:.1f} ps = {rate:.1f}/ps '
                   f'(threshold {threshold_per_ps:.0f}/ps after '
                   f'{grace_sim_ps:.0f} ps grace). Aborting subprocess; '
                   f'removing backup.checkpoint.swarm_{swarm_idx}.')
            print(msg, flush=True)
            ckpt = os.path.join(prod_dir, f'backup.checkpoint.swarm_{swarm_idx}')
            if os.path.exists(ckpt):
                try:
                    os.remove(ckpt)
                except OSError:
                    pass
            sentinel = os.path.join(prod_dir, f'PATHOLOGY.swarm_{swarm_idx}')
            try:
                with open(sentinel, 'w') as f:
                    f.write(msg + '\n')
                    f.write(f'latest_out_file: {latest}\n')
            except OSError:
                pass
            os._exit(2)
        except Exception:
            # Don't let watchdog errors take down the subprocess. Log and
            # keep polling — false-negative is preferable to crashing the
            # production run.
            import traceback
            traceback.print_exc()
            continue


def _run_swarm_member(model_xml, anchor_idx, swarm_idx, state_file, gpu_index,
                      log_path=None, total_steps=None):
    """
    Subprocess: run seekr2.run.run_openmm for one (anchor, swarm_K) pair.
    Outputs are written with the .swarm_K suffix because seekr2 uses
    swarm_index in the output basename. stdout/stderr → log_path so the
    parent's terminal isn't flooded by OpenMM StateDataReporter output.

    A daemon `_bounce_rate_watchdog` thread runs alongside and aborts the
    subprocess if a velocity-flip-MMVT pathology develops (cumulative
    bounce rate > 50/ps after 100 ps grace).

    total_steps overrides model.calculation_settings.num_production_steps so
    that bumping production_steps_per_anchor in config.yml extends an
    existing run without needing a re-setup. Also patched into the model
    object itself because seekr2.run reads from there in some code paths.
    """
    import os, sys, traceback
    if log_path is not None:
        sys.stdout = open(log_path, 'w', buffering=1)
        sys.stderr = sys.stdout
    # Respect an inherited CUDA_VISIBLE_DEVICES (e.g. SLURM's --gres=gpu:1
    # allocation per array task). Overwriting it here would silently bypass
    # the scheduler's GPU isolation — every concurrent task's workers would
    # land on physical GPU 0 regardless of what SLURM allocated. Only set
    # the var when the parent left it unset (standalone / single-host runs).
    if 'CUDA_VISIBLE_DEVICES' not in os.environ:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_index)
    import seekr2.modules.common_base as common_base
    import seekr2.run as seekr2_run

    model = common_base.load_model(str(model_xml))
    anchor = model.anchors[anchor_idx]

    # Resume from existing checkpoint if present, else fresh from state file.
    # When starting fresh, wipe any stale per-swarm output files that may
    # have been written by an earlier crashed run (seekr2 writes a header
    # row to mmvt.swarm_K.restart1.out during prepare() before any MD,
    # and refuses to overwrite next time without force_overwrite).
    ckpt = (Path(model.anchor_rootdir) / anchor.directory
            / anchor.production_directory
            / f'backup.checkpoint.swarm_{swarm_idx}')
    restart = ckpt.exists()
    if total_steps is None:
        total_steps = model.calculation_settings.num_production_steps
    else:
        # Patch the in-memory model so any seekr2 internals that read
        # num_production_steps directly see the bumped value.
        model.calculation_settings.num_production_steps = int(total_steps)

    print(f'[swarm] anchor {anchor_idx} swarm {swarm_idx}: '
          f'{"resume" if restart else "start"} on GPU {gpu_index} '
          f'({total_steps:,} steps)', flush=True)

    # Pre-flight: if a sentinel from a previous aborted run is present,
    # refuse to run this (anchor, K) until the user clears it. Prevents
    # silently restarting a known-bad state.
    prod_dir = (Path(model.anchor_rootdir) / anchor.directory
                / anchor.production_directory)
    sentinel = prod_dir / f'PATHOLOGY.swarm_{swarm_idx}'
    if sentinel.exists():
        print(f'[swarm] anchor {anchor_idx} swarm {swarm_idx}: pathology '
              f'sentinel present at {sentinel}; refusing to run. '
              f'Inspect and remove (along with bad .out/.dcd) before retry.',
              flush=True)
        sys.exit(2)

    # Pre-flight: guard against the cleanse_anchor_outputs cascade.
    # When restart=False (no checkpoint for THIS K), seekr2 calls
    # `cleanse_anchor_outputs` which deletes EVERYTHING in prod/ —
    # not just files for swarm_idx, but every other K's checkpoints,
    # .out files, and .dcd files too (seekr2/modules/runner_openmm.py:157).
    # Refuse to launch if any other K in the same prod_dir has data we
    # would otherwise lose.
    if not restart:
        import re as _re
        other_k_with_data = set()
        for cf in prod_dir.glob('backup.checkpoint.swarm_*'):
            try:
                other_k = int(cf.name.rsplit('_', 1)[1])
            except (ValueError, IndexError):
                continue
            if other_k != swarm_idx:
                other_k_with_data.add(other_k)
        # DCDs and .out files for other K's count too — even without a
        # checkpoint, those are real simulation data the user may want.
        for f in list(prod_dir.glob('mmvt*.dcd')) \
                + list(prod_dir.glob('mmvt.swarm_*.restart*.out')):
            try:
                if f.stat().st_size <= 100:
                    continue
            except OSError:
                continue
            m = _re.search(r'swarm_(\d+)', f.name)
            if not m:
                continue
            other_k = int(m.group(1))
            if other_k != swarm_idx:
                other_k_with_data.add(other_k)
        if other_k_with_data:
            print(
                f'[swarm] anchor {anchor_idx} swarm {swarm_idx}: REFUSING to '
                f'launch with force_overwrite=True (no checkpoint for K='
                f'{swarm_idx}, but K={sorted(other_k_with_data)} have data '
                f'in {prod_dir}). seekr2.cleanse_anchor_outputs is anchor-'
                f'scoped — it would wipe ALL K data, not just K={swarm_idx}. '
                f'To proceed, either:\n'
                f'  (a) restore backup.checkpoint.swarm_{swarm_idx} from '
                f'a previous run,\n'
                f'  (b) re-run Phase 1 to regenerate state.swarm_{swarm_idx}'
                f'.xml (deletes only that K`s state file, then a fresh '
                f'run will checkpoint cleanly), or\n'
                f'  (c) move/delete the surviving K`s data in {prod_dir} '
                f'if you intentionally want a clean anchor reset.',
                flush=True,
            )
            sys.exit(2)

    # Spawn the bounce-rate watchdog. Daemon thread = dies with the
    # subprocess; uses os._exit on detection so we don't have to wire a
    # stop signal back into seekr2's tight inner loop.
    import threading
    threading.Thread(
        target=_bounce_rate_watchdog,
        args=(str(prod_dir), swarm_idx),
        daemon=True,
    ).start()

    try:
        seekr2_run.run_openmm(
            model, anchor_idx, restart, total_steps,
            cuda_device_index='0',   # CUDA_VISIBLE_DEVICES already isolated this
            force_overwrite=(not restart),
            swarm_index=swarm_idx,
            load_state_file=state_file)
        print(f'[swarm] anchor {anchor_idx} swarm {swarm_idx}: done', flush=True)
    except Exception:
        traceback.print_exc()
        if log_path is not None:
            sys.stdout.flush()
        sys.exit(1)


# ── Stage entrypoint ─────────────────────────────────────────────────────────

# ── Phase 1 worker: parallel state generation ───────────────────────────────

def _relax_state_subprocess(prmtop_file, source_pdb, host_resname, guest_resname,
                            gpu_index, restraint_k, minimize_iter, relax_ps,
                            out_xml_str, log_path_str,
                            r_inner_nm=None, r_outer_nm=None,
                            rec_indices=None, lig_indices=None,
                            r_target_nm=None,
                            k_rad_kcal_per_mol_A2=500.0,
                            temperature_K=300.0):
    """
    Subprocess: load the HIDR-deposited PDB, minimize + brief NVT relax with
    backbone restraints, save OpenMM XML state. Stdout/stderr → log file so
    parallel workers don't clobber the parent's terminal.

    Wrapped in try/except so any error (CUDA init failure, NaN, etc.) gets
    written to the log file instead of vanishing into a 0-byte log.
    """
    import os, sys, traceback
    sys.stdout = open(log_path_str, 'w', buffering=1)
    sys.stderr = sys.stdout
    print(f'[worker] starting — gpu={gpu_index}, pdb={source_pdb}', flush=True)
    try:
        import parmed as pmd
        import numpy as np
        from pathlib import Path

        struct = pmd.load_file(str(prmtop_file), str(source_pdb))
        coords = np.array([[a.xx, a.xy, a.xz] for a in struct.atoms])

        # Pre-relax cell-containment check (was in the parent, moved here
        # so the parmed.load_file runs in parallel across workers instead
        # of serializing ~48 × 1-2 sec loads before Phase 1b dispatches).
        # Uses the seekr2 CV indices AND mass-weighted COMs to match
        # seekr2's get_openmm_center_of_mass_com — an unweighted .mean(0)
        # differs by 10-30 mÅ for asymmetric residues, enough to slip
        # past the 20 mÅ tolerance and land outside the cell.
        if (rec_indices is not None and lig_indices is not None
                and r_inner_nm is not None and r_outer_nm is not None):
            rec_arr = list(rec_indices); lig_arr = list(lig_indices)
            rm = np.array([struct.atoms[i].mass for i in rec_arr])
            lm = np.array([struct.atoms[i].mass for i in lig_arr])
            h_com = (coords[rec_arr] * rm[:, None]).sum(0) / rm.sum() * 0.1
            g_com = (coords[lig_arr] * lm[:, None]).sum(0) / lm.sum() * 0.1
            d_nm = float(np.linalg.norm(g_com - h_com))
            tol = 0.020
            if not (r_inner_nm - tol <= d_nm <= r_outer_nm + tol):
                print(f'[worker] source PDB at r={d_nm:.3f} nm outside cell '
                      f'[{r_inner_nm:.3f}, {r_outer_nm:.3f}] (tol={tol}); '
                      f'dropping (no relax)', flush=True)
                sys.exit(2)   # non-zero so dispatcher drops state from anchor_to_states

        # Retry strategies:
        #   NaN (vdW clash):     more minimize iters squash whatever clash
        #                        triggered it.
        #   OUTSIDE_CELL:        the radial restraint balance landed the COM
        #                        past a milestone. Stronger k_rad pulls the
        #                        equilibrium closer to r_target.
        # After retry, a still-failing member is dropped (anchor runs with
        # N<4 swarm members instead of crashing on step 0).
        def _do_relax(min_iter, ps, k_rad_mult=1.0):
            _relax_and_save_state(
                str(prmtop_file), coords, struct.box, struct.atoms,
                host_resname, guest_resname,
                gpu_index, restraint_k, min_iter, ps,
                Path(out_xml_str),
                r_inner_nm=r_inner_nm, r_outer_nm=r_outer_nm,
                rec_indices=rec_indices, lig_indices=lig_indices,
                r_target_nm=r_target_nm,
                k_rad_kcal_per_mol_A2=k_rad_kcal_per_mol_A2 * k_rad_mult,
                temperature_K=temperature_K)

        try:
            _do_relax(minimize_iter, relax_ps)
            print('[worker] OK', flush=True)
        except Exception as e:
            msg = str(e)
            if 'NaN' in msg:
                print(f'[worker] NaN in minimize — retrying with 4× iters '
                      f'({msg.splitlines()[-1]})', flush=True)
                _do_relax(minimize_iter * 4, relax_ps)
                print('[worker] OK on retry', flush=True)
            elif 'OUTSIDE_CELL' in msg:
                print(f'[worker] {msg.splitlines()[-1]} — retrying with 4× k_rad',
                      flush=True)
                _do_relax(minimize_iter, relax_ps, k_rad_mult=4.0)
                print('[worker] OK on retry', flush=True)
            else:
                raise
    except Exception:
        traceback.print_exc()
        sys.stdout.flush()
        sys.exit(1)


def stage_swarm(args):
    import parmed as pmd
    import seekr2.modules.common_base as common_base

    P = paths()
    assert P.json.exists(), 'Run --stage setup first'
    model_xml = P.root / 'model.xml'
    assert model_xml.exists(), \
        'Run --stage run first (need HIDR-deposited starting structures + model.xml)'

    cfg = json.loads(P.json.read_text())
    meta = cfg['_orientation_meta']
    O2_idx  = meta['host_O2_indices']
    O6_idx  = meta['host_O6_indices']
    # CV atom indices for post-relax cell-containment check (must match seekr2).
    rec_indices = list(cfg['workflow']['receptor_indices'])
    lig_indices = list(cfg['workflow']['ligand_indices'])

    # The 4 swarm sources, in swarm-K order. Use root_down/ as a
    # fallback for root_both/ (the renamed deprecated dir).
    SOURCES = []  # list of (k, tag, root_dir)
    SOURCES.append((0, 'A',      P.work / 'root'))
    SOURCES.append((1, 'orient', P.work / 'root_orient'))
    SOURCES.append((2, 'side',   P.work / 'root_side'))
    both_dir = P.work / 'root_both'
    if not both_dir.exists():
        legacy = P.work / 'root_down'
        if legacy.exists():
            both_dir = legacy
            print(f'[swarm] using legacy root_down/ as root_both/ (campaign D)')
    SOURCES.append((3, 'both',   both_dir))

    # Load each available source's seekr2 model
    curdir = os.getcwd()
    source_models = {}
    for k, tag, root_dir in SOURCES:
        if (root_dir / 'model.xml').exists():
            os.chdir(root_dir)
            source_models[k] = common_base.load_model('model.xml')
            os.chdir(curdir)
        else:
            print(f'[swarm] swarm {k} ({tag}): {root_dir} not ready — run --stage hidr_alts')

    if 0 not in source_models:
        raise RuntimeError('root/ has no model.xml — run --stage run first')

    model_main = source_models[0]

    def _milestone_radii(m_anchor):
        """Sorted milestone radii (nm) for an anchor."""
        return sorted(m.variables['radius'] for m in m_anchor.milestones
                      if 'radius' in m.variables)

    def _cell_bounds(m_anchor):
        """(r_inner, r_outer) in nm for the anchor's Voronoi cell."""
        radii = _milestone_radii(m_anchor)
        if not radii:
            return 0.0, float('inf')
        if len(radii) == 1:
            # Single milestone = innermost MD anchor; inner is implicitly 0
            return 0.0, radii[0]
        return radii[0], radii[-1]

    # Each source's anchor order matches C.anchor_radii (with a bulk anchor
    # appended) — the seekrflow workflow places them in this order, identical
    # across all 4 campaigns. So map each non-bulk anchor by its index into
    # C.anchor_radii to get the nominal radius. (Don't compute from milestone
    # midpoints — that breaks for the innermost anchor since its inner
    # boundary is implicitly 0 not the actual r_0.)
    def _nominal_radius_by_index(idx):
        return float(C.anchor_radii[idx]) if idx < len(C.anchor_radii) else None

    source_idx = {}   # k -> dict: radius -> m_anchor
    for k in source_models:
        idx = {}
        a_i = 0
        for a in source_models[k].anchors:
            if a.bulkstate:
                continue
            r = _nominal_radius_by_index(a_i)
            if r is not None:
                idx[round(r, 4)] = a
            a_i += 1
        source_idx[k] = idx

    # Diagnostic: warn if source radii don't all match
    main_radii = sorted(source_idx[0].keys())
    for k, idx in source_idx.items():
        if k == 0:
            continue
        src_radii = sorted(idx.keys())
        if src_radii != main_radii:
            unmatched_main = [r for r in main_radii if not any(abs(r-r2) < 0.01 for r2 in src_radii)]
            unmatched_src  = [r for r in src_radii  if not any(abs(r-r2) < 0.01 for r2 in main_radii)]
            print(f'[swarm] WARNING: swarm {k} has {len(src_radii)} anchors vs root has {len(main_radii)}')
            if unmatched_main: print(f'         main has but source lacks: {unmatched_main}')
            if unmatched_src:  print(f'         source has but main lacks:  {unmatched_src}')

    # ── Phase 1: generate XML state files ──
    cuda_gpus_list = list(C.cuda_gpus)
    n_gpus = max(1, len(cuda_gpus_list))
    mps_per_gpu = max(1, int(getattr(C, 'mps_per_gpu', 1)))
    max_concurrent = mps_per_gpu * n_gpus

    print(f'[swarm] generating up to {len(source_models)} swarm states per anchor '
          f'on {n_gpus} GPU(s) {cuda_gpus_list} (mps_per_gpu={mps_per_gpu}, '
          f'max concurrent={max_concurrent}); minimize {C.swarm_minimize_iter} '
          f'iter + relax {C.swarm_relax_ps} ps with restraints')

    # ── Phase 1a: collect all (anchor,k) → (source_pdb, out_xml) jobs ──
    # Cell-containment + cached-state checks are fast; do them in main process.
    # Defer only the OpenMM minimize+relax work to subprocesses.
    # alpha -> {k: state_file_path}. Dict, NOT list — the K index is the
    # *identity* of the swarm member (controls .out / .dcd suffix and
    # statefile name). A list shifts indices when an entry is dropped, so
    # downstream "for k, path in state_files.items()" stays correct even
    # when Phase 1 drops some K's for an anchor but not others.
    anchor_to_states = {}
    relax_jobs = []           # list of (anchor, k, tag, prmtop, source_pdb, out_xml)

    main_anchor_iter = 0
    for alpha, main_anchor in enumerate(model_main.anchors):
        if main_anchor.bulkstate:
            print(f'[swarm] anchor {alpha}: bulk anchor, skipping')
            continue

        r_inner, r_outer = _cell_bounds(main_anchor)
        r_main_val = _nominal_radius_by_index(main_anchor_iter)
        main_anchor_iter += 1
        if r_main_val is None:
            continue
        r_main = round(r_main_val, 4)

        state_files = {}   # k -> out_xml_str
        for k, tag, root_dir in SOURCES:
            if k not in source_models:
                continue
            tol_match = 0.01
            m_anchor = None
            for r_src, a_src in source_idx[k].items():
                if abs(r_src - r_main) < tol_match:
                    m_anchor = a_src
                    break
            if m_anchor is None:
                print(f'[swarm] anchor {alpha} (r={r_main:.3f}) swarm {k} ({tag}): '
                      f'no matching anchor in source, skipping')
                continue

            if m_anchor.amber_params is None or not m_anchor.amber_params.pdb_coordinates_filename:
                print(f'[swarm] anchor {alpha} swarm {k} ({tag}): no pdb_coordinates_filename, skipping')
                continue

            building = root_dir / m_anchor.directory / m_anchor.building_directory
            prmtop   = building / 'solvated.prmtop'
            pdb      = building / m_anchor.amber_params.pdb_coordinates_filename
            if not pdb.exists() or not prmtop.exists():
                print(f'[swarm] anchor {alpha} swarm {k} ({tag}): missing {pdb.name} or solvated.prmtop, skipping')
                continue

            out_xml = (P.root / main_anchor.directory
                       / main_anchor.building_directory
                       / f'state.swarm_{k}.xml')
            out_xml.parent.mkdir(parents=True, exist_ok=True)

            out_xml_str = str(out_xml.resolve())
            if out_xml.exists():
                print(f'[swarm] anchor {alpha} swarm {k} ({tag}): {out_xml.name} exists '
                      f'— skipping relax')
            else:
                # Defer: queue this for a subprocess in Phase 1b. The
                # worker does its OWN parmed.load_file + cell-containment
                # check in parallel — keeping that check in the parent
                # would serialize ~48 × 1-2 sec PDB loads before any
                # parallel dispatch begins.
                #
                # r_target = cell midpoint (NOT C.anchor_radii[α]). For
                # unevenly-spaced anchors r_main_val sits closer to one
                # milestone than the other, leaving the radial restraint
                # equilibrium dangerously close to that edge. Cell midpoint
                # maximizes margin to both milestones and is robust to
                # spacing asymmetries.
                r_target_mid = 0.5 * (float(r_inner) + float(r_outer))
                relax_jobs.append((alpha, k, tag, str(prmtop), str(pdb),
                                   out_xml_str, 0.0,
                                   float(r_inner), float(r_outer),
                                   float(r_target_mid)))
            state_files[k] = out_xml_str

        if state_files:
            anchor_to_states[alpha] = state_files

    # ── Phase 1b: parallel dispatch of relax jobs across GPUs ──
    if relax_jobs:
        print(f'\n[swarm] Phase 1: dispatching {len(relax_jobs)} state-relax job(s) '
              f'across {max_concurrent} concurrent slots')
        ctx = multiprocessing.get_context('spawn')
        log_dir = P.work / '.swarm_logs'
        log_dir.mkdir(exist_ok=True)
        pending = list(relax_jobs)
        running = []         # (Process, alpha, k, tag, out_xml, d_nm, gpu, log_path)
        launched = 0
        try:
            from tqdm import tqdm
            bar = tqdm(total=len(relax_jobs), desc='swarm relax',
                       unit='state', dynamic_ncols=True)
        except ImportError:
            bar = None

        while pending or running:
            # Reap finished
            for entry in list(running):
                proc, a, kk, tt, oxml, dnm, gpu, lp = entry
                if not proc.is_alive():
                    proc.join()
                    if proc.exitcode != 0:
                        msg = f'anchor {a:2d} swarm {kk} ({tt}): RELAX FAILED ' \
                              f'(exitcode={proc.exitcode}), see {lp}'
                        # Drop the bad K from the dict so seekr2 doesn't
                        # try to load a (missing or partial) state.swarm_K.xml.
                        # Dict-keyed-by-K so the surviving K's keep their
                        # identity — list-and-shift would clip the highest K.
                        d = anchor_to_states.get(a, {})
                        d.pop(kk, None)
                        if not d:
                            anchor_to_states.pop(a, None)
                        if bar is not None: bar.write(f'[swarm] {msg}')
                        else:               print(f'[swarm] {msg}')
                    else:
                        msg = f'anchor {a:2d} swarm {kk} ({tt}): OK'
                        if bar is not None: bar.write(f'[swarm] {msg}')
                        else:               print(f'[swarm] {msg}')
                    if bar is not None: bar.update(1)
                    running.remove(entry)

            # Launch new
            while pending and len(running) < max_concurrent:
                (alpha, k, tag, prmtop_path, pdb_path, out_xml_path, dnm,
                 r_inner_nm, r_outer_nm, r_target_nm) = pending.pop(0)
                # Least-loaded scheduling: pick the GPU with fewest currently-
                # running workers. Ties broken by cuda_gpus_list order.
                # Variable-duration workers (NaN retries take ~3× longer)
                # make launch-order modulo imbalance over time.
                gpu_load = {g: 0 for g in cuda_gpus_list}
                for entry in running:
                    if entry[6] in gpu_load:
                        gpu_load[entry[6]] += 1
                gpu = min(cuda_gpus_list, key=lambda g: gpu_load[g])
                log_path = log_dir / f'relax_a{alpha:02d}_s{k}.log'
                proc = ctx.Process(
                    target=_relax_state_subprocess,
                    args=(prmtop_path, pdb_path,
                          C.host_resname, C.guest_resname,
                          str(gpu), C.swarm_relax_restraint_k,
                          C.swarm_minimize_iter, C.swarm_relax_ps,
                          out_xml_path, str(log_path),
                          r_inner_nm, r_outer_nm,
                          rec_indices, lig_indices,
                          r_target_nm,
                          float(C.directional_k_rad),
                          float(C.temperature_K)),
                    daemon=False,
                )
                proc.start()
                running.append((proc, alpha, k, tag, out_xml_path, dnm,
                                gpu, log_path))
                launched += 1

            if running:
                time.sleep(2)

        if bar is not None:
            bar.close()

    if not anchor_to_states:
        print('[swarm] no anchors have valid swarm states — aborting')
        return

    # ── Phase 1c: deposit multi-frame PDBs to satisfy seekr2's frame assertion.
    # When seekr2 launches with --swarm_index=K, it reads frame K from the
    # anchor's pdb_coordinates_filename. Our state.swarm_K.xml overrides the
    # positions afterward via simulation.loadState(), so the PDB content is
    # immaterial — we just need K+1 frames present. We write a multi-frame
    # PDB containing N copies of the original single-frame HIDR-deposited
    # PDB, and update model.xml to reference it.
    n_frames = len(SOURCES)   # one frame per swarm slot, even if some swarm members were dropped
    print(f'[swarm] writing multi-frame PDBs ({n_frames} frames each) for '
          f'seekr2 swarm-frame assertion...')
    multiframe_pdb_name = None
    for alpha, _ in sorted(anchor_to_states.items()):
        main_anchor = model_main.anchors[alpha]
        building = P.root / main_anchor.directory / main_anchor.building_directory
        if not main_anchor.amber_params or not main_anchor.amber_params.pdb_coordinates_filename:
            continue
        single_pdb = building / main_anchor.amber_params.pdb_coordinates_filename
        if not single_pdb.exists():
            continue
        # Derive a multi-frame filename, e.g. hidr_metadyn_at_0.100_0.pdb → ..._swarm.pdb
        multi_pdb = building / single_pdb.name.replace('_0.pdb', '_swarm.pdb')
        if not multi_pdb.exists():
            _write_multiframe_pdb(single_pdb, n_frames, multi_pdb)
        # Point the anchor's pdb_coordinates_filename at the multi-frame PDB
        main_anchor.amber_params.pdb_coordinates_filename = multi_pdb.name
        multiframe_pdb_name = multi_pdb.name
    # Re-serialize model.xml with the updated filename references
    if multiframe_pdb_name is not None:
        os.chdir(P.root)
        model_main.serialize('model.xml')
        os.chdir(curdir)
        print(f'[swarm] root/model.xml updated to use multi-frame PDB ({multiframe_pdb_name})')

    # ── Phase 2: rolling pool of (anchor, swarm_K) jobs ──
    cuda_gpus = list(C.cuda_gpus)
    mps_per_gpu = max(1, int(C.mps_per_gpu))
    max_concurrent = mps_per_gpu * len(cuda_gpus)

    # Build flat job list in K-major order: all swarm_0 first, then all
    # swarm_1, etc. With seekr2's per-anchor LOCK (one swarm member per
    # anchor at a time), anchor-major ordering would queue (0,0)(0,1)..
    # (0,3) consecutively and the dispatcher's "skip jobs whose anchor is
    # busy" filter forces each anchor to process its 4 members serially —
    # leaving the inner anchors idle once the outer ones become the only
    # pending work, capping tail-end parallelism at n_outer_anchors_left.
    # K-major spreads the 4 K-members across all anchors per wave so the
    # full 8-slot capacity stays filled until the last wave.
    jobs = []   # list of (anchor_idx, swarm_idx, state_file)
    # Iterate K explicitly. state_files is a {k: path} dict; k is the true
    # swarm-K identity, not a list position. Drops in Phase 1 leave gaps
    # in K which we silently skip rather than re-pack.
    all_ks = sorted({k for s in anchor_to_states.values() for k in s})
    for k in all_ks:
        for alpha, state_files in sorted(anchor_to_states.items()):
            if k in state_files:
                jobs.append((alpha, k, state_files[k]))

    print(f'\n[swarm] launching {len(jobs)} (anchor, swarm) jobs '
          f'(GPUs={cuda_gpus}, mps_per_gpu={mps_per_gpu}, max concurrent={max_concurrent})')

    pending = list(jobs)
    running = []   # list of (Process, anchor_idx, swarm_idx, gpu_index)
    running_anchors = set()    # anchors currently in flight — at most ONE swarm
                               # member per anchor at a time, because seekr2's
                               # LOCK file at anchor_N/prod/LOCK is per-anchor
                               # and concurrent processes will kill each other
    launched = 0
    try:
        from tqdm import tqdm
        bar = tqdm(total=len(jobs), desc='swarm prod', unit='job',
                   dynamic_ncols=True)
    except ImportError:
        bar = None

    while pending or running:
        # Reap finished
        for entry in list(running):
            proc, alpha, k, gpu = entry
            if not proc.is_alive():
                proc.join()
                running.remove(entry)
                running_anchors.discard(alpha)
                tag = 'FAILED' if proc.exitcode != 0 else 'done'
                msg = f'anchor {alpha:2d} swarm {k}: {tag} (exitcode={proc.exitcode})'
                if bar is not None:
                    bar.update(1)
                    bar.write(f'[swarm] {msg}')
                else:
                    print(f'[swarm] {msg}')

        # Launch new — pick the first pending job whose anchor is NOT busy.
        # GPU assignment: least-loaded (count currently-running workers per
        # GPU, pick min). Run-times vary widely (anchor depth, swarm member),
        # so launch-order modulo doesn't track real occupancy.
        while pending and len(running) < max_concurrent:
            idx = next((i for i, job in enumerate(pending)
                        if job[0] not in running_anchors), None)
            if idx is None:
                # Every remaining pending job is for an anchor already running;
                # have to wait for one of them to finish.
                break
            alpha, k, state_file = pending.pop(idx)
            gpu_load = {g: 0 for g in cuda_gpus}
            for entry in running:
                if entry[3] in gpu_load:
                    gpu_load[entry[3]] += 1
            gpu_index = min(cuda_gpus, key=lambda g: gpu_load[g])
            log_dir = P.work / '.swarm_logs'
            log_dir.mkdir(exist_ok=True)
            log_path = log_dir / f'run_a{alpha:02d}_s{k}.log'
            proc = multiprocessing.Process(
                target=_run_swarm_member,
                args=(str(model_xml), alpha, k, state_file, gpu_index,
                      str(log_path), int(C.production_steps_per_anchor)),
                daemon=False)
            proc.start()
            running.append((proc, alpha, k, gpu_index))
            running_anchors.add(alpha)
            launched += 1
            launch_msg = (f'anchor {alpha:2d} swarm {k}: launched on GPU {gpu_index} '
                          f'(running={len(running)}/{max_concurrent}, '
                          f'queued={len(pending)})')
            if bar is not None:
                bar.write(f'[swarm] {launch_msg}')
            else:
                print(f'[swarm] {launch_msg}; log: {log_path}')

        # Refresh postfix after BOTH reap and launch have run, so the
        # "running=N" count tracks reality. Setting it only on reap was
        # stale by one dispatch cycle (showed N-1 between a finish and
        # the next finish, even though a launch had refilled the slot).
        if bar is not None:
            bar.set_postfix_str(
                f'running={len(running)}, queued={len(pending)}')

        if running:
            time.sleep(2)

    if bar is not None:
        bar.close()
    print('[swarm] all jobs complete')
