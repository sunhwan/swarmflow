"""
Stage: directional SMD HIDR with 4 oriented campaigns (continuous-pull mode).

Alternative to metaD HIDR. Pulls cholesterol along the SIGNED projection of
the ligand onto the host O2→O6 axis — direction-aware, so the deposited
HIDR structures end up on the side intended by the bound-state transform.

Per campaign (A = no flip, B = orient flip, C = side flip + orient flip,
D = side flip):
  1. Apply the geometric transform to the equilibrated bound state and
     relax (minimize + NPT) → complex-equil-<tag>.{pdb,rst7}.
  2. Build the seekr2 model + anchor tree under root_<tag>/ via
     seekr2.prepare.prepare() (campaign A reuses root/ from setup).
  3. Build OpenMM Simulation with:
       • host backbone restraints (35 ring carbons only)
       • directional + radial CustomCentroidBondForces (k_dir + k_rad)
  4. Single continuous ramp through schedule [s_init, signed_radii[0..N-1]];
     update s_target + r_target = |s_target| every chunk.
  5. Post-extract per-anchor frames from pull.dcd (best (s,r) match within
     a ±200 ps window of when s_target hit s_α).
  6. Save snapshot as anchor_α/building/hidr_metadyn_at_<r>_0.pdb and
     re-serialize root_<tag>/model.xml with the new pdb_coordinates_filename.

Output is drop-in compatible with the existing swarm + analysis stages.
"""

import json
import math
import multiprocessing
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np

from ._config import C, paths


# Each entry: (transform_chain_for_bound_state, exit_sign).
#
# `exit_sign` is the side of host_axis to which directional SMD drives
# cholesterol — declared explicitly so the 2×2 (orient × side) lattice at
# the exit is fixed regardless of which side the geometric transform
# happened to put the bound state on.
#
#   A (no transform):           exit +side  (replaces seekrtools' loose metaD HIDR)
#   B (orient flipped):         exit −side  (orient flip flips natural exit)
#   C (side flipped + orient flip): exit −side  (through-cavity transit case)
#   D (side flipped):           exit +side  (orient flip from side_mirror cancels)
#
# 'A' is the main campaign that lives under root/. The other three live under
# root_<tag>/ and are produced via geometric transform + relax + directional pull.
TRANSFORMS = {
    # Order matters for GPU pairing: stage_hidr_smd dispatches launches
    # round-robin across cuda_gpus. With 2 GPUs and order [A, side, orient,
    # both], each GPU gets one through-cavity (slow) + one same-side (fast)
    # campaign so both finish around the same time. Reorder if your bound
    # state inverts the slow/fast classification.
    'A':      {'chain': [],                             'exit_sign': +1.0},
    'side':   {'chain': ['orient_flip', 'side_mirror'], 'exit_sign': -1.0},
    'orient': {'chain': ['orient_flip'],                'exit_sign': -1.0},
    'both':   {'chain': ['side_mirror'],                'exit_sign': +1.0},
}


# ── Coordinate transforms ────────────────────────────────────────────────────

def _orient_flip(coords, guest_idx, guest_axis_idx, host_axis):
    out = coords.copy()
    g_axis = coords[guest_axis_idx[1]] - coords[guest_axis_idx[0]]
    g_axis /= np.linalg.norm(g_axis)
    perp = np.cross(host_axis, g_axis)
    if np.linalg.norm(perp) < 1e-6:
        seed = np.array([1.0, 0.0, 0.0])
        if abs(seed @ g_axis) > 0.9:
            seed = np.array([0.0, 1.0, 0.0])
        perp = np.cross(g_axis, seed)
    perp /= np.linalg.norm(perp)
    R = 2 * np.outer(perp, perp) - np.eye(3)
    g_com = out[guest_idx].mean(0)
    out[guest_idx] = (out[guest_idx] - g_com) @ R.T + g_com
    return out


def _side_mirror(coords, host_idx, guest_idx, host_axis):
    out = coords.copy()
    h_com = out[host_idx].mean(0)
    delta = out[guest_idx] - h_com
    proj_along = (delta @ host_axis)[:, None] * host_axis[None, :]
    out[guest_idx] = h_com + delta - 2 * proj_along
    return out


def _apply_transform_chain(coords, host_idx, guest_idx, guest_axis_idx,
                           host_axis, chain):
    out = coords
    for t in chain:
        if t == 'orient_flip':
            out = _orient_flip(out, guest_idx, guest_axis_idx, host_axis)
        elif t == 'side_mirror':
            out = _side_mirror(out, host_idx, guest_idx, host_axis)
        else:
            raise ValueError(f'Unknown transform: {t}')
    return out


# ── Step 1: relax transformed bound state (NPT, restrained solute) ──────────

def _relax_transformed_complex(prmtop, src_pdb, src_rst7, transform_chain,
                               host_resname, guest_resname, O2_idx, O6_idx,
                               guest_axis_idx, gpu_index, restraint_k,
                               minimize_iter, relax_ps, out_pdb, out_rst7):
    import parmed as pmd
    from openmm.app import (AmberPrmtopFile, AmberInpcrdFile, Simulation,
                             PME, HBonds, StateDataReporter)
    from openmm import (LangevinMiddleIntegrator, Platform, MonteCarloBarostat,
                        CustomExternalForce)
    import openmm.unit as unit

    struct = pmd.load_file(str(prmtop), str(src_pdb))
    coords = np.array([[a.xx, a.xy, a.xz] for a in struct.atoms])
    host_idx  = [i for i, a in enumerate(struct.atoms) if a.residue.name == host_resname]
    guest_idx = [i for i, a in enumerate(struct.atoms) if a.residue.name == guest_resname]

    O2_cen = coords[O2_idx].mean(0)
    O6_cen = coords[O6_idx].mean(0)
    host_axis = O6_cen - O2_cen
    host_axis /= np.linalg.norm(host_axis)

    coords_new = _apply_transform_chain(
        coords, host_idx, guest_idx, guest_axis_idx, host_axis, transform_chain)

    g_com_old = coords[guest_idx].mean(0)
    g_com_new = coords_new[guest_idx].mean(0)
    print(f'[hidr_smd] transform={transform_chain}, ligand COM displacement: '
          f'{np.linalg.norm(g_com_new - g_com_old):.3f} Å')

    prmtop_obj = AmberPrmtopFile(str(prmtop))
    inpcrd = AmberInpcrdFile(str(src_rst7))

    system = prmtop_obj.createSystem(
        nonbondedMethod=PME, nonbondedCutoff=0.9 * unit.nanometer,
        constraints=HBonds, rigidWater=True)

    restraint = CustomExternalForce("k_r*((x-x0)^2 + (y-y0)^2 + (z-z0)^2)")
    restraint.addGlobalParameter(
        "k_r", restraint_k * unit.kilocalories_per_mole / unit.angstrom**2)
    restraint.addPerParticleParameter("x0")
    restraint.addPerParticleParameter("y0")
    restraint.addPerParticleParameter("z0")
    n_restrained = 0
    for i, atom in enumerate(struct.atoms):
        if atom.residue.name in (host_resname, guest_resname) and atom.element != 1:
            restraint.addParticle(i, [coords_new[i, 0] * 0.1,
                                       coords_new[i, 1] * 0.1,
                                       coords_new[i, 2] * 0.1])
            n_restrained += 1
    system.addForce(restraint)

    dt_ps = float(C.production_timestep_ps)
    integrator = LangevinMiddleIntegrator(
        float(C.temperature_K) * unit.kelvin, 1.0 / unit.picosecond, dt_ps * unit.picoseconds)
    platform = Platform.getPlatformByName('CUDA')
    simulation = Simulation(prmtop_obj.topology, system, integrator, platform,
                            {'CudaDeviceIndex': str(gpu_index),
                             'CudaPrecision': 'mixed'})
    simulation.context.setPositions(unit.Quantity(coords_new.tolist(), unit.angstrom))
    if inpcrd.boxVectors is not None:
        simulation.context.setPeriodicBoxVectors(*inpcrd.boxVectors)

    print(f'[hidr_smd] minimizing ({minimize_iter} iter)...')
    simulation.minimizeEnergy(maxIterations=minimize_iter)
    simulation.context.setVelocitiesToTemperature(float(C.temperature_K) * unit.kelvin)
    simulation.reporters.append(StateDataReporter(
        sys.stdout, 5000, step=True, potentialEnergy=True,
        temperature=True, volume=True, density=True, speed=True))

    barostat = MonteCarloBarostat(1.0 * unit.atmosphere, float(C.temperature_K) * unit.kelvin, 25)
    system.addForce(barostat)
    simulation.context.reinitialize(preserveState=True)
    nsteps = int(round(relax_ps / dt_ps))
    print(f'[hidr_smd] NPT relax {relax_ps} ps ({nsteps} steps @ {dt_ps*1000:.0f} fs)...')
    simulation.step(nsteps)

    state = simulation.context.getState(getPositions=True, enforcePeriodicBox=True)
    positions = state.getPositions()
    bv = state.getPeriodicBoxVectors()
    bv_ang = np.array([[v.value_in_unit(unit.angstrom) for v in row] for row in bv])
    a = np.linalg.norm(bv_ang[0]); b = np.linalg.norm(bv_ang[1]); c = np.linalg.norm(bv_ang[2])
    alpha = math.degrees(math.acos(np.clip(np.dot(bv_ang[1], bv_ang[2]) / (b * c), -1, 1)))
    beta  = math.degrees(math.acos(np.clip(np.dot(bv_ang[0], bv_ang[2]) / (a * c), -1, 1)))
    gamma = math.degrees(math.acos(np.clip(np.dot(bv_ang[0], bv_ang[1]) / (a * b), -1, 1)))

    out_struct = pmd.load_file(str(prmtop))
    for i, atom in enumerate(out_struct.atoms):
        pos = positions[i].value_in_unit(unit.angstrom)
        atom.xx, atom.xy, atom.xz = pos
    out_struct.box = [a, b, c, alpha, beta, gamma]
    out_struct.save(str(out_pdb), overwrite=True)
    out_struct.save(str(out_rst7), format='rst7', overwrite=True)
    print(f'[hidr_smd] saved {out_pdb}')


# ── Post-pull frame extraction ───────────────────────────────────────────────

def _extract_anchor_frames(prmtop_file, dcd_path, dcd_interval_ps, pre_pull_ps,
                           host_idx, guest_idx, host_axis,
                           signed_radii_nm, anchor_pdbs,
                           t_at_anchor_ps,
                           cell_bounds_nm=None,
                           window_ps=200.0,
                           bar=None):
    """
    For each anchor, pick the DCD frame closest to (s_target, r_target) within
    a ±window_ps window centered on when s_target was reached, then save the
    chosen frame as a PDB at the predetermined anchor_pdbs[α] path.

    Score: |Δs| + 2·|Δr|. r-match is weighted higher because seekr2's MMVT
    cell-containment is on |COM−COM|; an extracted frame whose |r| is past a
    milestone will fail in swarm regardless of how well s matches.
    """
    import MDAnalysis as mda
    import parmed as pmd

    u = mda.Universe(str(prmtop_file), str(dcd_path))
    n_frames = len(u.trajectory)
    print(f'[hidr_smd]   {n_frames} DCD frames '
          f'({n_frames * dcd_interval_ps:.0f} ps total)')

    # Per-frame (s, r). Frame i (0-indexed) was written at simulation time
    # (i+1) * dcd_interval_ps because DCDReporter writes its FIRST report at
    # step = reportInterval, not step 0.
    frame_data = []
    for ts in u.trajectory:
        pos_ang = u.atoms.positions
        h = pos_ang[host_idx].mean(0) * 0.1
        g = pos_ang[guest_idx].mean(0) * 0.1
        com = g - h
        s = float(com @ host_axis)
        r = float(np.linalg.norm(com))
        t = (ts.frame + 1) * dcd_interval_ps
        frame_data.append((ts.frame, t, s, r))

    for alpha, (s_target, out_pdb) in enumerate(zip(signed_radii_nm, anchor_pdbs)):
        r_target = abs(s_target)
        t_alpha = pre_pull_ps + t_at_anchor_ps[alpha]
        candidates = [fd for fd in frame_data
                      if abs(fd[1] - t_alpha) <= window_ps]
        if not candidates:
            candidates = [min(frame_data, key=lambda x: abs(x[1] - t_alpha))]
        best = min(candidates,
                   key=lambda x: abs(x[2] - s_target) + 2.0 * abs(x[3] - r_target))
        f_idx, t_best, s_best, r_best = best

        cell_warn = ''
        if cell_bounds_nm is not None and alpha < len(cell_bounds_nm):
            r_in, r_out = cell_bounds_nm[alpha]
            margin = 0.020
            if r_best < r_in or r_best > r_out:
                cell_warn = f' [OOB cell=({r_in:.3f},{r_out:.3f})]'
            elif r_best - r_in < margin or r_out - r_best < margin:
                cell_warn = f' [EDGE cell=({r_in:.3f},{r_out:.3f})]'

        match_warn = ''
        if abs(s_best - s_target) > 0.05 or abs(r_best - r_target) > 0.05:
            match_warn = ' [POOR-MATCH]'

        msg = (f'anchor {alpha:2d}: t={t_best:.0f}ps frame={f_idx} '
               f's={s_best:+.3f}/{s_target:+.3f} nm '
               f'|r|={r_best:.3f}/{r_target:.3f} nm'
               f'{match_warn}{cell_warn}')
        if bar is not None:
            bar.set_postfix_str(msg); bar.update(1)
        else:
            print(f'[hidr_smd]   {msg}')

        u.trajectory[f_idx]
        positions_ang = u.atoms.positions
        out_struct = pmd.load_file(str(prmtop_file))
        for i, atom in enumerate(out_struct.atoms):
            atom.xx, atom.xy, atom.xz = positions_ang[i]
        if u.dimensions is not None:
            out_struct.box = list(u.dimensions)
        out_pdb.parent.mkdir(parents=True, exist_ok=True)
        out_struct.save(str(out_pdb), overwrite=True)


# ── Step 4-5: directional SMD pull ───────────────────────────────────────────

def _directional_pull(prmtop_file, start_pdb, start_rst7,
                      host_idx, guest_idx, O2_idx, O6_idx,
                      restraint_idx,
                      signed_radii_nm, anchor_pdbs,
                      pull_velocity_nm_per_ps,
                      k_dir_kcal_per_mol_A2, restraint_k_kcal_per_mol_A2,
                      gpu_index,
                      tolerance_nm=0.005, max_pull_time_ps=1000.0,
                      initial_equil_ps=500.0,
                      k_rad_kcal_per_mol_A2=200.0, dcd_path=None,
                      cell_bounds_nm=None):
    """
    Pull cholesterol's signed projection onto host_axis from its bound
    value to each anchor's signed target, sequentially. Save snapshot at
    each anchor.

    Coordinates inside this function are in OpenMM units (nm); inputs from
    parmed are in Å (we convert).
    """
    import parmed as pmd
    from openmm.app import (AmberPrmtopFile, AmberInpcrdFile, Simulation,
                             PME, HBonds, DCDReporter)
    from openmm import (LangevinMiddleIntegrator, Platform,
                        CustomExternalForce, CustomCentroidBondForce)
    import openmm.unit as unit
    try:
        from tqdm import tqdm
        bar = tqdm(total=len(signed_radii_nm), desc='SMD pull',
                   unit='anchor', dynamic_ncols=True)
    except ImportError:
        bar = None

    # Load topology + box from inpcrd
    prmtop_obj = AmberPrmtopFile(str(prmtop_file))
    inpcrd = AmberInpcrdFile(str(start_rst7))
    struct_init = pmd.load_file(str(prmtop_file), str(start_pdb))
    coords_ang = np.array([[a.xx, a.xy, a.xz] for a in struct_init.atoms])

    O2_cen = coords_ang[O2_idx].mean(0)
    O6_cen = coords_ang[O6_idx].mean(0)
    host_axis = O6_cen - O2_cen
    host_axis /= np.linalg.norm(host_axis)
    h_com_init = coords_ang[host_idx].mean(0)
    g_com_init = coords_ang[guest_idx].mean(0)
    s_init_nm = float((g_com_init - h_com_init) @ host_axis) * 0.1
    print(f'[hidr_smd] directional pull: s_init={s_init_nm:+.3f} nm, '
          f'host_axis=[{host_axis[0]:+.3f}, {host_axis[1]:+.3f}, {host_axis[2]:+.3f}]')

    # System
    system = prmtop_obj.createSystem(
        nonbondedMethod=PME, nonbondedCutoff=0.9 * unit.nanometer,
        constraints=HBonds, rigidWater=True)

    # Backbone restraint — pin the host's macrocycle scaffold (ring
    # carbons, identified at setup time by SMARTS [#6;r6]) so host_axis
    # stays stable, but let OH groups and CH2 bridges flex naturally.
    restraint = CustomExternalForce("k_r*((x-x0)^2 + (y-y0)^2 + (z-z0)^2)")
    restraint.addGlobalParameter(
        "k_r",
        restraint_k_kcal_per_mol_A2 * unit.kilocalories_per_mole / unit.angstrom**2)
    restraint.addPerParticleParameter("x0")
    restraint.addPerParticleParameter("y0")
    restraint.addPerParticleParameter("z0")
    for i in restraint_idx:
        x_nm = coords_ang[i, 0] * 0.1
        y_nm = coords_ang[i, 1] * 0.1
        z_nm = coords_ang[i, 2] * 0.1
        restraint.addParticle(int(i), [x_nm, y_nm, z_nm])
    system.addForce(restraint)
    print(f'[hidr_smd] {len(restraint_idx)} host backbone atoms restrained '
          f'@ {restraint_k_kcal_per_mol_A2} kcal/mol/Å²')

    # Directional SMD: 0.5 * k * (axis · (g_com − h_com) − s_target)^2
    # Pins the SIGNED projection on host_axis — controls direction.
    pull = CustomCentroidBondForce(
        2,
        "0.5*k_dir*(ax*(x2-x1) + ay*(y2-y1) + az*(z2-z1) - s_target)^2"
    )
    pull.addGlobalParameter(
        "k_dir",
        k_dir_kcal_per_mol_A2 * unit.kilocalories_per_mole / unit.angstrom**2)
    pull.addGlobalParameter("ax", float(host_axis[0]))
    pull.addGlobalParameter("ay", float(host_axis[1]))
    pull.addGlobalParameter("az", float(host_axis[2]))
    pull.addGlobalParameter("s_target", s_init_nm)
    g1 = pull.addGroup(host_idx)
    g2 = pull.addGroup(guest_idx)
    pull.addBond([g1, g2], [])
    system.addForce(pull)

    # Radial restraint: 0.5 * k_rad * (|COM-COM| − r_target)^2
    # Pins the COM-COM distance — controls cell containment. Started at
    # k_rad=0 so it's "off" during the directional ramp; turned on after
    # the ramp converges so |r| stays inside the anchor's spherical cell.
    radial = CustomCentroidBondForce(
        2,
        "0.5*k_rad*(distance(g1,g2) - r_target)^2"
    )
    # initial value is set via setParameter() once context is built; the
    # Quantity here just establishes units (kJ/mol/nm² internally)
    radial.addGlobalParameter(
        "k_rad",
        0.0 * unit.kilojoules_per_mole / unit.nanometer**2)
    radial.addGlobalParameter("r_target", abs(s_init_nm))    # placeholder
    g1r = radial.addGroup(host_idx)
    g2r = radial.addGroup(guest_idx)
    radial.addBond([g1r, g2r], [])
    system.addForce(radial)

    # Pre-compute the active k_rad in OpenMM internal units (kJ/mol/nm²)
    k_rad_active = (k_rad_kcal_per_mol_A2
                    * unit.kilocalories_per_mole / unit.angstrom**2
                    ).value_in_unit(unit.kilojoules_per_mole / unit.nanometer**2)

    # Build simulation. Timestep follows hmr_enabled (4 fs with HMR, 2 fs
    # otherwise) — single source of truth in C.production_timestep_ps.
    timestep_ps = float(C.production_timestep_ps)
    integrator = LangevinMiddleIntegrator(
        float(C.temperature_K) * unit.kelvin, 1.0 / unit.picosecond, timestep_ps * unit.picoseconds)
    platform = Platform.getPlatformByName('CUDA')
    simulation = Simulation(prmtop_obj.topology, system, integrator, platform,
                            {'CudaDeviceIndex': str(gpu_index),
                             'CudaPrecision': 'mixed'})
    simulation.context.setPositions(unit.Quantity(coords_ang.tolist(), unit.angstrom))
    if inpcrd.boxVectors is not None:
        simulation.context.setPeriodicBoxVectors(*inpcrd.boxVectors)
    simulation.context.setVelocitiesToTemperature(float(C.temperature_K) * unit.kelvin)

    # DCD reporter for post-hoc inspection of the pull trajectory.
    # dcd_interval_ps controls write frequency: 50 ps default → ~50 MB/campaign;
    # 10 ps → ~250 MB/campaign. Extractor needs ≥4 frames per anchor's window
    # for a good (s, r) match; with the default ±200 ps window, even 100 ps
    # interval works.
    dcd_interval_ps_local = float(getattr(C, 'directional_dcd_interval_ps', 50.0))
    dcd_interval_steps = max(100, int(dcd_interval_ps_local / timestep_ps))
    if dcd_path is not None:
        Path(dcd_path).parent.mkdir(parents=True, exist_ok=True)
        simulation.reporters.append(DCDReporter(str(dcd_path), dcd_interval_steps))

    # Brief equilibration of solvent at the bound state (5 ps regardless of dt)
    simulation.step(int(round(5.0 / timestep_ps)))

    def _measure_state():
        """Read current signed projection AND |r| (both nm)."""
        st = simulation.context.getState(getPositions=True)
        pos_nm = np.asarray(st.getPositions(asNumpy=True).value_in_unit(unit.nanometer))
        h = pos_nm[host_idx].mean(0)
        g = pos_nm[guest_idx].mean(0)
        com_vec = g - h
        return float(com_vec @ host_axis), float(np.linalg.norm(com_vec))

    # Radial restraint ON throughout the continuous ramp, with r_target
    # tracking |s_target| each chunk. This pins cholesterol to the host
    # axis (|r| ≈ |s|) instead of letting perpendicular drift accumulate
    # over 30+ ns of slow pull. Through-cavity transit (s_target → 0) also
    # gets r_target → 0, which is the same configuration the old per-anchor
    # design used for inner anchors — works because there's no host atom at
    # the geometric center of the BCD ring (the COM is just empty space).
    simulation.context.setParameter('k_rad', k_rad_active)
    simulation.context.setParameter('r_target', abs(s_init_nm))

    # ── Continuous SMD ramp through ALL anchors ─────────────────────────
    # Schedule: [s_init, signed_radii[0], signed_radii[1], ...].
    # The first leg handles arbitrary starts: backward (s_init past s_α[0] in
    # exit_sign direction), forward, or through-cavity (s_init and s_α[0]
    # have opposite signs). Subsequent legs are forward through outer
    # anchors. Both restraints (k_dir + k_rad) are active throughout, with
    # r_target updated to track |s_target| each chunk — pins cholesterol to
    # the host axis at radius matching the axial target.
    schedule = [s_init_nm] + list(signed_radii_nm)
    print(f'[hidr_smd] continuous schedule (nm): '
          f'{[f"{s:+.3f}" for s in schedule]}')

    # Optional pre-pull equilibration at s_init (post-flip campaigns can
    # still be relaxing). The 5 ps warm-up at line 336 is always done; this
    # adds the rest if initial_equil_ps > 5.
    pre_pull_ps = 5.0
    if initial_equil_ps > pre_pull_ps:
        extra_ps = initial_equil_ps - pre_pull_ps
        print(f'[hidr_smd] pre-pull equilibration: {extra_ps:.0f} ps at s={s_init_nm:+.3f}')
        simulation.context.setParameter('s_target', s_init_nm)
        simulation.step(int(extra_ps / timestep_ps))
        pre_pull_ps = initial_equil_ps

    chunk = 100
    t_at_anchor_ps = []   # cumulative pull time after pre_pull; 1 entry per anchor
    cumulative_ps = 0.0

    for i in range(1, len(schedule)):
        s_from = schedule[i - 1]
        s_to   = schedule[i]
        leg_ps = abs(s_to - s_from) / pull_velocity_nm_per_ps
        leg_steps = max(100, int(leg_ps / timestep_ps))
        n_chunks = max(1, leg_steps // chunk)
        for c in range(n_chunks):
            s_now = s_from + (s_to - s_from) * (c + 1) / n_chunks
            simulation.context.setParameter("s_target", s_now)
            simulation.context.setParameter("r_target", abs(s_now))
            simulation.step(chunk)
        # Round-off correction so the next leg starts at exactly s_to.
        simulation.context.setParameter("s_target", s_to)
        simulation.context.setParameter("r_target", abs(s_to))
        cumulative_ps += leg_ps
        t_at_anchor_ps.append(cumulative_ps)
        # Progress: 1 tick per leg (anchor reached). Postfix shows where we
        # are in the schedule and how much pull-time has elapsed.
        if bar is not None:
            s_actual, r_actual = _measure_state()
            bar.set_postfix_str(
                f's={s_actual:+.3f}/{s_to:+.3f} nm '
                f'|r|={r_actual:.3f}/{abs(s_to):.3f} nm '
                f't={cumulative_ps/1000:.1f} ns')
            bar.update(1)
        else:
            print(f'[hidr_smd]   anchor {i-1:2d} reached: '
                  f's_target={s_to:+.3f} nm, t={cumulative_ps/1000:.2f} ns')

    # No final hold or minimize — the ramp itself provides ≥4 DCD candidates
    # in the outermost anchor's ±window, and minimize doesn't write to DCD
    # so the post-minimized state would never be seen by the extractor.

    if bar is not None:
        bar.close()

    # Extract per-anchor frames from the DCD.
    print(f'[hidr_smd] extracting per-anchor frames from {dcd_path}')
    _extract_anchor_frames(
        prmtop_file=prmtop_file,
        dcd_path=dcd_path,
        dcd_interval_ps=dcd_interval_ps_local,
        pre_pull_ps=pre_pull_ps,
        host_idx=host_idx, guest_idx=guest_idx, host_axis=host_axis,
        signed_radii_nm=signed_radii_nm,
        anchor_pdbs=anchor_pdbs,
        t_at_anchor_ps=t_at_anchor_ps,
        cell_bounds_nm=cell_bounds_nm)


# ── Run one campaign (relax + prepare + directional pull) ────────────────────

def _run_one_campaign(tag, transform_chain, exit_sign):
    """
    Campaign A (tag='A') uses complex-equil.{pdb,rst7} directly (no transform)
    and uses root/ (already built by setup). Other campaigns apply a
    geometric transform, save to complex-equil-<tag>.{pdb,rst7}, and build
    root_<tag>/ directly via seekr2.prepare.prepare() (no seekrflow involved).
    """
    P = paths()
    is_main = (tag == 'A')

    if is_main:
        out_pdb  = P.equil_pdb           # complex-equil.pdb (from --stage equil)
        out_rst7 = P.equil_rst7
        root_dir = P.root                # root/
    else:
        out_pdb  = P.work / f'complex-equil-{tag}.pdb'
        out_rst7 = P.work / f'complex-equil-{tag}.rst7'
        root_dir = P.work / f'root_{tag}'

    if (root_dir / 'model.xml').exists():
        anchor_pdbs = list(root_dir.glob('anchor_*/building/hidr_metadyn_at_*.pdb'))
        if anchor_pdbs:
            print(f'[hidr_smd] tag={tag}: already done — '
                  f'{len(anchor_pdbs)} anchors in {root_dir}')
            return

    cfg  = json.loads(P.json.read_text())
    meta = cfg['_orientation_meta']
    O2_idx = meta['host_O2_indices']
    O6_idx = meta['host_O6_indices']
    guest_axis_idx = meta['guest_axis_indices']

    if is_main:
        if not out_pdb.exists():
            raise RuntimeError(
                f'{out_pdb} not found — run --stage equil first')
        print(f'[hidr_smd] tag=A: using {out_pdb} (no transform)')
    elif not out_pdb.exists():
        print(f'[hidr_smd] tag={tag}: relaxing transformed complex...')
        _relax_transformed_complex(
            P.solvated_top, P.equil_pdb, P.equil_rst7, transform_chain,
            C.host_resname, C.guest_resname, O2_idx, O6_idx, guest_axis_idx,
            C.cuda_gpus[0],
            C.equil_restraint_k, 2500, 500.0,
            out_pdb, out_rst7)
    else:
        print(f'[hidr_smd] tag={tag}: reusing existing {out_pdb}')

    out_pdb_abs  = out_pdb.resolve()
    out_rst7_abs = out_rst7.resolve()
    root_dir_abs = root_dir.resolve()

    # Prepare model.xml + anchor tree (or skip for campaign A where setup
    # already built it). For non-A campaigns, build root_<tag>/ directly via
    # seekr2.prepare — no seekrflow JSON dance, no ROOT monkey-patch.
    import seekr2.modules.common_base as common_base
    import parmed as pmd
    from ._seekr_model import prepare_root

    lig_idx = cfg['workflow']['ligand_indices']
    rec_idx = cfg['workflow']['receptor_indices']

    curdir = os.getcwd()
    if is_main:
        # setup stage already built root/model.xml using complex-equil.pdb.
        # Verify and skip rebuild.
        root_xml = root_dir / 'model.xml'
        if not root_xml.exists():
            print(f'[hidr_smd] tag={tag}: setup did not build {root_xml} — '
                  f'rebuilding')
            prepare_root(prmtop_path=P.solvated_top.resolve(),
                         pdb_path=out_pdb_abs,
                         root_dir=root_dir_abs,
                         ligand_indices=lig_idx,
                         receptor_indices=rec_idx,
                         force_overwrite=True)
        else:
            print(f'[hidr_smd] tag={tag}: reusing root/model.xml from setup')
    else:
        print(f'[hidr_smd] tag={tag}: preparing model.xml in {root_dir.name}/...')
        prepare_root(prmtop_path=P.solvated_top.resolve(),
                     pdb_path=out_pdb_abs,
                     root_dir=root_dir_abs,
                     ligand_indices=lig_idx,
                     receptor_indices=rec_idx,
                     force_overwrite=True)

    # Pull cholesterol to exit_sign × r_α for each anchor. exit_sign is
    # declared per campaign (TRANSFORMS dict) so the (orient × side)
    # 2×2 lattice is enforced at the EXIT, not inferred from the bound
    # state. For C in particular, this means dragging cholesterol from
    # +bound through the cavity to −exit (the through-cavity transit case).
    struct = pmd.load_file(str(P.solvated_top), str(out_pdb_abs))
    coords = np.array([[a.xx, a.xy, a.xz] for a in struct.atoms])
    host_idx_atoms  = [i for i, a in enumerate(struct.atoms)
                       if a.residue.name == C.host_resname]
    guest_idx_atoms = [i for i, a in enumerate(struct.atoms)
                       if a.residue.name == C.guest_resname]
    host_axis = coords[O6_idx].mean(0) - coords[O2_idx].mean(0)
    host_axis /= np.linalg.norm(host_axis)
    h_com = coords[host_idx_atoms].mean(0)
    g_com = coords[guest_idx_atoms].mean(0)
    s_bound_nm = float((g_com - h_com) @ host_axis) * 0.1
    print(f'[hidr_smd] tag={tag}: bound s={s_bound_nm:+.3f} nm, '
          f'exit_sign={exit_sign:+.0f}'
          + ('   (through-cavity transit)' if s_bound_nm * exit_sign < 0 else ''))

    # Build anchor target list (signed) and output paths
    os.chdir(root_dir_abs)
    model = common_base.load_model('model.xml')
    os.chdir(curdir)

    signed_radii = []
    anchor_out_pdbs = []
    # seekr2.prepare places anchors in the same order as C.anchor_radii
    # (with a bulk anchor appended at the end), so the mapping from non-bulk
    # anchor index → nominal radius is direct.
    # Don't try to derive r_alpha from milestone midpoints — that breaks
    # for the innermost anchor (only one milestone exists, and the implicit
    # inner=0 makes the midpoint = half the outer milestone, not r_α).
    radii_iter = iter(C.anchor_radii)
    for alpha, anchor in enumerate(model.anchors):
        if anchor.bulkstate:
            continue
        try:
            r_alpha = next(radii_iter)
        except StopIteration:
            print(f'[hidr_smd] WARNING: anchor {alpha} has no matching radius in C.anchor_radii')
            break
        signed_radii.append(exit_sign * r_alpha)

        anchor_dir = root_dir_abs / anchor.directory / anchor.building_directory
        anchor_dir.mkdir(parents=True, exist_ok=True)
        fname = f'hidr_metadyn_at_{r_alpha:.3f}_0.pdb'
        anchor_out_pdbs.append(anchor_dir / fname)
        if anchor.amber_params is not None:
            anchor.amber_params.pdb_coordinates_filename = fname

    # Run the directional pull
    velocity_nm_per_ps = C.directional_pull_velocity_nm_per_ns / 1000.0
    print(f'[hidr_smd] tag={tag}: pulling through {len(signed_radii)} anchors '
          f'(velocity={C.directional_pull_velocity_nm_per_ns:.2f} nm/ns)')
    # Backbone restraint set: the receptor_indices computed at setup time
    # by SMARTS [#6;r6] (35 ring carbons for BCD). Stored in the metadata
    # JSON written by setup. receptor_indices is identical across campaigns
    # since they share the same prmtop and SMARTS-discovered atoms.
    backbone_idx = rec_idx

    # Per-MD-anchor cell milestones (matches kinetics.py + seekr2 convention:
    # midpoints between anchor radii; innermost has r_inner=0). Used only for
    # logging here — same bounds the swarm stage will check post-relax.
    radii = list(C.anchor_radii)
    n_md = len(signed_radii)   # excludes bulk
    cell_bounds_nm = []
    for a in range(n_md):
        r_in  = 0.0 if a == 0 else 0.5 * (radii[a - 1] + radii[a])
        r_out = 0.5 * (radii[a] + radii[a + 1])  # next entry exists (bulk or MD)
        cell_bounds_nm.append((r_in, r_out))

    pull_dcd = root_dir_abs / 'pull.dcd'
    _directional_pull(
        P.solvated_top, out_pdb_abs, out_rst7_abs,
        host_idx_atoms, guest_idx_atoms, O2_idx, O6_idx,
        backbone_idx,
        signed_radii, anchor_out_pdbs,
        velocity_nm_per_ps,
        C.directional_k_dir, C.directional_restraint_k,
        C.cuda_gpus[0],
        tolerance_nm=C.directional_tolerance_nm,
        max_pull_time_ps=C.directional_max_pull_time_ps,
        initial_equil_ps=C.directional_initial_equil_ps,
        k_rad_kcal_per_mol_A2=C.directional_k_rad,
        dcd_path=pull_dcd,
        cell_bounds_nm=cell_bounds_nm)

    # Save updated model.xml (with new pdb_coordinates_filename per anchor)
    os.chdir(root_dir_abs)
    model.serialize('model.xml')
    os.chdir(curdir)

    # Copy solvated.prmtop into each anchor's building/ (HIDR normally does this)
    for alpha, anchor in enumerate(model.anchors):
        if anchor.bulkstate:
            continue
        anchor_dir = root_dir_abs / anchor.directory / anchor.building_directory
        prmtop_dst = anchor_dir / 'solvated.prmtop'
        if not prmtop_dst.exists():
            shutil.copy(str(P.solvated_top.resolve()), str(prmtop_dst))

    print(f'[hidr_smd] tag={tag}: done — directional HIDR structures in {root_dir}')


def _campaign_worker(tag, chain, exit_sign, gpu_index, work_dir_str,
                     config_dict, log_path_str):
    """
    Subprocess: run one campaign on a specific GPU. stdout/stderr go to
    log_path so multiple parallel campaigns don't clobber the parent's
    terminal.
    """
    import os, sys
    os.chdir(work_dir_str)
    sys.stdout = open(log_path_str, 'w', buffering=1)
    sys.stderr = sys.stdout

    # Re-populate swarmflow._config.C in this subprocess
    from swarmflow import _config as cfg_mod
    cfg_mod.C.__dict__.clear()
    cfg_mod.C.__dict__.update(config_dict)
    cfg_mod.C.cuda_gpus = [str(gpu_index)]    # restrict to one GPU

    from swarmflow.hidr_smd import _run_one_campaign
    _run_one_campaign(tag, chain, exit_sign)


def stage_hidr_smd(args):
    """Run the 3 alternate HIDR campaigns (B, C, D) in parallel across GPUs."""
    P = paths()
    assert P.equil_pdb.exists(), 'Run --stage equil first'
    assert P.json.exists(),     'Run --stage setup first'

    cuda_gpus = list(C.cuda_gpus)
    n_gpus = max(1, len(cuda_gpus))
    mps_per_gpu = max(1, int(getattr(C, 'mps_per_gpu', 1)))
    max_concurrent = mps_per_gpu * n_gpus
    config_dict = dict(C.__dict__)
    cwd_str = os.getcwd()

    # Use spawn so CUDA contexts in the parent don't inherit.
    ctx = multiprocessing.get_context('spawn')

    # Build queue of (tag, chain, exit_sign) — only campaigns not yet done
    pending = []
    for tag, spec in TRANSFORMS.items():
        root_dir = P.work / f'root_{tag}'
        if (root_dir / 'model.xml').exists():
            anchor_pdbs = list(root_dir.glob('anchor_*/building/hidr_metadyn_at_*.pdb'))
            if anchor_pdbs:
                print(f'[hidr_smd] campaign {tag.upper()}: already done '
                      f'({len(anchor_pdbs)} anchors), skipping')
                continue
        pending.append((tag, spec['chain'], spec['exit_sign']))

    if not pending:
        print('[hidr_smd] nothing to do')
        return

    # Chunked GPU assignment: with N campaigns ÷ n_gpus, each GPU gets
    # ceil(N/n_gpus) consecutive launches. With N=4 and TRANSFORMS in
    # alternating slow/fast order (same-side, through-cavity, same-side,
    # through-cavity relative to the bound state), this pairs one slow +
    # one fast on each GPU — vs round-robin which would put all slow on
    # one GPU and all fast on the other.
    n_pending = len(pending)
    campaigns_per_gpu = max(1, (n_pending + n_gpus - 1) // n_gpus)
    print(f'[hidr_smd] dispatching {n_pending} campaign(s) across '
          f'{n_gpus} GPU(s) {cuda_gpus}, mps_per_gpu={mps_per_gpu} '
          f'(max concurrent={max_concurrent}, '
          f'{campaigns_per_gpu} per GPU)')
    running = []   # list of (Process, tag, gpu_index, log_path)
    launched = 0

    while pending or running:
        # Reap finished
        for entry in list(running):
            proc, tag, gpu, log_path = entry
            if not proc.is_alive():
                proc.join()
                status = 'done' if proc.exitcode == 0 \
                    else f'FAILED (exitcode={proc.exitcode})'
                print(f'[hidr_smd] campaign {tag.upper()}: {status} '
                      f'(GPU {gpu}); log: {log_path}')
                running.remove(entry)

        # Launch new — concurrency cap = mps_per_gpu * n_gpus.
        # Requires the NVIDIA MPS daemon for >1 process per GPU to actually
        # share rather than time-slice.
        while pending and len(running) < max_concurrent:
            tag, chain, exit_sign = pending.pop(0)
            gpu = cuda_gpus[min(n_gpus - 1, launched // campaigns_per_gpu)]
            log_path = P.work / f'hidr_smd_{tag}.log'
            proc = ctx.Process(
                target=_campaign_worker,
                args=(tag, chain, exit_sign, gpu, cwd_str,
                      config_dict, str(log_path)),
                daemon=False,
            )
            proc.start()
            running.append((proc, tag, gpu, log_path))
            launched += 1
            print(f'[hidr_smd] launched campaign {tag.upper()} on GPU {gpu} '
                  f'(running={len(running)}/{max_concurrent}, queued={len(pending)}); '
                  f'log: {log_path}')

        if running:
            time.sleep(5)

    print('[hidr_smd] all campaigns complete')
