"""Stage: minimize + 3-phase equilibration (NVT restr → NPT restr → NPT free)."""

import math
import sys

import numpy as np

from ._config import C, paths


def stage_equil(args):
    P = paths()
    if P.equil_pdb.exists():
        print('[equil] already done — skipping')
        return

    assert P.solvated_top.exists(), 'Run solvate stage first'
    import parmed as pmd
    from openmm.app import (AmberPrmtopFile, AmberInpcrdFile, Simulation,
                             PME, HBonds, StateDataReporter)
    from openmm import (LangevinMiddleIntegrator, Platform, MonteCarloBarostat,
                        CustomExternalForce)
    import openmm.unit as unit

    print('[equil] loading system...')
    prmtop = AmberPrmtopFile(str(P.solvated_top))
    inpcrd = AmberInpcrdFile(str(P.solvated_rst7))

    system = prmtop.createSystem(
        nonbondedMethod=PME, nonbondedCutoff=0.9 * unit.nanometer,
        constraints=HBonds, rigidWater=True, hydrogenMass=None)

    # Positional restraint on solute heavy atoms (k_r is a global parameter
    # so we can release restraints by setting k_r=0 without rebuilding context)
    restraint = CustomExternalForce("k_r*((x-x0)^2 + (y-y0)^2 + (z-z0)^2)")
    restraint.addGlobalParameter("k_r",
        C.equil_restraint_k * unit.kilocalories_per_mole / unit.angstrom**2)
    restraint.addPerParticleParameter("x0")
    restraint.addPerParticleParameter("y0")
    restraint.addPerParticleParameter("z0")

    pre = pmd.load_file(str(P.solvated_top))
    solute_resnames = {C.host_resname, C.guest_resname}
    n_restrained = 0
    for i, atom in enumerate(pre.atoms):
        if atom.residue.name in solute_resnames and atom.element != 1:
            pos = inpcrd.positions[i].value_in_unit(unit.nanometer)
            restraint.addParticle(i, [pos[0], pos[1], pos[2]])
            n_restrained += 1
    system.addForce(restraint)
    print(f'[equil] {n_restrained} solute heavy atoms restrained '
          f'@ {C.equil_restraint_k} kcal/mol/Å²')

    # Equilibration ALWAYS uses 2 fs regardless of hmr_enabled — density
    # relaxation literature does this conservatively at the standard timestep,
    # the time savings of running equil at 4 fs are tiny (~1 ns total), and
    # an equilibrated box at 2 fs is correct input for HMR-active production.
    T = float(C.temperature_K)
    integrator = LangevinMiddleIntegrator(
        T * unit.kelvin, 1.0 / unit.picosecond, 0.002 * unit.picoseconds)
    platform = Platform.getPlatformByName('CUDA')
    simulation = Simulation(prmtop.topology, system, integrator, platform,
                            {'CudaDeviceIndex': C.gpu_index, 'CudaPrecision': 'mixed'})
    simulation.context.setPositions(inpcrd.positions)
    if inpcrd.boxVectors is not None:
        simulation.context.setPeriodicBoxVectors(*inpcrd.boxVectors)

    print('[equil] minimizing...')
    simulation.minimizeEnergy(maxIterations=5000)
    simulation.context.setVelocitiesToTemperature(T * unit.kelvin)
    simulation.reporters.append(StateDataReporter(
        sys.stdout, C.equil_report, step=True,
        potentialEnergy=True, temperature=True, volume=True, density=True,
        speed=True))

    timestep_ps = 0.002
    nvt_steps      = int(round(C.equil_nvt_ps              / timestep_ps))
    npt_restr_steps = int(round(C.equil_npt_restrained_ps / timestep_ps))
    npt_free_steps  = int(round(C.equil_npt_free_ps        / timestep_ps))

    print(f'\n[equil] Phase 1: NVT {C.equil_nvt_ps} ps ({nvt_steps} steps), '
          f'restraints ON, T={T:.1f} K')
    simulation.step(nvt_steps)

    print(f'\n[equil] Phase 2: NPT {C.equil_npt_restrained_ps} ps '
          f'({npt_restr_steps} steps), restraints ON')
    barostat = MonteCarloBarostat(1.0 * unit.atmosphere, T * unit.kelvin, 25)
    system.addForce(barostat)
    simulation.context.reinitialize(preserveState=True)
    simulation.step(npt_restr_steps)

    print(f'\n[equil] Phase 3: NPT {C.equil_npt_free_ps} ps '
          f'({npt_free_steps} steps), restraints OFF')
    simulation.context.setParameter('k_r', 0.0)
    simulation.step(npt_free_steps)

    state = simulation.context.getState(getPositions=True, enforcePeriodicBox=True)
    positions = state.getPositions()
    bv = state.getPeriodicBoxVectors()
    bv_ang = np.array([[v.value_in_unit(unit.angstrom) for v in row] for row in bv])
    a = np.linalg.norm(bv_ang[0])
    b = np.linalg.norm(bv_ang[1])
    c_ = np.linalg.norm(bv_ang[2])
    alpha = math.degrees(math.acos(np.clip(np.dot(bv_ang[1], bv_ang[2]) / (b * c_), -1, 1)))
    beta  = math.degrees(math.acos(np.clip(np.dot(bv_ang[0], bv_ang[2]) / (a * c_), -1, 1)))
    gamma = math.degrees(math.acos(np.clip(np.dot(bv_ang[0], bv_ang[1]) / (a * b), -1, 1)))

    struct = pmd.load_file(str(P.solvated_top))
    for i, atom in enumerate(struct.atoms):
        pos = positions[i].value_in_unit(unit.angstrom)
        atom.xx, atom.xy, atom.xz = pos
    struct.box = [a, b, c_, alpha, beta, gamma]

    # Center host COM in box center and re-wrap all residues as whole units.
    # OpenMM's enforcePeriodicBox=True wraps atoms individually; if the guest
    # COM lands near a periodic boundary it can end up in a different image from
    # the host, making the raw COM-COM distance misleading and breaking seekr2's
    # CV calculation.  Centering the host + residue-COM wrapping eliminates that.
    box_lengths = np.array([a, b, c_])
    box_center  = box_lengths / 2.0
    host_atoms  = [at for at in struct.atoms if at.residue.name == C.host_resname]
    if host_atoms:
        host_com = np.array([[at.xx, at.xy, at.xz] for at in host_atoms]).mean(0)
        shift = box_center - host_com
        for at in struct.atoms:
            at.xx += shift[0]; at.xy += shift[1]; at.xz += shift[2]
        for res in struct.residues:
            ratoms = list(res.atoms)
            com = np.array([[at.xx, at.xy, at.xz] for at in ratoms]).mean(0)
            wrap = -np.floor(com / box_lengths) * box_lengths
            if np.any(wrap != 0.0):
                for at in ratoms:
                    at.xx += wrap[0]; at.xy += wrap[1]; at.xz += wrap[2]
        guest_atoms = [at for at in struct.atoms if at.residue.name == C.guest_resname]
        if guest_atoms:
            guest_com = np.array([[at.xx, at.xy, at.xz] for at in guest_atoms]).mean(0)
            d = np.linalg.norm(guest_com - box_center)
            print(f'[equil] re-centered: host COM → box center; '
                  f'guest COM {d:.2f} Å from box center')

    struct.save(str(P.equil_pdb), overwrite=True)
    struct.save(str(P.equil_rst7), format='rst7', overwrite=True)

    total_ps = C.equil_nvt_ps + C.equil_npt_restrained_ps + C.equil_npt_free_ps
    print(f'\n[equil] final box: a={a:.2f} b={b:.2f} c={c_:.2f} Å, '
          f'angles=({alpha:.1f},{beta:.1f},{gamma:.1f})°')
    print(f'[equil] total {total_ps} ps — {P.equil_pdb}')
