"""
Stage: HIDR via metadynamics (single campaign).

Wraps seekrtools.hidr.hidr in metaD mode. Uses the seekr2 model that --stage
setup already built under root/. Seeds the metaD walk from the bound-state
PDB at the bound anchor (anchor 0 by default), and lets seekrtools deposit
"settled" PDBs in each non-bulk anchor's building/ as the bias landscape
fills up.

Differences from --stage hidr_smd (continuous-pull SMD with 4 oriented
campaigns):
  - Single campaign — only root/ is populated, no root_orient/ / root_side/
    / root_both/. swarm Phase 1 will run with N=1 swarm member per anchor
    by default. To get 4 K-members for variance estimation, run with
    --n-swarm-members 4 (TODO: not yet wired) which seeds 4 trajectories
    from the same metaD-deposited pose with different velocity seeds.
  - Direction-blind on |COM-COM| — the CV is just the spherical distance,
    so cholesterol can exit on either side of the cavity. For asymmetric
    binding (where direction matters), prefer hidr_smd.
  - Better cavity-rim sampling — metaD's bias accumulation can find
    pathways through the cavity rim that SMD's straight-line pull misses.

After seekrtools completes, we rename the deposited PDBs from its native
`hidr_settled_at_<idx>.pdb` to the SMD-compatible `hidr_metadyn_at_<r>_0.pdb`
form, so downstream stages (swarm, analyze) read them with no changes.
"""

import os
from pathlib import Path

from ._config import C, paths


def stage_hidr_metad(args):
    P = paths()
    root_xml = P.root / 'model.xml'
    assert root_xml.exists(), \
        'Run --stage setup first (no root/model.xml — setup builds it via seekr2.prepare)'
    assert P.equil_pdb.exists(), 'Run --stage equil first'

    import openmm.unit as unit
    import seekr2.modules.common_base as base
    import seekrtools.hidr.hidr as srt_hidr
    import seekrtools.hidr.hidr_base as hidr_base

    # Load the seekr2 model that setup built (root/model.xml). The model's
    # anchor.amber_params.pdb_coordinates_filename for the bound anchor
    # already points at complex-equil.pdb (copied into building/ by
    # setup), so seekrtools sees a valid starting structure.
    curdir = os.getcwd()
    os.chdir(P.root)
    model = base.load_model('model.xml')
    os.chdir(curdir)

    # Convert ns/ps to step counts at the live (HMR-aware) timestep.
    dt_ps = float(C.production_timestep_ps)
    equil_steps    = int(round(C.metad_equilibration_ns * 1000.0 / dt_ps))
    settling_steps = int(round(C.metad_settling_ps             / dt_ps))
    print(f'[hidr_metad] equilibration: {equil_steps:,} steps '
          f'({C.metad_equilibration_ns} ns @ {dt_ps*1000:.0f} fs)')
    print(f'[hidr_metad] settling:      {settling_steps:,} steps '
          f'({C.metad_settling_ps} ps @ {dt_ps*1000:.0f} fs)')
    print(f'[hidr_metad] metaD: σ={C.metad_sigma_nm} nm, '
          f'biasfactor={C.metad_biasfactor}, '
          f'h={C.metad_height_kjmol} kJ/mol')

    # seekrtools wants pdb_files as a list with one entry per starting
    # anchor. We seed only the bound anchor; metaD walks from there.
    starting_pdb = str(P.equil_pdb.resolve())
    print(f'[hidr_metad] seeding from {starting_pdb}')

    os.chdir(P.work)
    try:
        srt_hidr.hidr(
            model=model,
            destination='any',
            pdb_files=[starting_pdb],
            mode='metadyn',
            equilibration_steps=equil_steps,
            settling_steps=settling_steps,
            settling_frames=1,
            metadyn_sigma=float(C.metad_sigma_nm) * unit.nanometers,
            metadyn_biasfactor=float(C.metad_biasfactor),
            metadyn_height=float(C.metad_height_kjmol)
                           * unit.kilojoules_per_mole,
            force_overwrite=False,
            skip_checks=False,
        )
    finally:
        os.chdir(curdir)

    # seekrtools writes hidr_settled_at_<idx>.pdb per non-bulk anchor and
    # updates anchor.amber_params.pdb_coordinates_filename in the in-memory
    # model + serializes model.xml. Rename to the SMD convention so swarm
    # Phase 1 and downstream tools work without changes.
    print(f'[hidr_metad] renaming settled PDBs to SMD convention '
          f'(hidr_metadyn_at_<r>_0.pdb) ...')
    radii = list(C.anchor_radii)
    n_renamed = 0
    for alpha, anchor in enumerate(model.anchors):
        if anchor.bulkstate:
            continue
        if alpha >= len(radii):
            print(f'[hidr_metad] WARNING: anchor {alpha} has no matching radius')
            continue
        r_alpha = radii[alpha]
        building = P.root / anchor.directory / anchor.building_directory
        cur_name = anchor.amber_params.pdb_coordinates_filename
        cur_pdb  = building / cur_name
        smd_name = f'hidr_metadyn_at_{r_alpha:.3f}_0.pdb'
        smd_pdb  = building / smd_name
        if cur_pdb.exists() and cur_pdb != smd_pdb:
            cur_pdb.rename(smd_pdb)
            anchor.amber_params.pdb_coordinates_filename = smd_name
            n_renamed += 1
        elif smd_pdb.exists():
            anchor.amber_params.pdb_coordinates_filename = smd_name
        else:
            print(f'[hidr_metad] WARNING: anchor {alpha} ({cur_name}) '
                  f'not found in {building}')

    if n_renamed > 0:
        os.chdir(P.root)
        try:
            model.serialize('model.xml')
        finally:
            os.chdir(curdir)
        print(f'[hidr_metad] re-serialized model.xml with {n_renamed} '
              f'updated pdb_coordinates_filename entries')

    print(f'[hidr_metad] done — root/ populated. '
          f'Run --stage swarm next.')
