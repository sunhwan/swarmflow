"""
Helper: construct a seekr2 Model_input directly from C + paths, and invoke
seekr2.prepare.prepare() to write model.xml + anchor tree.

Replaces seekrflow.modules.seekr_input.prepare_model() — same end result, no
seekrflow dependency. Used by setup.py for the main `root/` directory and by
hidr_smd.py / hidr_metad.py for per-campaign `root_<tag>/` directories.
"""

from pathlib import Path

import seekr2.modules.common_base as base
import seekr2.modules.common_cv as common_cv
import seekr2.modules.common_prepare as common_prepare
import seekr2.prepare as seekr2_prepare

from ._config import C


def build_model_input(prmtop_path,
                      pdb_path,
                      root_dir,
                      ligand_indices,
                      receptor_indices,
                      anchor_radii_nm=None,
                      bound_anchor_idx=0):
    """
    Construct a seekr2 Model_input from `C` and the supplied paths.

    Parameters
    ----------
    prmtop_path : Path or str
        AMBER prmtop file (typically work_<name>/solvated.prmtop). Will be
        copied into each anchor's building/ directory by seekr2.prepare.
    pdb_path : Path or str
        Initial-coordinates PDB. All anchors share this file initially; HIDR
        replaces them per anchor. For campaign A: complex-equil.pdb. For
        flipped campaigns: complex-equil-<tag>.pdb.
    root_dir : Path or str
        Where to write model.xml + anchor_* tree (e.g., 'root' or 'root_orient').
        Resolved to absolute by seekr2.prepare.prepare().
    ligand_indices, receptor_indices : list[int]
        Atom indices for the spherical CV's two COM groups.
    anchor_radii_nm : list[float], optional
        Defaults to C.anchor_radii. The LAST entry is the bulk anchor (no MD).
    bound_anchor_idx : int, default 0
        Which anchor index represents the bound state (innermost, typically 0).
    """
    if anchor_radii_nm is None:
        anchor_radii_nm = list(C.anchor_radii)

    n_total = len(anchor_radii_nm)
    bulk_idx = n_total - 1
    # seekr2.prepare needs absolute (or at least cwd-relative) paths so it can
    # locate and COPY the files into each anchor's building/ directory. After
    # the copy, model.xml stores just the basename.
    prmtop_filename = str(Path(prmtop_path).resolve())
    pdb_filename = str(Path(pdb_path).resolve())

    # CV: spherical (COM-COM distance), one anchor per radius.
    cv_input = common_cv.Spherical_cv_input()
    cv_input.index = 0
    cv_input.group1 = list(receptor_indices)
    cv_input.group2 = list(ligand_indices)
    cv_input.bd_group1 = []
    cv_input.bd_group2 = []
    cv_input.input_anchors = []
    cv_input.variable_name = "r"
    cv_input.state_points = []

    for i, r in enumerate(anchor_radii_nm):
        a = common_cv.Spherical_cv_anchor()
        a.radius = float(r)
        a.bound_state = (i == bound_anchor_idx)
        a.bulk_anchor = (i == bulk_idx)

        # MD anchors (every non-bulk index) get amber params; the bulk anchor
        # is BD territory and starts as a Browndye anchor with no MD inputs.
        if not a.bulk_anchor:
            ap = base.Amber_params()
            ap.prmtop_filename = prmtop_filename
            ap.pdb_coordinates_filename = pdb_filename
            # box_vectors set automatically by seekr2 from prmtop on prepare()
            ap.box_vectors = None
            a.starting_amber_params = ap
        cv_input.input_anchors.append(a)

    # MMVT settings — production length comes from C; HMR-aware via the
    # derived production_steps_per_anchor.
    mmvt_settings = common_prepare.MMVT_input_settings()
    mmvt_settings.md_output_interval = int(C.md_output_interval)
    mmvt_settings.md_steps_per_anchor = int(C.production_steps_per_anchor)

    model_input = common_prepare.Model_input()
    model_input.calculation_type = "mmvt"
    model_input.calculation_settings = mmvt_settings
    model_input.temperature = float(C.temperature_K)
    model_input.pressure = 1.0
    model_input.ensemble = "nvt"
    model_input.root_directory = str(root_dir)
    model_input.md_program = "openmm"
    model_input.run_minimization = False
    model_input.hydrogenMass = (float(C.hmr_hydrogen_mass)
                                if C.hmr_enabled else None)
    model_input.constraints = "hbonds"
    model_input.rigidWater = True
    model_input.integrator_type = "langevin"
    model_input.timestep = float(C.production_timestep_ps)
    model_input.nonbonded_cutoff = 0.9
    model_input.browndye_settings_input = None
    model_input.toy_settings_input = None
    model_input.cv_inputs = [cv_input]

    return model_input


def prepare_root(prmtop_path, pdb_path, root_dir,
                 ligand_indices, receptor_indices,
                 force_overwrite=True):
    """
    Build Model_input from C and call seekr2.prepare.prepare() to write
    model.xml + the anchor tree under root_dir. Returns the constructed
    Model_input (useful for callers that want to inspect or re-serialize).

    seekr2.prepare.prepare() copies prmtop_path and pdb_path into each
    anchor's building/ directory, writes root/model.xml with absolute paths
    resolved, and seeds Browndye for the bulk anchor.
    """
    model_input = build_model_input(
        prmtop_path=prmtop_path,
        pdb_path=pdb_path,
        root_dir=root_dir,
        ligand_indices=ligand_indices,
        receptor_indices=receptor_indices,
    )
    seekr2_prepare.prepare(model_input, force_overwrite=force_overwrite)
    return model_input
