"""
Helper: construct a seekr2 Model_input directly from C + paths, and invoke
seekr2.prepare.prepare() to write model.xml + anchor tree.

Replaces seekrflow.modules.seekr_input.prepare_model() — same end result, no
seekrflow dependency. Used by setup.py for the main `root/` directory and by
hidr_smd.py / hidr_metad.py for per-campaign `root_<tag>/` directories.
"""

import contextlib
import os
from pathlib import Path

import seekr2.modules.common_base as base
import seekr2.modules.common_cv as common_cv
import seekr2.modules.common_prepare as common_prepare
import seekr2.prepare as seekr2_prepare

from ._config import C


def _count_pqr_atoms(pqr_path):
    """Count ATOM/HETATM lines in a PQR file."""
    n = 0
    with open(pqr_path) as f:
        for line in f:
            if line.startswith(('ATOM', 'HETATM')):
                n += 1
    return n


def _build_bd_settings_input(receptor_indices, ligand_indices):
    """
    Build a seekr2 Browndye_settings_input from C if BD is enabled and both
    PQR paths exist on disk. Returns None when BD is disabled — seekr2
    handles a None browndye_settings_input by skipping the b_surface tree.

    NOTE on indices: seekr2.Browndye_settings_input.receptor_indices /
    ligand_indices index into the PQR files (NOT the solvated-system prmtop).
    They define the binding-site COM atoms for the BD ghost-atom placement.
    The MD-side receptor_indices/ligand_indices passed in here are global
    prmtop atom indices and so are NOT reusable for BD. The tutorial XML
    uses every PQR atom (= COM of the whole host / whole ligand); we mirror
    that by defaulting to range(N_atoms_in_pqr) for each side.
    """
    if not getattr(C, 'bd_enabled', False):
        return None

    rec = getattr(C, 'bd_receptor_pqr', None)
    lig = getattr(C, 'bd_ligand_pqr',   None)
    if not rec or not lig:
        print('[bd] bd_enabled=True but bd_receptor_pqr / bd_ligand_pqr '
              'not set in config.yml — skipping BD setup')
        return None

    rec_path = Path(rec).resolve()
    lig_path = Path(lig).resolve()
    for p in (rec_path, lig_path):
        if not p.exists():
            print(f'[bd] {p} not found — run bd_prep.py first; skipping BD')
            return None

    n_rec_pqr = _count_pqr_atoms(rec_path)
    n_lig_pqr = _count_pqr_atoms(lig_path)

    bd = common_prepare.Browndye_settings_input()
    bd.binary_directory          = str(getattr(C, 'bd_binary_directory', '') or '')
    bd.receptor_pqr_filename     = str(rec_path)
    bd.ligand_pqr_filename       = str(lig_path)
    bd.apbs_grid_spacing         = float(getattr(C, 'bd_apbs_grid_spacing', 0.5))
    bd.num_b_surface_trajectories = int(getattr(C, 'bd_num_b_surface_trajectories', 10000))
    bd.n_threads                 = int(getattr(C, 'bd_n_threads', 1))
    bd.receptor_indices          = list(range(n_rec_pqr))
    bd.ligand_indices            = list(range(n_lig_pqr))

    bd.ions = []
    for ion_cfg in (getattr(C, 'bd_ions', []) or []):
        ion = base.Ion()
        ion.radius = float(ion_cfg['radius'])
        ion.charge = float(ion_cfg['charge'])
        ion.conc   = float(ion_cfg['conc'])
        bd.ions.append(ion)

    print(f'[bd] BD enabled — receptor.pqr={rec_path.name} ({n_rec_pqr} atoms), '
          f'ligand.pqr={lig_path.name} ({n_lig_pqr} atoms), '
          f'n_b_surface={bd.num_b_surface_trajectories}, '
          f'apbs_grid={bd.apbs_grid_spacing} Å')
    return bd


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
    model_input.browndye_settings_input = _build_bd_settings_input(
        receptor_indices=receptor_indices, ligand_indices=ligand_indices)
    model_input.toy_settings_input = None
    model_input.cv_inputs = [cv_input]

    return model_input


@contextlib.contextmanager
def _bd_path_shim():
    """
    Workaround for a seekr2 bug: common_prepare.generate_bd_files() calls
    `make_pqrxml(ligand_pqr_filename)` without forwarding browndye_bin_dir
    (line ~1006 of common_prepare.py), so the ligand pqr2xml invocation
    relies on PATH. If browndye2 isn't on PATH, that call silently writes a
    0-byte ligand.xml and the subsequent APBS step fails. Prepend
    bd_binary_directory to PATH for the duration of seekr2.prepare.
    """
    bin_dir = getattr(C, 'bd_binary_directory', '') or ''
    if not (getattr(C, 'bd_enabled', False) and bin_dir):
        yield
        return
    saved = os.environ.get('PATH', '')
    os.environ['PATH'] = f'{bin_dir}:{saved}' if saved else bin_dir
    try:
        yield
    finally:
        os.environ['PATH'] = saved


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
    with _bd_path_shim():
        seekr2_prepare.prepare(model_input, force_overwrite=force_overwrite)
    return model_input
