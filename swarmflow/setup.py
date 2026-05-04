"""
Stage: compute SMARTS-based atom indices, write metadata JSON, and build
the campaign-A seekr2 model + anchor tree under root/.

The JSON (work_<name>/seekrflow_<name>.json) is now a metadata-only snapshot
of indices + paths consumed by hidr_smd, swarm, analyze, and the helper
scripts. Its name is legacy from the seekrflow era but kept for caller
compatibility (diagnose_hidr_paths.py, merge_swarm_pdbs.py, etc. glob for
`seekrflow_*.json`).

The seekr2 model (root/model.xml + anchor tree) is built directly via
seekr2.prepare.prepare() — no more seekrflow dependency. Alt-campaign roots
(root_orient/, root_side/, root_both/) are still built later by hidr_smd.
"""

import json

import numpy as np

from ._config import C, paths
from ._seekr_model import prepare_root


def _compute_guest_axis(struct, guest_heavy: list) -> list:
    """
    Find two guest atoms that best represent the principal axis.
    Priority: ring + sp3 + ≥2 heavy neighbors → ring → sp3 → any interior → any.
    """
    if C.guest_axis_atoms is not None:
        name_to_idx = {struct.atoms[i].name: i for i in guest_heavy}
        indices = [name_to_idx[n] for n in C.guest_axis_atoms]
        print(f'[setup] Guest axis atoms (user): {C.guest_axis_atoms} -> {indices}')
        return indices

    coords = np.array([[struct.atoms[i].xx, struct.atoms[i].xy, struct.atoms[i].xz]
                       for i in guest_heavy])
    centered = coords - coords.mean(0)
    _, _, Vt = np.linalg.svd(centered, full_matrices=False)
    axis = Vt[0]
    proj_map = {i: (centered[k] @ axis) for k, i in enumerate(guest_heavy)}

    heavy_set = set(guest_heavy)
    heavy_neighbors = {}
    for i in guest_heavy:
        atom = struct.atoms[i]
        n_heavy = sum(1 for bond in atom.bonds
                      for nb in [bond.atom1, bond.atom2]
                      if nb is not atom and nb.idx in heavy_set)
        heavy_neighbors[i] = n_heavy

    import networkx as nx
    G = nx.Graph()
    G.add_nodes_from(guest_heavy)
    for i in guest_heavy:
        for bond in struct.atoms[i].bonds:
            nb = bond.atom1 if bond.atom2 is struct.atoms[i] else bond.atom2
            if nb.idx in heavy_set:
                G.add_edge(i, nb.idx)
    ring_atoms = set()
    for cycle in nx.cycle_basis(G):
        ring_atoms.update(cycle)

    def is_sp3(idx):
        atom = struct.atoms[idx]
        if atom.element_name == 'C':
            return atom.type.lower() in ('c3', 'cx')
        return atom.element_name in ('O', 'N', 'S')

    for tier_label, candidates in [
        ('ring + sp3 + ≥2 heavy neighbors',
         [i for i in guest_heavy if i in ring_atoms and is_sp3(i) and heavy_neighbors[i] >= 2]),
        ('ring + ≥2 heavy neighbors',
         [i for i in guest_heavy if i in ring_atoms and heavy_neighbors[i] >= 2]),
        ('sp3 + ≥2 heavy neighbors',
         [i for i in guest_heavy if is_sp3(i) and heavy_neighbors[i] >= 2]),
        ('any + ≥2 heavy neighbors',
         [i for i in guest_heavy if heavy_neighbors[i] >= 2]),
        ('any heavy atom', guest_heavy),
    ]:
        if len(candidates) >= 2:
            break

    cand_sorted = sorted(candidates, key=lambda i: proj_map[i])
    idx_min, idx_max = cand_sorted[0], cand_sorted[-1]
    a_min, a_max = struct.atoms[idx_min], struct.atoms[idx_max]
    print(f'[setup] Guest axis ({tier_label}): '
          f'{a_min.name}({a_min.type}) idx={idx_min} ↔ '
          f'{a_max.name}({a_max.type}) idx={idx_max}')
    return [idx_min, idx_max]


def compute_indices(prmtop: str, pdb: str):
    from rdkit import Chem
    import parmed as pmd

    struct = pmd.load_file(prmtop, pdb)
    host_offset = next(i for i, a in enumerate(struct.atoms)
                       if a.residue.name == C.host_resname)

    tmp_pdb = paths().work / '_host_tmp.pdb'
    struct[f':{C.host_resname}'].save(str(tmp_pdb), overwrite=True)
    mol = Chem.MolFromPDBFile(str(tmp_pdb), removeHs=False, sanitize=False)
    Chem.SanitizeMol(mol,
        sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL
                   ^ Chem.SanitizeFlags.SANITIZE_PROPERTIES)
    tmp_pdb.unlink()

    patt_bb = Chem.MolFromSmarts(C.host_backbone_smarts)
    bb_local = [m[0] for m in mol.GetSubstructMatches(patt_bb)]

    patt_ax = Chem.MolFromSmarts(C.host_o2_o6_smarts)
    O2_local, O6_local = set(), set()
    for match in mol.GetSubstructMatches(patt_ax):
        for qi, mi in enumerate(match):
            mapnum = patt_ax.GetAtomWithIdx(qi).GetAtomMapNum()
            if mapnum == 1:
                O2_local.add(mi)
            elif mapnum == 2:
                O6_local.add(mi)

    print(f'[setup] Host backbone ring-C: {len(bb_local)} (expect 35 for BCD)')
    print(f'[setup] Host O2: {len(O2_local)} (expect 7), O6: {len(O6_local)} (expect 7)')

    host_backbone  = [i + host_offset for i in bb_local]
    host_O2_global = [i + host_offset for i in sorted(O2_local)]
    host_O6_global = [i + host_offset for i in sorted(O6_local)]

    guest_heavy = [i for i, a in enumerate(struct.atoms)
                   if a.residue.name == C.guest_resname and a.element_name != 'H']

    guest_axis = _compute_guest_axis(struct, guest_heavy)

    return {
        'receptor_indices': host_backbone,
        'ligand_indices':   guest_heavy,
        'host_O2_indices':  host_O2_global,
        'host_O6_indices':  host_O6_global,
        'guest_axis_indices': guest_axis,
    }


def stage_setup(args):
    P = paths()
    root_xml = P.root / 'model.xml'
    if P.json.exists() and root_xml.exists():
        print(f'[setup] {P.json.name} and {root_xml} both exist — skipping')
        return

    assert P.solvated_top.exists(), 'Run solvate stage first'
    assert P.equil_pdb.exists(),    'Run equil stage first'

    indices = compute_indices(str(P.solvated_top), str(P.equil_pdb))

    config = {
        'name': C.name,
        'structure_version': '1.2',
        'workflow': {
            'type': 'protein_ligand_seekr2',
            'solvated_system_for_md': {
                'parameters_topology': {
                    'type': 'Amber',
                    'prmtop_filename': str(P.solvated_top.resolve()),
                },
                'solvated_pdb': str(P.equil_pdb.resolve()),
            },
            'ligand_indices':   indices['ligand_indices'],
            'receptor_indices': indices['receptor_indices'],
            'receptor_pqr_filename_for_bd': '',
            'ligand_pqr_filename_for_bd':   '',
            'parameterizer_information': {
                'ligand_sdf_file': '',
                'ligand_resname': C.guest_resname,
                'receptor_ligand_pdb_filename': str(P.equil_pdb.resolve()),
            },
            'hidr_settings': {
                'type': 'hidr_metaD',
                'gaussian_height': 0.5,
                'gaussian_width': 0.05,
                'bias_factor': 10.0,
            },
            'mmvt_settings': {
                'type': 'MMVT',
                'md_output_interval': C.md_output_interval,
                'md_steps_per_anchor': C.production_steps_per_anchor,
                'cv_type': 'com_com_distance',
                'anchor_radius_list': C.anchor_radii,
            },
            'md_settings': {
                'type': 'MD',
                'engine': 'openmm',
                'integrator': 'langevin',
                'nonbonded_cutoff': 0.9,
                'friction': 1.0,
                'barostat_period': None,
                'stepsize': float(C.production_timestep_ps),
            },
            'bd_settings': None,
        },
        'physical_attributes': {
            'temperature': float(C.temperature_K),
            'pressure': None,
            'ionic_strength': C.ion_conc_m,
            'hydrogen_mass': float(C.hmr_hydrogen_mass) if C.hmr_enabled else 1.008,
        },
        'work_directory': C.work_dir,
        'root_directory': None,
        'parameterizer': None,
        'run_settings': {
            'resources': [],
            'bd_stage_resource_name': 'local',
            'hidr_stage_resource_name': 'local',
            'seekr_stage_resource_name': 'local',
        },
        '_orientation_meta': {
            'host_O2_indices':    indices['host_O2_indices'],
            'host_O6_indices':    indices['host_O6_indices'],
            'guest_axis_indices': indices['guest_axis_indices'],
            'prmtop': str(P.solvated_top.resolve()),
        },
    }

    P.json.write_text(json.dumps(config, indent=4))
    print(f'[setup] indices/metadata -> {P.json}')

    # Build the campaign-A seekr2 model + anchor tree under root/.
    # Uses complex-equil.pdb as the initial coordinates for all anchors;
    # hidr_smd will replace each anchor's coordinates with its SMD-deposited
    # pose. Other campaigns (root_orient/, root_side/, root_both/) are built
    # by hidr_smd from the post-flip equilibrated structures.
    print(f'[setup] building seekr2 model + anchor tree under {P.root} ...')
    prepare_root(
        prmtop_path=P.solvated_top.resolve(),
        pdb_path=P.equil_pdb.resolve(),
        root_dir=P.root.resolve(),
        ligand_indices=indices['ligand_indices'],
        receptor_indices=indices['receptor_indices'],
        force_overwrite=True,
    )
    print(f'[setup] seekr2 model -> {root_xml}')
