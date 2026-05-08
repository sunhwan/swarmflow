"""
Shared config + path objects.

`C` is the runtime configuration. It is created empty at import time and
mutated in place by `load_config()` so that every stage module that does
`from ._config import C` sees the same up-to-date namespace.
"""

from pathlib import Path
from types import SimpleNamespace

import yaml


DEFAULTS = {
    # Inputs
    'name':          'BCD_cholesterol',
    'input_pdb':     'head.pdb',
    'host_sdf':      'BCD.sdf',
    'guest_sdf':     'chl.sdf',
    'host_resname':  'MOL',
    'guest_resname': 'LIG',

    # Pre-computed GAFF2 parameters; null = compute from sdf files
    'host_mol2':    None,
    'host_frcmod':  None,
    'guest_mol2':   None,
    'guest_frcmod': None,

    # Solvation
    'box_buffer_ang': 12.0,
    'ion_conc_m':      0.15,

    # Equilibration: 100 ps NVT (restrained) → 500 ps NPT (restrained) → 500 ps NPT (free)
    'equil_nvt_ps':              100,
    'equil_npt_restrained_ps':   500,
    'equil_npt_free_ps':         500,
    'equil_restraint_k':        10.0,
    'equil_report':           10000,
    'gpu_index':                '0',

    # MMVT
    'anchor_radii': [0.05, 0.15, 0.25, 0.35, 0.50, 0.60, 0.70,
                     0.80, 0.90, 1.10, 1.30, 1.50, 1.70, 1.90],
    'benchmark_steps_per_anchor':  50000,
    'production_ns_per_anchor':       1.0,    # ns of MD per (anchor, swarm K); steps derived from timestep
    'md_output_interval':           10000,
    'hmr_enabled':                 False,    # Hydrogen Mass Repartitioning. When True: H mass=3.024, timestep=4 fs
    'hmr_hydrogen_mass':           3.024,    # H mass (amu) when HMR active
    'temperature_K':               300.0,    # MD temperature in Kelvin (all stages: equil, HIDR, swarm, MMVT)

    # metaD HIDR (stage_hidr_metad) — knobs for seekrtools.hidr in metaD mode.
    # Direction-blind on |COM-COM|; for direction-aware HIDR use stage_hidr_smd.
    'metad_equilibration_ns':       0.05,    # equilibration before metaD bias deposition starts (ns)
    'metad_settling_ps':            50.0,    # post-arrival settling at each anchor (ps)
    'metad_sigma_nm':               0.05,    # Gaussian width along the spherical CV (nm)
    'metad_biasfactor':            10.0,     # well-tempered metaD bias factor (dimensionless)
    'metad_height_kjmol':           1.0,     # Gaussian height in kJ/mol

    # Browndye2 association-rate stage. Active only when bd_enabled=True
    # AND both PQR paths exist on disk. setup wires these into the seekr2
    # Model_input.browndye_settings_input; seekr2.prepare auto-builds
    # b_surface/ under root/. stage_bd then runs the b-surface trajectories
    # via seekr2.run.run(model, 'b_surface', ...).
    'bd_enabled':                       False,
    'bd_binary_directory':              '',     # path to browndye2 bin dir; '' = look on PATH
    'bd_receptor_pqr':                  None,   # path (rel to project dir) to receptor PQR; null disables BD
    'bd_ligand_pqr':                    None,   # path (rel to project dir) to ligand PQR; null disables BD
    'bd_apbs_grid_spacing':             0.5,    # APBS grid spacing in Å
    'bd_num_b_surface_trajectories':    10000,  # number of b-surface trajectories (paper used 110000)
    'bd_n_threads':                     1,      # threads for nam_simulation
    'bd_ions':                          [],     # list of dicts: {'radius':1.2, 'charge':1.0, 'conc':0.15}

    'bottleneck_threshold':            30,
    'adjust_imbalance_ratio':         5.0,
    'adjust_min_gap_nm':             0.05,

    # Run
    'cuda_gpus':   ['0', '1'],
    'mps_per_gpu': 1,

    # Swarm-stage relaxation after geometric transforms
    'swarm_relax_ps':         10.0,
    'swarm_relax_restraint_k': 10.0,    # kcal/mol/Å² on solute heavy atoms
    'swarm_minimize_iter':    2500,

    # Directional HIDR (hidr_alts) — SMD-style pull along signed host axis
    'directional_pull_velocity_nm_per_ns': 0.1,    # 10× slower than the old 1.0 default — gives waters time to repack during through-cavity transits
    'directional_k_dir':          100.0,           # kcal/mol/Å² coupling on signed projection (direction)
    'directional_k_rad':          500.0,           # kcal/mol/Å² coupling on |COM-COM| during hold (cell containment)
    'directional_restraint_k':     10.0,           # kcal/mol/Å² host heavy-atom restraint
    'directional_tolerance_nm':    0.005,          # converged when |s_actual − s_target| < this
    'directional_max_pull_time_ps': 1000.0,        # give up per anchor after this many ps
    'directional_initial_equil_ps': 500.0,         # extra equilibration at FIRST anchor (after geometric flip)
    'directional_dcd_interval_ps':   50.0,         # pull.dcd write interval — coarser = smaller DCD; finer = more frame-extraction candidates per anchor

    # SMARTS
    'host_o2_o6_smarts':    'O1CC([O:1])CCC1C([O:2])',
    'host_backbone_smarts': '[#6;r6]',
    'guest_axis_atoms': None,
}

STAGES = ['param', 'solvate', 'equil', 'check', 'setup', 'report',
          'adjust', 'hidr_smd', 'hidr_metad', 'bd', 'swarm', 'kinetics',
          'extract', 'analyze']


# Singleton, mutated in place by load_config(). Never reassign.
C: SimpleNamespace = SimpleNamespace()


def load_config(config_file: str, cli_overrides: dict) -> SimpleNamespace:
    cfg = dict(DEFAULTS)

    if config_file and Path(config_file).exists():
        with open(config_file) as f:
            file_cfg = yaml.safe_load(f) or {}
        cfg.update({k: v for k, v in file_cfg.items() if v is not None})
    elif config_file:
        print(f'[warn] config file {config_file!r} not found, using defaults')

    cfg.update({k: v for k, v in cli_overrides.items() if v is not None})

    cfg['work_dir'] = cfg.get('work_dir') or f"work_{cfg['name']}"
    if isinstance(cfg.get('cuda_gpus'), str):
        cfg['cuda_gpus'] = [s.strip() for s in cfg['cuda_gpus'].split(',') if s.strip()]

    # Derived: timestep depends on HMR. With HMR, H mass goes from 1.008 to
    # 3.024 amu (~3×) which lets the integrator stably advance at 4 fs vs the
    # standard 2 fs. All production stages (hidr_alts, swarm Phase 1, MMVT
    # Phase 2 via the model) read this single derived value.
    cfg['production_timestep_ps'] = 0.004 if cfg.get('hmr_enabled') else 0.002

    # Derived: steps from ns + timestep. config.yml only carries the
    # physics-meaningful production_ns_per_anchor; the integrator-step count
    # is computed here so toggling HMR doesn't silently halve simulation time.
    ns = float(cfg.get('production_ns_per_anchor', 1.0))
    dt_ps = cfg['production_timestep_ps']
    cfg['production_steps_per_anchor'] = int(round(ns * 1000.0 / dt_ps))

    C.__dict__.clear()
    C.__dict__.update(cfg)
    return C


def paths() -> SimpleNamespace:
    """Derived Path objects from the current C."""
    work = Path(C.work_dir)
    return SimpleNamespace(
        work          = work,
        complex_dir   = work / 'complex',
        solvated_top  = work / 'solvated.prmtop',
        solvated_rst7 = work / 'solvated.rst7',
        solvated_pdb  = work / 'solvated.pdb',
        equil_pdb     = work / 'complex-equil.pdb',
        equil_rst7    = work / 'complex-equil.rst7',
        json          = work / f'seekrflow_{C.name}.json',
        root          = work / 'root',
    )
