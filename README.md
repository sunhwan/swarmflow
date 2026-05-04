# swarmflow

Stage-based driver for [seekr2](https://github.com/seekrcentral/seekr2) milestoning calculations with a 4-member MMVT swarm. Wraps a full host–guest binding-kinetics pipeline — parameterization, solvation, equilibration, anchor setup, HIDR, MMVT production, and convergence diagnostics — behind a single `swarmflow <stage>` CLI.

The pipeline is system-agnostic at the engine level. System-specific things (input PDB/SDFs, host/guest SMARTS, anchor radii, force-field overrides) live in a per-project `config.yml`.

## Pipeline

| Stage | What it does |
|---|---|
| `param` | Host/guest GAFF2 parameterization (antechamber + cyclodextrin fragmenter); vacuum complex via tleap |
| `solvate` | Two-pass tleap (solvate → count waters → re-solvate with neutralizing ions); optional HMR |
| `equil` | Minimize + 3-phase equilibration (NVT-restrained → NPT-restrained → NPT-free) |
| `check` | Box-vs-Rmax minimum image distance check |
| `setup` | SMARTS atom-index discovery; builds `root/model.xml` + anchor tree via `seekr2.prepare.prepare()` |
| `report` | Per-anchor crossing counts (aggregates all swarm members) + bottleneck plot |
| `adjust` | Propose new anchor_radii by splitting bottleneck/imbalanced anchors |
| `hidr_smd` | Continuous-pull SMD HIDR for 4 oriented campaigns (A, side, orient, both) in parallel |
| `hidr_metad` | metaD HIDR via `seekrtools.hidr` — single campaign, direction-blind on \|COM-COM\| |
| `swarm` | 4-member MMVT swarm per anchor; Phase 1 (parallel minimize-only state generation) + Phase 2 (K-major MMVT production) |
| `kinetics` | `seekr2.analyze` + Jacobian-corrected PMF + ΔG_bind (rate-based & PMF-based) + 4-block convergence + per-anchor late-block drift + sliding-window k_off plot |
| `extract` | Dump anchor trajectory snapshots as PDB |
| `analyze` | Host-axis vs guest-axis orientation + min-distance from MMVT trajectories |

## Install

### conda (recommended, GPU host)

```bash
mamba create -n swarmflow -c conda-forge \
    python=3.10 openmm=8.1 seekr2_openmm_plugin ambertools parmed \
    rdkit networkx numpy scipy matplotlib-base pyyaml tqdm \
    openff-toolkit-base openff-units mdanalysis pymbar
mamba activate swarmflow

# seekr2 + seekrtools + paprika are pip installs from upstream:
pip install git+https://github.com/seekrcentral/seekr2
pip install git+https://github.com/seekrcentral/seekrtools
pip install git+https://github.com/GilsonLabUCSD/pAPRika

# swarmflow itself, editable:
pip install -e .
```

### Docker

```bash
docker build -t swarmflow .
docker run --rm --gpus all -v "$PWD":/work swarmflow kinetics
```

The image bakes in the conda env + seekr2/seekrtools/paprika; mount your project directory at `/work` (the container's WORKDIR). Requires the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) on the host.

## Usage

Scaffold a project with `swarmflow init`:

```bash
swarmflow init my-system
cd my-system
# drop input.pdb, host.sdf, guest.sdf into this dir
```

A **swarmflow project** is a directory containing `config.yml` plus inputs:

```
my-system/
├── config.yml          # all knobs (anchor_radii, GPU list, HIDR mode, HMR, …)
├── input.pdb           # bound-state structure (host + guest, no solvent)
├── host.sdf            # host topology (cyclodextrin, etc.)
└── guest.sdf           # guest topology
```

Run from the project directory:

```bash
swarmflow run --from param --to setup    # param→solvate→equil→check→setup
swarmflow hidr_smd                       # or: swarmflow hidr_metad
swarmflow swarm
swarmflow kinetics
swarmflow analyze
swarmflow status                         # which stages have produced output?
```

Stages are **idempotent** — each one short-circuits if its primary output already exists. Re-running the whole pipeline after a partial crash resumes from where it left off. To extend an existing swarm run, bump `production_ns_per_anchor` in `config.yml` and re-run `swarmflow swarm`; seekr2 resumes each (anchor, K) from its checkpoint.

Stage-specific options live on the stage's own subparser — try `swarmflow kinetics --help` to see only what's relevant. Global config keys can be overridden inline:

```bash
swarmflow swarm --gpu 0,1 --mps-per-gpu 2
swarmflow kinetics --num-error-samples 500 --skip-sliding-window
swarmflow extract --frames last --every 1
```

## Helpers

Inspection and cleanup subcommands, all system-agnostic (read `config.yml`):

```bash
swarmflow clean <mode> [-y]        # remove specific stage outputs to allow re-runs
swarmflow diagnose                 # per-anchor table of HIDR campaign quality
swarmflow merge_hidr [--by-anchor] # merge HIDR PDBs into multi-model views for PyMOL/VMD
swarmflow verify_bound             # geometry summary of the 4 equilibrated bound states
```

`swarmflow clean --help` lists the modes (`from-setup`, `hidr`, `hidr-alts`, `swarm`, `post-equil`, `post-solvate`, `views`, `all`). It prints what it would delete and asks for confirmation; `-y` skips the prompt.

## Architecture

- `swarmflow/cli.py` — argparse dispatcher with one subparser per stage plus `run` (range), `status`, `stages`, and `init`. `swarmflow/__main__.py` is the `python -m swarmflow` entry.
- `swarmflow/_config.py` — owns the `C` namespace (`SimpleNamespace` mutated in place by `load_config()`); derives `production_timestep_ps` and `production_steps_per_anchor` from physics-meaningful inputs.
- `swarmflow/_seekr_model.py` — constructs `seekr2.Model_input` directly and calls `seekr2.prepare.prepare()`. No `seekrflow` dependency.
- `swarmflow/<stage>.py` — one module per stage, exposing `def stage_<name>(args): ...`. Each stage reads `C` and operates on the `paths()` directory tree.
- `swarmflow/tools/` — helper subcommands (`clean`, `diagnose`, `merge_hidr`, `verify_bound`); each exposes `def cmd_<name>(args): ...`.
- `swarmflow/templates/config.yml.template` — annotated config emitted by `swarmflow init`.
- `swarmflow/fragmenter.py` — verbatim copy of `paprika.build.system.fragmenter` (bypasses paprika's heavy `__init__` imports).

State flows between stages **on disk** inside `work_<name>/` — no in-memory state survives between invocations.

## See also

- [seekr2 documentation](https://seekr2.readthedocs.io) — milestoning theory, Model_input schema
- [seekrtools](https://github.com/seekrcentral/seekrtools) — HIDR + analysis utilities
