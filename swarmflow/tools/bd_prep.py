"""
Generate Browndye2 input PQRs (receptor.pqr + ligand.pqr) for the BD stage.

Reads `work_<name>/solvated.prmtop` + `work_<name>/solvated.pdb` (produced
by the `solvate` stage, or by a tutorial-style drop-in of pre-built
parameters), runs cpptraj to materialize a snapshot inpcrd, then
`ambpdb -pqr` to produce a PQR with parm7-derived charges, and finally
splits it by residue name into `receptor.pqr` (host atoms) +
`ligand.pqr` (guest atoms) in the project directory.

The output filenames match the defaults referenced from `config.yml`:

    bd_receptor_pqr: receptor.pqr
    bd_ligand_pqr:   ligand.pqr

so `setup` will pick them up automatically (when `bd_enabled: true`).

Run after `solvate` and before `setup` — or any time after the prmtop +
pdb exist on disk, since BD only needs the charges + a starting snapshot
of the host and ligand.
"""

import shutil
import subprocess
import sys
from pathlib import Path

from .._config import C, paths


def _ensure_tool(name: str) -> str:
    path = shutil.which(name)
    if not path:
        sys.exit(f'[bd_prep] {name!r} not found on PATH — activate the '
                 'conda env that provides AmberTools (cpptraj + ambpdb)')
    return path


def _filter_pqr(src: Path, dst: Path, resname: str) -> int:
    """Copy ATOM/HETATM lines whose residue field equals `resname`. Returns count."""
    n = 0
    with src.open() as fin, dst.open('w') as fout:
        for line in fin:
            if not line.startswith(('ATOM', 'HETATM')):
                continue
            # PDB residue name is columns 18-20 (0-indexed 17:20). The upstream
            # tutorial used `resname in line` which is loose — column-based is
            # safer for short resnames that could collide with atom names.
            if line[17:20].strip() == resname:
                fout.write(line)
                n += 1
    return n


def cmd_bd_prep(args):
    P = paths()
    project_dir = Path('.').resolve()

    if not P.solvated_top.exists() or not P.solvated_pdb.exists():
        sys.exit(f'[bd_prep] {P.solvated_top} and/or {P.solvated_pdb} not '
                 f'found — run `swarmflow solvate` (or the tutorial '
                 f'prep_from_tutorial.py) first')

    receptor_out = project_dir / 'receptor.pqr'
    ligand_out   = project_dir / 'ligand.pqr'
    if receptor_out.exists() and ligand_out.exists() \
            and not getattr(args, 'force_overwrite', False):
        print(f'[bd_prep] {receptor_out.name} and {ligand_out.name} already '
              f'exist in {project_dir} — use --force-overwrite to regenerate')
        return

    _ensure_tool('cpptraj')
    _ensure_tool('ambpdb')

    # Stage intermediates inside work_<name>/ so the project dir stays clean
    # in case of crashes mid-run.
    inpcrd     = P.work / '_bd_prep_snapshot.inpcrd'
    cpptraj_in = P.work / '_bd_prep_snapshot.cpptraj'
    full_pqr   = P.work / '_bd_prep_full.pqr'

    cpptraj_in.write_text(
        f'parm {P.solvated_top.resolve()}\n'
        f'trajin {P.solvated_pdb.resolve()}\n'
        f'trajout {inpcrd.resolve()} restart\n'
        f'run\n'
    )
    try:
        print(f'[bd_prep] $ cpptraj -i {cpptraj_in.name}')
        subprocess.run(['cpptraj', '-i', str(cpptraj_in)], check=True)
    finally:
        cpptraj_in.unlink(missing_ok=True)

    print(f'[bd_prep] $ ambpdb -p {P.solvated_top.name} '
          f'-c {inpcrd.name} -pqr > {full_pqr.name}')
    with full_pqr.open('w') as fout:
        subprocess.run(
            ['ambpdb', '-p', str(P.solvated_top),
                       '-c', str(inpcrd), '-pqr'],
            stdout=fout, check=True,
        )
    inpcrd.unlink(missing_ok=True)

    n_recep = _filter_pqr(full_pqr, receptor_out, C.host_resname)
    n_lig   = _filter_pqr(full_pqr, ligand_out,   C.guest_resname)
    full_pqr.unlink(missing_ok=True)

    if n_recep == 0:
        receptor_out.unlink(missing_ok=True)
        sys.exit(f'[bd_prep] receptor.pqr empty — host_resname '
                 f'{C.host_resname!r} matched 0 atoms')
    if n_lig == 0:
        ligand_out.unlink(missing_ok=True)
        sys.exit(f'[bd_prep] ligand.pqr empty — guest_resname '
                 f'{C.guest_resname!r} matched 0 atoms')

    print(f'[bd_prep] {receptor_out.name}: {n_recep} atoms ({C.host_resname})')
    print(f'[bd_prep] {ligand_out.name}: {n_lig} atoms ({C.guest_resname})')
    print(f'[bd_prep] next: ensure bd_enabled=true + bd_receptor_pqr/'
          f'bd_ligand_pqr point at these files in config.yml, then run '
          f'`swarmflow setup` (which auto-builds root/b_surface/) and '
          f'`swarmflow bd`')
