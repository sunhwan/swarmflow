"""Stage: parameterize host + guest with GAFF2 and build vacuum complex."""

import json
import subprocess
import textwrap
from pathlib import Path

from ._config import C, paths


def _build_vacuum_complex(complex_pdb, host_mol2, host_frcmod,
                          guest_mol2, guest_frcmod, out_dir,
                          host_resname, guest_resname):
    tleap_in = textwrap.dedent(f"""\
        source leaprc.gaff2
        source leaprc.lipid21
        set default PBRadii mbondi2
        loadamberparams {Path(host_frcmod).resolve()}
        {host_resname} = loadmol2 {Path(host_mol2).resolve()}
        loadamberparams {Path(guest_frcmod).resolve()}
        {guest_resname} = loadmol2 {Path(guest_mol2).resolve()}
        model = loadpdb {Path(complex_pdb).resolve()}
        check model
        savepdb model vac.pdb
        saveamberparm model vac.prmtop vac.rst7
        saveamberparm {host_resname} receptor.prmtop receptor.rst7
        saveamberparm {guest_resname} ligand.prmtop ligand.rst7
        quit
    """)
    inp_file = Path(out_dir) / 'build.tleap.in'
    inp_file.write_text(tleap_in)
    result = subprocess.run(['tleap', '-f', 'build.tleap.in'],
                            cwd=str(out_dir), capture_output=True, text=True)
    if not (Path(out_dir) / 'vac.prmtop').exists():
        print(result.stdout[-3000:])
        raise RuntimeError('tleap failed to build vacuum complex')
    print(f'[param] vacuum complex: {out_dir}/vac.prmtop')


def stage_param(args):
    P = paths()
    if (P.complex_dir / 'vac.prmtop').exists():
        print('[param] already done — skipping')
        return

    P.work.mkdir(exist_ok=True)
    P.complex_dir.mkdir(parents=True, exist_ok=True)
    param_dir = P.work / 'param'

    host_mol2,   host_frcmod  = C.host_mol2,   C.host_frcmod
    guest_mol2,  guest_frcmod = C.guest_mol2,  C.guest_frcmod

    # ── Host (BCD) ───────────────────────────────────────────────────────────
    if host_mol2 is None:
        from rdkit.Chem import AllChem as Chem
        from .fragmenter import CyclodextrinFragmenter

        host_gaff_dir = param_dir / 'host_gaff2'
        host_gaff_dir.mkdir(parents=True, exist_ok=True)

        host_input = Path(C.host_sdf)
        assert host_input.exists(), f'host_sdf not found: {host_input}'
        mol = Chem.MolFromMolFile(str(host_input), removeHs=False, sanitize=False)

        frag = CyclodextrinFragmenter(mol)
        frag.parametrize(
            output_mol2=str(host_gaff_dir / 'host.gaff2.mol2'),
            output_frcmod=str(host_gaff_dir / 'host.frcmod'),
            residue_name=C.host_resname,
            atom_type='gaff2', charge_method='bcc',
            work_dir=str(host_gaff_dir),
        )
        host_mol2   = str(host_gaff_dir / 'host.gaff2.mol2')
        host_frcmod = str(host_gaff_dir / 'host.frcmod')
        print(f'[param] host parameterized: {host_mol2}')

    # ── Guest (cholesterol) ──────────────────────────────────────────────────
    if guest_mol2 is None:
        from rdkit.Chem import AllChem as Chem

        guest_gaff_dir = param_dir / 'guest_gaff2'
        guest_gaff_dir.mkdir(parents=True, exist_ok=True)

        guest_input = Path(C.guest_sdf)
        assert guest_input.exists(), f'guest_sdf not found: {guest_input}'
        mol = Chem.MolFromMolFile(str(guest_input), removeHs=False)
        charge = Chem.GetFormalCharge(mol) if mol else 0

        cmd = ['antechamber', '-fi', 'sdf', '-fo', 'mol2',
               '-i', str(guest_input.resolve()),
               '-o', 'guest.gaff2.mol2',
               '-c', 'bcc', '-s', '2', '-at', 'gaff2',
               '-rn', C.guest_resname]
        if charge != 0:
            cmd += ['-nc', str(charge)]
        subprocess.run(cmd, cwd=str(guest_gaff_dir), check=True)
        subprocess.run(['parmchk2', '-i', 'guest.gaff2.mol2', '-f', 'mol2',
                        '-o', 'guest.frcmod', '-s', 'gaff2'],
                       cwd=str(guest_gaff_dir), check=True)
        guest_mol2   = str(guest_gaff_dir / 'guest.gaff2.mol2')
        guest_frcmod = str(guest_gaff_dir / 'guest.frcmod')
        print(f'[param] guest parameterized: {guest_mol2}')

    _build_vacuum_complex(C.input_pdb, host_mol2, host_frcmod,
                          guest_mol2, guest_frcmod, P.complex_dir,
                          C.host_resname, C.guest_resname)

    manifest = {'host_mol2': host_mol2, 'host_frcmod': host_frcmod,
                'guest_mol2': guest_mol2, 'guest_frcmod': guest_frcmod}
    (P.work / 'param_manifest.json').write_text(json.dumps(manifest, indent=2))
    print('[param] done')
