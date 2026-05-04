"""Stage: solvate the vacuum complex (truncated octahedron + neutralizing ions)."""

import json
import subprocess
import textwrap
from pathlib import Path

from ._config import C, paths


def _load_manifest(P):
    mfile = P.work / 'param_manifest.json'
    if mfile.exists():
        return json.loads(mfile.read_text())

    # Manifest deleted but param/ outputs still present: reconstruct.
    param_dir = P.work / 'param'
    candidates = {
        'host_mol2':   param_dir / 'host_gaff2' / 'host.gaff2.mol2',
        'host_frcmod': param_dir / 'host_gaff2' / 'host.frcmod',
        'guest_mol2':  param_dir / 'guest_gaff2' / 'guest.gaff2.mol2',
        'guest_frcmod':param_dir / 'guest_gaff2' / 'guest.frcmod',
    }
    if all(p.exists() for p in candidates.values()):
        manifest = {k: str(v) for k, v in candidates.items()}
        mfile.write_text(json.dumps(manifest, indent=2))
        print(f'[solvate] param_manifest.json missing; reconstructed from {param_dir}')
        return manifest

    # Last-ditch: user-specified paths in config.
    cfg_paths = {'host_mol2': C.host_mol2, 'host_frcmod': C.host_frcmod,
                 'guest_mol2': C.guest_mol2, 'guest_frcmod': C.guest_frcmod}
    if all(cfg_paths.values()):
        return cfg_paths
    raise RuntimeError(
        'param_manifest.json missing and parameter files not found in '
        f'{param_dir} or config. Run --stage param first.')


def stage_solvate(args):
    import parmed as pmd

    P = paths()
    if P.solvated_top.exists():
        print('[solvate] already done — skipping')
        return

    P.work.mkdir(exist_ok=True)
    assert (P.complex_dir / 'vac.prmtop').exists(), 'Run param stage first'

    manifest = _load_manifest(P)

    # Pass 1 — solvate only (no ions) just to learn the water count.
    # We can't carry box info across tleap invocations cleanly (loadpdb
    # drops CRYST1), so pass 2 re-runs solvateOct from vac.pdb. solvateOct
    # is deterministic for the same input + buffer, so the water count
    # matches and box info is preserved within one tleap call.
    pre_top = P.work / 'solvated_pre.prmtop'
    pre_rst = P.work / 'solvated_pre.rst7'
    tleap_pass1 = textwrap.dedent(f"""\
        source leaprc.gaff2
        source leaprc.lipid21
        source leaprc.water.tip3p
        set default PBRadii mbondi2
        loadamberparams {Path(manifest['host_frcmod']).resolve()}
        {C.host_resname} = loadmol2 {Path(manifest['host_mol2']).resolve()}
        loadamberparams {Path(manifest['guest_frcmod']).resolve()}
        {C.guest_resname} = loadmol2 {Path(manifest['guest_mol2']).resolve()}
        model = loadpdb {(P.complex_dir / 'vac.pdb').resolve()}
        solvateOct model TIP3PBOX {C.box_buffer_ang}
        saveamberparm model {pre_top.name} {pre_rst.name}
        quit
    """)
    inp1 = P.work / 'solvate.tleap.in'
    inp1.write_text(tleap_pass1)
    result = subprocess.run(['tleap', '-f', inp1.name],
                            cwd=str(P.work), capture_output=True, text=True)
    if not pre_top.exists():
        print(result.stdout[-3000:])
        raise RuntimeError('tleap solvation pass 1 failed')

    pre = pmd.load_file(str(pre_top))
    n_waters = sum(1 for r in pre.residues if r.name in ('WAT', 'HOH', 'TIP3'))
    n_ions = max(1, int(C.ion_conc_m / 55.5 * n_waters))
    print(f'[solvate] solvated water count: {n_waters}')
    print(f'[solvate] adding {n_ions} Na+ / {n_ions} Cl- '
          f'(target {C.ion_conc_m} M)')

    # Pass 2 — single shot: solvate + neutralize + add ions. addionsrand
    # *replaces* random waters, so we still end up with the right total
    # count and a box-flagged prmtop.
    tleap_pass2 = textwrap.dedent(f"""\
        source leaprc.gaff2
        source leaprc.lipid21
        source leaprc.water.tip3p
        set default PBRadii mbondi2
        loadamberparams {Path(manifest['host_frcmod']).resolve()}
        {C.host_resname} = loadmol2 {Path(manifest['host_mol2']).resolve()}
        loadamberparams {Path(manifest['guest_frcmod']).resolve()}
        {C.guest_resname} = loadmol2 {Path(manifest['guest_mol2']).resolve()}
        model = loadpdb {(P.complex_dir / 'vac.pdb').resolve()}
        solvateOct model TIP3PBOX {C.box_buffer_ang}
        addionsrand model Na+ 0
        addionsrand model Cl- 0
        addionsrand model Na+ {n_ions}
        addionsrand model Cl- {n_ions}
        savepdb model solvated.pdb
        saveamberparm model solvated.prmtop solvated.rst7
        quit
    """)
    inp2 = P.work / 'solvate2.tleap.in'
    inp2.write_text(tleap_pass2)
    result = subprocess.run(['tleap', '-f', inp2.name],
                            cwd=str(P.work), capture_output=True, text=True)
    if not P.solvated_top.exists():
        print(result.stdout[-3000:])
        raise RuntimeError('tleap solvation pass 2 (ion addition) failed')

    for f in (pre_top, pre_rst):
        f.unlink(missing_ok=True)
    print(f'[solvate] done — {P.solvated_top}')

    # Hydrogen Mass Repartitioning (HMR). Redistribute mass from heavy atoms
    # bonded to H (typically C, N) into the H atoms so H mass goes from
    # 1.008 → ~3.024 amu, allowing the integrator to advance at 4 fs vs the
    # standard 2 fs. Total system mass conserved. Required: HBonds + rigid
    # water constraints in subsequent dynamics (already used everywhere).
    if getattr(C, 'hmr_enabled', False):
        import parmed as pmd
        from parmed.tools import HMassRepartition
        target_h_mass = float(getattr(C, 'hmr_hydrogen_mass', 3.024))
        print(f'[solvate] HMR enabled — repartitioning H mass to {target_h_mass} amu')
        p = pmd.load_file(str(P.solvated_top))
        m_before = sum(a.mass for a in p.atoms)
        n_h = sum(1 for a in p.atoms if a.atomic_number == 1)
        HMassRepartition(p, target_h_mass).execute()
        m_after = sum(a.mass for a in p.atoms)
        h_masses = [a.mass for a in p.atoms if a.atomic_number == 1]
        print(f'[solvate]   {n_h} H atoms; mean H mass: '
              f'{sum(h_masses)/len(h_masses):.4f} amu (target {target_h_mass})')
        print(f'[solvate]   total mass: {m_before:.3f} → {m_after:.3f} amu '
              f'(Δ = {m_after - m_before:+.6f}, should be ~0)')
        if abs(m_after - m_before) > 1e-3:
            raise RuntimeError(
                f'[solvate] HMR did not conserve total mass '
                f'(Δ = {m_after - m_before:+.6f} amu)')
        p.save(str(P.solvated_top), overwrite=True)
        print(f'[solvate] HMR applied → {P.solvated_top}')
