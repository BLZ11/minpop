"""Fast regression tests (about a minute on one core). Run: pytest -q"""
import os
import subprocess
import sys
import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
XYZ = os.path.join(HERE, "..", "examples", "xyz")
PKG = os.path.join(HERE, "..", "minpop")
O_CHARGE_H2O_631G = -0.611678   # ROHF/6-31G -> STO-3G MBS, Gaussian-validated path


def run(script, *args):
    cmd = [sys.executable, os.path.join(PKG, script), *args]
    out = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    assert out.returncode == 0, out.stdout[-2000:] + out.stderr[-2000:]
    return out.stdout


def charges(text):
    lines = text.splitlines()
    i = [k for k, l in enumerate(lines) if "MBS Mulliken charges and spin" in l][0]
    rows = []
    for l in lines[i + 2:]:
        if not l.strip() or l.startswith(" Sum"):
            break
        rows.append([float(x) for x in l.split()[2:4]])
    return np.array(rows)


def test_rohf_h2o_sto3g_reference():
    """Gaussian-MBS convention path: values fixed by the validated STO-3G code."""
    q = charges(run("minpop_rohf.py", "-xyz", f"{XYZ}/h2o.xyz", "-basis", "6-31G"))
    assert abs(q[0, 0] - O_CHARGE_H2O_631G) < 1e-4   # value from the validated code
    assert abs(q.sum(0)[0]) < 1e-5                     # charges sum to zero


def test_uhf_ch2_triplet_spin():
    q = charges(run("minpop_uhf.py", "-xyz", f"{XYZ}/ch2.xyz", "-mult", "3",
                    "-basis", "6-31G"))
    assert abs(q[:, 1].sum() - 2.0) < 1e-5


def test_anorcc_truncations_shapes():
    sys.path.insert(0, os.path.join(PKG, "basis_sets"))
    from anorcc_basis_pyscf import ANO_RCC_MB, ANO_RCC_VDZP
    assert [(s[0], len(s[1]) - 1) for s in ANO_RCC_MB["Pt"]] == [(0, 6), (1, 5), (2, 3), (3, 1)]
    assert [(s[0], len(s[1]) - 1) for s in ANO_RCC_VDZP["H"]] == [(0, 2), (1, 1)]
    assert [(s[0], len(s[1]) - 1) for s in ANO_RCC_MB["Fe"]] == [(0, 4), (1, 3), (2, 1)]


def test_ano_rcc_mb_projection_h2o():
    out = run("minpop_rohf.py", "-xyz", f"{XYZ}/h2o.xyz", "-basis", "ano-rcc-vdzp",
              "-minimal-basis", "ano-rcc-mb")
    assert "ANO-RCC-MB" in out
    q = charges(out)
    assert abs(q.sum(0)[0]) < 1e-5


def test_bond_symmetry_selftest():
    sys.path.insert(0, PKG)
    import bond_symmetry
    from pyscf import gto, scf
    mol = gto.M(atom="C 0 0 0.667; C 0 0 -0.667; H 0 0.923 1.238; H 0 -0.923 1.238; "
                     "H 0 0.923 -1.238; H 0 -0.923 -1.238", basis="sto-3g", verbose=0)
    mf = scf.RHF(mol).run()
    comps = bond_symmetry.bond_components(mol, mf.make_rdm1())
    r = comps[(0, 1)]
    assert abs(r["sigma"] + r["pi"] + r["delta"] - r["total"]) < 1e-12
    assert r["sigma"] > 0.35 and r["pi"] > 0.15
    cm = bond_symmetry.class_matrices(mol, mf.make_rdm1())
    off = ~np.eye(mol.natm, dtype=bool)
    tot = sum(cm["atom"][c] for c in cm["classes"])
    assert np.abs((tot - cm["atom_total"])[off]).max() < 1e-12


def test_bond_symmetry_section_in_report():
    out = run("minpop_uhf.py", "-xyz", f"{XYZ}/ch2.xyz", "-mult", "3", "-basis", "6-31G",
              "-symmetry")
    assert "MBS σ Condensed to atoms (all electrons)" in out
    assert "MBS π Atomic-Atomic Spin Densities" in out


@pytest.mark.skipif(__import__("importlib").util.find_spec("gvbpp") is None,
                    reason="gvbpp not installed")
def test_gvbpp_h2():
    out = run("minpop_gvbpp.py", "-xyz", f"{XYZ}/h2.xyz", "-npairs", "1", "-basis", "cc-pvdz")
    assert "GVB-PP pairs:" in out
    assert "2 of 2 natural orbitals projected" in out
