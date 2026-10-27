# MIT License
#
# Copyright (c) 2026, Barbaro Zulueta
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""
bond_symmetry: σ / π / δ / φ resolution of MinPop (MBS Mulliken) bond populations.

Works for any angular momentum (s through f and beyond), any element, and any
molecular orientation. Used by the MinPop drivers through -symmetry and
-symmetry-shells; also usable on its own.

What it computes
----------------
For an atom pair A-B the Mulliken bond population is

    W_AB = sum_{mu in A, nu in B} P_{mu nu} S_{mu nu}

(the off-diagonal "Condensed to atoms" element). The AO blocks of P and S are
rotated into the local diatomic frame whose z' axis lies along the bond; real
spherical harmonics transform within each shell by a Wigner D matrix
(pyscf.symm.Dmatrix). In that frame the two-centre overlap is diagonal in the
magnetic quantum number m: an AO with m on A overlaps only an AO with the same
m on B. Each term P'S' therefore belongs to exactly one symmetry class,

    σ  |m| = 0      π  |m| = 1      δ  |m| = 2      φ  |m| = 3,

and the classes sum to W_AB exactly. No analytic overlap formulas, no fitted
exponents, no renormalization. The same split applies to the spin density, in
which case a negative entry means the unpaired electrons occupy the
antibonding combination of that symmetry.

The diatomic selection rule is asserted on every pair: if any rotated
cross-overlap between AOs of different m exceeds `check`, the rotation
convention and the AO ordering disagree and a RuntimeError is raised rather
than a silent misclassification.

Same-atom (on-site) populations have no bond axis and no symmetry class; the
class matrices carry only the inter-atomic terms. class_matrices() also
returns the full matrices (with on-site terms) so the sum rule can be checked.

API
---
    bond_components(mol_min, dm, S=None, pairs=None)
        -> {(i, j): {'total', 'sigma', 'pi', 'delta', 'phi',
                     'shells': {(shell_i, shell_j, class): value}, 'direction'}}
    class_matrices(mol_min, dm, S=None)
        -> {'classes', 'shell_labels', 'shell': {class: matrix},
            'atom': {class: matrix}, 'shell_total', 'atom_total'}
    format_table(mol_min, comps)   plain-text summary (σ π δ φ columns)
    SYMBOL                         {'sigma': 'σ', 'pi': 'π', 'delta': 'δ', 'phi': 'φ'}

`dm` is any density matrix in the minimal basis in PySCF AO order: total,
alpha, beta, spin, or a single pair's density (as minpop_gvbpp.py does).

Provenance
----------
This module replaces the direction-cosine code of BEBOP's projection.py,
which handled 2p/3p only, assumed Gaussian's STO-3G AO layout, and used
analytic Slater-type-orbital overlaps. Its π projector, built from
sin = -sign(c) sqrt(1 - c²) per Cartesian component, is exact only when the
bond lies in a coordinate plane (for a C-C bond along (1,1,1) it reports
π = 0.37 where the exact value is 0.0002), and its σ term carries a sign
error on bonding densities. The rotation here is exact for any orientation.
"""

import numpy as np
from pyscf.symm import Dmatrix

CLASS_NAMES = {0: 'sigma', 1: 'pi', 2: 'delta', 3: 'phi', 4: 'gamma'}
SYMBOL = {'sigma': '\u03c3', 'pi': '\u03c0', 'delta': '\u03b4', 'phi': '\u03c6', 'gamma': '\u03b3'}
_Z = np.array([0.0, 0.0, 1.0])


def _ao_shell_info(mol):
    """Per-AO (atom, shell label, l, m) in PySCF spherical order, plus
    per-shell (start index, n_contractions, l)."""
    labels = mol.ao_labels(fmt=None)
    ao_loc = mol.ao_loc_nr()
    atom, l_of, m_of, shell_of = [], [], [], []
    shells = []
    for ish in range(mol.nbas):
        l = mol.bas_angular(ish)
        nctr = mol.bas_nctr(ish)
        ia = mol.bas_atom(ish)
        if l == 0:
            ms = [0]
        elif l == 1:
            ms = [1, -1, 0]          # PySCF p order: x, y, z -> m = +1, -1, 0
        else:
            ms = list(range(-l, l + 1))
        shells.append((ao_loc[ish], nctr, l))
        for k in range(nctr):
            p0 = ao_loc[ish] + k * (2 * l + 1)
            shell_label = _shell_label(labels[p0][2].strip())   # '5dxy' -> '5d'
            for m in ms:
                atom.append(ia); l_of.append(l); m_of.append(m); shell_of.append(shell_label)
    return (np.array(atom), np.array(l_of), np.array(m_of), np.array(shell_of), shells)


def _shell_label(ao_name):
    """'5dxy' -> '5d', '2px' -> '2p', '4f-3' -> '4f', '1s' -> '1s'."""
    i = 0
    while i < len(ao_name) and ao_name[i].isdigit():
        i += 1
    return ao_name[:i + 1]


def _rotation_to_bond(mol, shells, nao, direction):
    """Block-diagonal AO rotation U such that U^T M U expresses the AO blocks
    in the frame with z' along `direction`."""
    c = np.asarray(direction, dtype=float)
    c = c / np.linalg.norm(c)
    if np.allclose(c, _Z):
        return np.eye(nao)
    # Euler angles of the rotation carrying z onto c, in the convention that
    # makes U^T S U block-diagonal in m for PySCF's real harmonics. Verified
    # exact (|S'| < 1e-15 for m mismatch) on random and axial directions;
    # pyscf.symm.Dmatrix.get_euler_angles is not reliable here.
    alpha = np.arctan2(c[1], c[0])
    beta = np.arccos(np.clip(c[2], -1.0, 1.0))
    gamma = 0.0
    U = np.zeros((nao, nao))
    cache = {}
    for p0, nctr, l in shells:
        if l not in cache:
            cache[l] = Dmatrix.Dmatrix(l, alpha, beta, gamma, reorder_p=True)
        D = cache[l]
        for k in range(nctr):
            a = p0 + k * (2 * l + 1)
            U[a:a + 2 * l + 1, a:a + 2 * l + 1] = D
    return U


def bond_components(mol, dm, S=None, pairs=None, lmax_class=4, check=1e-8):
    """
    Decompose Mulliken bond populations into sigma/pi/delta/phi classes.

    Parameters
    ----------
    mol : pyscf.gto.Mole
        Minimal-basis molecule (spherical functions) the density is expressed in.
    dm : ndarray (nao, nao)
        Density matrix in PySCF AO order (total, alpha, beta or spin).
    S : ndarray, optional
        Overlap matrix; computed from mol if omitted.
    pairs : iterable of (i, j), optional
        Atom pairs to decompose; default all i < j.
    lmax_class : int
        Highest |m| class to report.
    check : float
        Tolerance for the diatomic selection rule: the rotated cross-overlap
        between AOs with different m must vanish. A violation means the
        rotation convention and the AO ordering disagree; raise rather than
        silently misclassify.

    Returns
    -------
    dict keyed by (i, j) with entries
        'total'  : W_ij = sum P S over the pair (Mulliken overlap population)
        'sigma', 'pi', 'delta', 'phi' ... : class contributions
        'shells' : {(shell_i, shell_j, class): contribution}
        'direction' : unit vector i -> j
    """
    if S is None:
        S = mol.intor('int1e_ovlp')
    nao = mol.nao_nr()
    atom, l_of, m_of, shell_of, shells = _ao_shell_info(mol)
    if pairs is None:
        pairs = [(i, j) for i in range(mol.natm) for j in range(i + 1, mol.natm)]
    out = {}
    for i, j in pairs:
        c = mol.atom_coord(j) - mol.atom_coord(i)
        U = _rotation_to_bond(mol, shells, nao, c)
        ia = np.where(atom == i)[0]
        ja = np.where(atom == j)[0]
        Ui = U[np.ix_(ia, ia)]
        Uj = U[np.ix_(ja, ja)]
        Pp = Ui.T @ dm[np.ix_(ia, ja)] @ Uj
        Sp = Ui.T @ S[np.ix_(ia, ja)] @ Uj
        mi = m_of[ia][:, None]
        mj = m_of[ja][None, :]
        same_m = (mi == mj)
        worst = np.abs(Sp[~same_m]).max() if (~same_m).any() else 0.0
        if worst > check:
            raise RuntimeError(f"diatomic selection rule violated for pair "
                               f"({i},{j}): |S'| = {worst:.2e} between AOs of "
                               f"different m; rotation convention mismatch")
        W = Pp * Sp
        rec = {'total': float(W.sum()), 'direction': c / np.linalg.norm(c), 'shells': {}}
        for k in range(lmax_class + 1):
            mask = same_m & (np.abs(mi) == k)
            rec[CLASS_NAMES[k]] = float(W[mask].sum())
        for a_idx, a in enumerate(ia):
            for b_idx, b in enumerate(ja):
                if not same_m[a_idx, b_idx] or abs(W[a_idx, b_idx]) < 1e-12:
                    continue
                key = (shell_of[a], shell_of[b], CLASS_NAMES[abs(int(m_of[a]))])
                rec['shells'][key] = rec['shells'].get(key, 0.0) + float(W[a_idx, b_idx])
        out[(i, j)] = rec
    return out


def format_table(mol, comps, threshold=1e-3, shells=False):
    """Plain-text summary for pairs with |total| >= threshold, σ π δ φ columns."""
    classes = ['sigma', 'pi', 'delta', 'phi']
    lines = [" MBS bond populations by symmetry (local diatomic frame):",
             f"  {'pair':14s}{'R (A)':>8s}{'total':>10s}"
             + "".join(f"{SYMBOL[c]:>10s}" for c in classes)]
    for (i, j), r in sorted(comps.items()):
        if abs(r['total']) < threshold:
            continue
        R = np.linalg.norm(mol.atom_coord(j) - mol.atom_coord(i)) * 0.52917721092
        name = f"{mol.atom_symbol(i)}{i + 1}-{mol.atom_symbol(j)}{j + 1}"
        lines.append(f"  {name:14s}{R:8.3f}{r['total']:10.5f}"
                     + "".join(f"{r.get(c, 0.0):10.5f}" for c in classes))
        if shells:
            for (si, sj, cl), v in sorted(r['shells'].items(), key=lambda kv: -abs(kv[1])):
                if abs(v) >= threshold:
                    lines.append(f"      {si:>4s}-{sj:<4s} {SYMBOL[cl]:3s} {v:10.5f}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Class-resolved matrices condensed to shells and to atoms
# ---------------------------------------------------------------------------

def class_matrices(mol, dm, S=None, lmax_class=3):
    """
    Symmetry-class bond population matrices condensed to shells and atoms.

    Returns
    -------
    dict with
        'classes'      : list of class names reported
        'shell_labels' : ['Pt1 1s', 'Pt1 2s', ..., 'O2 2p'] (one per (atom, shell))
        'shell'        : {class: (nsh, nsh) symmetric matrix of bond terms}
        'atom'         : {class: (natm, natm) symmetric matrix of bond terms}
        'shell_total'  : (nsh, nsh) full Mulliken population matrix condensed
                         to shells, including on-site (same-atom) entries
        'atom_total'   : (natm, natm) full matrix, the usual condensed-to-atoms

    On-site (same-atom) populations have no bond axis and therefore no
    symmetry class; they appear in the *_total matrices only. Off the
    diagonal, the class matrices sum to the total matrices exactly.
    """
    if S is None:
        S = mol.intor('int1e_ovlp')
    atom, l_of, m_of, shell_of, shells = _ao_shell_info(mol)
    # (atom, shell) index per AO
    keys = []
    for a, s in zip(atom, shell_of):
        k = (int(a), s)
        if k not in keys:
            keys.append(k)
    sh_idx = np.array([keys.index((int(a), s)) for a, s in zip(atom, shell_of)])
    nsh, natm = len(keys), mol.natm
    labels = [f"{mol.atom_symbol(a)}{a + 1} {s}" for a, s in keys]
    classes = [CLASS_NAMES[k] for k in range(lmax_class + 1)]

    PS = dm * S
    shell_total = np.zeros((nsh, nsh))
    np.add.at(shell_total, (sh_idx[:, None].repeat(len(sh_idx), 1), sh_idx[None, :].repeat(len(sh_idx), 0)), PS)
    atom_total = np.zeros((natm, natm))
    np.add.at(atom_total, (atom[:, None].repeat(len(atom), 1), atom[None, :].repeat(len(atom), 0)), PS)

    shell_cls = {c: np.zeros((nsh, nsh)) for c in classes}
    atom_cls = {c: np.zeros((natm, natm)) for c in classes}
    comps = bond_components(mol, dm, S, lmax_class=lmax_class)
    for (i, j), rec in comps.items():
        for c in classes:
            atom_cls[c][i, j] = atom_cls[c][j, i] = rec[c]
        for (si, sj, c), v in rec['shells'].items():
            a = keys.index((i, si)); b = keys.index((j, sj))
            shell_cls[c][a, b] += v
            shell_cls[c][b, a] += v
    return {'classes': classes, 'shell_labels': labels, 'shell': shell_cls,
            'atom': atom_cls, 'shell_total': shell_total, 'atom_total': atom_total}


if __name__ == '__main__':
    # Self-test: classes sum to the Mulliken overlap population and the
    # decomposition is orientation invariant.
    from pyscf import gto, scf
    base = 'C 0 0 0.667; C 0 0 -0.667; H 0 0.923 1.238; H 0 -0.923 1.238; H 0 0.923 -1.238; H 0 -0.923 -1.238'
    ref = None
    for rot in [np.eye(3), np.array([[0, 0, 1.], [1, 0, 0], [0, 1, 0]]),
                np.linalg.qr(np.array([[1., 2, 3], [4, 5, 6.5], [7, 8.2, 10]]))[0]]:
        atoms = []
        for tok in base.split(';'):
            s, *xyz = tok.split()
            atoms.append([s, tuple(rot @ np.array(xyz, float))])
        mol = gto.M(atom=atoms, basis='sto-3g', verbose=0)
        mf = scf.RHF(mol).run()
        comps = bond_components(mol, mf.make_rdm1())
        PS = mf.make_rdm1() * mol.intor('int1e_ovlp')
        at = np.array([l[0] for l in mol.ao_labels(fmt=None)])
        W = PS[np.ix_(at == 0, at == 1)].sum()
        r = comps[(0, 1)]
        assert abs(r['sigma'] + r['pi'] + r['delta'] - r['total']) < 1e-12
        assert abs(r['total'] - W) < 1e-12
        if ref is None:
            ref = (r['sigma'], r['pi'])
        assert abs(r['sigma'] - ref[0]) < 1e-9 and abs(r['pi'] - ref[1]) < 1e-9, (r, ref)
    print(f"ethylene C=C: σ {ref[0]:.5f}, π {ref[1]:.5f}; invariant under 3 orientations: OK")
    cm = class_matrices(mol, mf.make_rdm1())
    off = ~np.eye(mol.natm, dtype=bool)
    tot = sum(cm['atom'][c] for c in cm['classes'])
    assert np.abs((tot - cm['atom_total'])[off]).max() < 1e-12
    offs = ~np.eye(len(cm['shell_labels']), dtype=bool)
    tots = sum(cm['shell'][c] for c in cm['classes'])
    same_atom = np.array([[a.split()[0] == b.split()[0] for b in cm['shell_labels']] for a in cm['shell_labels']])
    assert np.abs((tots - cm['shell_total'])[~same_atom]).max() < 1e-12
    print("class matrices sum to the condensed-to-atoms and condensed-to-shells matrices off-site: OK")
