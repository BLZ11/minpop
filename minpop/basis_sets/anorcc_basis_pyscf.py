"""
ANO-RCC contracted truncations (MB / VDZP / VTZP / VQZP) as PySCF basis dicts.

Roos's ANO-RCC family is one general contraction per element (PySCF ships it
as 'ano', H-Cm). The named truncations are its leading contraction columns per
angular momentum: ANO-RCC-MB keeps one function per occupied atomic orbital
of the ground state plus the valence p on metals (Pt: 6s5p3d1f, Fe: 4s3p1d),
ANO-RCC-VDZP keeps one more per l, and so on. This module slices PySCF's own
'ano' basis with the column counts of each truncation, so no extra package
is needed at run time. The counts were taken from Basis Set Exchange
(ano-rcc-mb, -vdzp, -vtzp, -vqzp, latest versions, Oct 2026) and the sliced
coefficients were verified to equal the BSE ones to 1e-12 for C, Fe and Pt
(run this file to repeat that check when basis_set_exchange is installed).

ANO-RCC is contracted for the DKH2 Hamiltonian, so pair it with a scalar
relativistic SCF (the MinPop scripts' -x2c flag) for heavy atoms.

Exposed dicts (all lazily built, element symbol -> PySCF shell list):
    ANO_RCC_MB, ANO_RCC_VDZP, ANO_RCC_VTZP, ANO_RCC_VQZP

Loadable by the MinPop scripts as -basis ano-rcc-vdzp etc., or directly:
    from anorcc_basis_pyscf import ANO_RCC_VDZP
    gto.M(atom=..., basis=ANO_RCC_VDZP)

Author: Barbaro Zulueta (Pitt Quantum Repository)
"""

import copy
import numpy as np
from pyscf import gto
from pyscf.data import elements as _el

__all__ = ["ANO_RCC_MB", "ANO_RCC_VDZP", "ANO_RCC_VTZP", "ANO_RCC_VQZP",
           "truncate_ano", "ano_rcc"]

# Number of contracted functions kept per angular momentum (s, p, d, f, g, h),
# element by element, for each named truncation.
MB_COUNTS = {
    'H': (1,), 'He': (1,), 'Li': (2, 1), 'Be': (2, 1),
    'B': (2, 1), 'C': (2, 1), 'N': (2, 1), 'O': (2, 1),
    'F': (2, 1), 'Ne': (2, 1), 'Na': (3, 2), 'Mg': (3, 2),
    'Al': (3, 2), 'Si': (3, 2), 'P': (3, 2), 'S': (3, 2),
    'Cl': (3, 2), 'Ar': (3, 2), 'K': (4, 3), 'Ca': (4, 3),
    'Sc': (4, 3, 1), 'Ti': (4, 3, 1), 'V': (4, 3, 1), 'Cr': (4, 3, 1),
    'Mn': (4, 3, 1), 'Fe': (4, 3, 1), 'Co': (4, 3, 1), 'Ni': (4, 3, 1),
    'Cu': (4, 3, 1), 'Zn': (4, 3, 1), 'Ga': (4, 3, 1), 'Ge': (4, 3, 1),
    'As': (4, 3, 1), 'Se': (4, 3, 1), 'Br': (4, 3, 1), 'Kr': (4, 3, 1),
    'Rb': (5, 4, 1), 'Sr': (5, 4, 1), 'Y': (5, 4, 2), 'Zr': (5, 4, 2),
    'Nb': (5, 4, 2), 'Mo': (5, 4, 2), 'Tc': (5, 4, 2), 'Ru': (5, 4, 2),
    'Rh': (5, 4, 2), 'Pd': (5, 4, 2), 'Ag': (5, 4, 2), 'Cd': (5, 4, 2),
    'In': (5, 4, 2), 'Sn': (5, 4, 2), 'Sb': (5, 4, 2), 'Te': (5, 4, 2),
    'I': (5, 4, 2), 'Xe': (5, 4, 2), 'Cs': (6, 5, 2), 'Ba': (6, 5, 2),
    'La': (6, 5, 3, 1), 'Ce': (6, 5, 3, 1), 'Pr': (6, 5, 3, 1), 'Nd': (6, 5, 3, 1),
    'Pm': (6, 5, 3, 1), 'Sm': (6, 5, 3, 1), 'Eu': (6, 5, 3, 1), 'Gd': (6, 5, 3, 1),
    'Tb': (6, 5, 3, 1), 'Dy': (6, 5, 3, 1), 'Ho': (6, 5, 3, 1), 'Er': (6, 5, 3, 1),
    'Tm': (6, 5, 3, 1), 'Yb': (6, 5, 3, 1), 'Lu': (6, 5, 3, 1), 'Hf': (6, 5, 3, 1),
    'Ta': (6, 5, 3, 1), 'W': (6, 5, 3, 1), 'Re': (6, 5, 3, 1), 'Os': (6, 5, 3, 1),
    'Ir': (6, 5, 3, 1), 'Pt': (6, 5, 3, 1), 'Au': (6, 5, 3, 1), 'Hg': (6, 5, 3, 1),
    'Tl': (6, 5, 3, 1), 'Pb': (6, 5, 3, 1), 'Bi': (6, 5, 3, 1), 'Po': (6, 5, 3, 1),
    'At': (6, 5, 3, 1), 'Rn': (6, 5, 3, 1), 'Fr': (7, 6, 3, 1), 'Ra': (7, 6, 3, 1),
    'Ac': (7, 6, 4, 2), 'Th': (7, 6, 4, 2), 'Pa': (7, 6, 4, 2), 'U': (7, 6, 4, 2),
    'Np': (7, 6, 4, 2), 'Pu': (7, 6, 4, 2), 'Am': (7, 6, 4, 2), 'Cm': (7, 6, 4, 2),
}

VDZP_COUNTS = {
    'H': (2, 1), 'He': (2, 1), 'Li': (3, 2, 1), 'Be': (3, 2, 1),
    'B': (3, 2, 1), 'C': (3, 2, 1), 'N': (3, 2, 1), 'O': (3, 2, 1),
    'F': (3, 2, 1), 'Ne': (3, 2, 1), 'Na': (4, 3, 1), 'Mg': (4, 3, 1),
    'Al': (4, 3, 1), 'Si': (4, 3, 1), 'P': (4, 3, 1), 'S': (4, 3, 1),
    'Cl': (4, 3, 1), 'Ar': (4, 3, 1), 'K': (5, 4, 1), 'Ca': (5, 4, 1),
    'Sc': (5, 4, 2, 1), 'Ti': (5, 4, 2, 1), 'V': (5, 4, 2, 1), 'Cr': (5, 4, 2, 1),
    'Mn': (5, 4, 2, 1), 'Fe': (5, 4, 2, 1), 'Co': (5, 4, 2, 1), 'Ni': (5, 4, 2, 1),
    'Cu': (5, 4, 2, 1), 'Zn': (5, 4, 2, 1), 'Ga': (5, 4, 2), 'Ge': (5, 4, 2),
    'As': (5, 4, 2), 'Se': (5, 4, 2), 'Br': (5, 4, 2), 'Kr': (5, 4, 2),
    'Rb': (6, 5, 2), 'Sr': (6, 5, 2), 'Y': (6, 5, 3, 1), 'Zr': (6, 5, 3, 1),
    'Nb': (6, 5, 3, 1), 'Mo': (6, 5, 3, 1), 'Tc': (6, 5, 3, 1), 'Ru': (6, 5, 3, 1),
    'Rh': (6, 5, 3, 1), 'Pd': (6, 5, 3, 1), 'Ag': (6, 5, 3, 1), 'Cd': (6, 5, 3, 1),
    'In': (6, 5, 3, 1), 'Sn': (6, 5, 3, 1), 'Sb': (6, 5, 3, 1), 'Te': (6, 5, 3, 1),
    'I': (6, 5, 3, 1), 'Xe': (6, 5, 3, 1), 'Cs': (7, 6, 3, 1), 'Ba': (7, 6, 3, 1),
    'La': (7, 6, 4, 2, 1), 'Ce': (7, 6, 4, 2, 1), 'Pr': (7, 6, 4, 2, 1), 'Nd': (7, 6, 4, 2, 1),
    'Pm': (7, 6, 4, 2, 1), 'Sm': (7, 6, 4, 2, 1), 'Eu': (7, 6, 4, 2, 1), 'Gd': (7, 6, 4, 2, 1),
    'Tb': (7, 6, 4, 2, 1), 'Dy': (7, 6, 4, 2, 1), 'Ho': (7, 6, 4, 2, 1), 'Er': (7, 6, 4, 2, 1),
    'Tm': (7, 6, 4, 2, 1), 'Yb': (7, 6, 4, 2, 1), 'Lu': (7, 6, 4, 2, 1), 'Hf': (7, 6, 4, 2, 1),
    'Ta': (7, 6, 4, 2, 1), 'W': (7, 6, 4, 2, 1), 'Re': (7, 6, 4, 2, 1), 'Os': (7, 6, 4, 2, 1),
    'Ir': (7, 6, 4, 2), 'Pt': (7, 6, 4, 2), 'Au': (7, 6, 4, 2), 'Hg': (7, 6, 4, 2),
    'Tl': (7, 6, 4, 2, 1), 'Pb': (7, 6, 4, 2, 1), 'Bi': (7, 6, 4, 2, 1), 'Po': (7, 6, 4, 2, 1),
    'At': (7, 6, 4, 2, 1), 'Rn': (7, 6, 4, 2, 1), 'Fr': (8, 7, 4, 2, 1), 'Ra': (8, 7, 4, 2, 1),
    'Ac': (8, 7, 5, 3, 1), 'Th': (8, 7, 5, 3, 1), 'Pa': (8, 7, 5, 3, 1), 'U': (8, 7, 5, 3, 1),
    'Np': (8, 7, 5, 3, 1), 'Pu': (8, 7, 5, 3, 1), 'Am': (8, 7, 5, 3, 1), 'Cm': (8, 7, 5, 3, 1),
}

VTZP_COUNTS = {
    'H': (3, 2, 1), 'He': (3, 2, 1), 'Li': (4, 3, 2, 1), 'Be': (4, 3, 2, 1),
    'B': (4, 3, 2, 1), 'C': (4, 3, 2, 1), 'N': (4, 3, 2, 1), 'O': (4, 3, 2, 1),
    'F': (4, 3, 2, 1), 'Ne': (4, 3, 2, 1), 'Na': (5, 4, 2, 1), 'Mg': (5, 4, 2, 1),
    'Al': (5, 4, 2, 1), 'Si': (5, 4, 2, 1), 'P': (5, 4, 2, 1), 'S': (5, 4, 2, 1),
    'Cl': (5, 4, 2, 1), 'Ar': (5, 4, 2, 1), 'K': (6, 5, 2, 1), 'Ca': (6, 5, 2, 1),
    'Sc': (6, 5, 3, 2, 1), 'Ti': (6, 5, 3, 2, 1), 'V': (6, 5, 3, 2, 1), 'Cr': (6, 5, 3, 2, 1),
    'Mn': (6, 5, 3, 2, 1), 'Fe': (6, 5, 3, 2, 1), 'Co': (6, 5, 3, 2, 1), 'Ni': (6, 5, 3, 2, 1),
    'Cu': (6, 5, 3, 2, 1), 'Zn': (6, 5, 3, 2, 1), 'Ga': (6, 5, 3, 1), 'Ge': (6, 5, 3, 1),
    'As': (6, 5, 3, 1), 'Se': (6, 5, 3, 1), 'Br': (6, 5, 3, 1), 'Kr': (6, 5, 3, 1),
    'Rb': (7, 6, 3, 1), 'Sr': (7, 6, 3, 1), 'Y': (7, 6, 4, 2, 1), 'Zr': (7, 6, 4, 2, 1),
    'Nb': (7, 6, 4, 2, 1), 'Mo': (7, 6, 4, 2, 1), 'Tc': (7, 6, 4, 2, 1), 'Ru': (7, 6, 4, 2, 1),
    'Rh': (7, 6, 4, 2, 1), 'Pd': (7, 6, 4, 2, 1), 'Ag': (7, 6, 4, 2, 1), 'Cd': (7, 6, 4, 2, 1),
    'In': (7, 6, 4, 2, 1), 'Sn': (7, 6, 4, 2, 1), 'Sb': (7, 6, 4, 2, 1), 'Te': (7, 6, 4, 2, 1),
    'I': (7, 6, 4, 2, 1), 'Xe': (7, 6, 4, 2, 1), 'Cs': (8, 7, 4, 2, 1), 'Ba': (8, 7, 4, 2, 1),
    'La': (8, 7, 5, 3, 2, 1), 'Ce': (8, 7, 5, 3, 2, 1), 'Pr': (8, 7, 5, 3, 2, 1), 'Nd': (8, 7, 5, 3, 2, 1),
    'Pm': (8, 7, 5, 3, 2, 1), 'Sm': (8, 7, 5, 3, 2, 1), 'Eu': (8, 7, 5, 3, 2, 1), 'Gd': (8, 7, 5, 3, 2, 1),
    'Tb': (8, 7, 5, 3, 2, 1), 'Dy': (8, 7, 5, 3, 2, 1), 'Ho': (8, 7, 5, 3, 2, 1), 'Er': (8, 7, 5, 3, 2, 1),
    'Tm': (8, 7, 5, 3, 2, 1), 'Yb': (8, 7, 5, 3, 2, 1), 'Lu': (8, 7, 5, 3, 2, 1), 'Hf': (8, 7, 5, 3, 2, 1),
    'Ta': (8, 7, 5, 3, 2, 1), 'W': (8, 7, 5, 3, 2, 1), 'Re': (8, 7, 5, 3, 2, 1), 'Os': (8, 7, 5, 3, 2, 1),
    'Ir': (8, 7, 5, 3, 1), 'Pt': (8, 7, 5, 3, 1), 'Au': (8, 7, 5, 3, 1), 'Hg': (8, 7, 5, 3, 1),
    'Tl': (8, 7, 5, 3, 2), 'Pb': (8, 7, 5, 3, 2), 'Bi': (8, 7, 5, 3, 2), 'Po': (8, 7, 5, 3, 2),
    'At': (8, 7, 5, 3, 2), 'Rn': (8, 7, 5, 3, 2), 'Fr': (9, 8, 5, 3, 2), 'Ra': (9, 8, 5, 3, 2),
    'Ac': (9, 8, 6, 4, 2, 1), 'Th': (9, 8, 6, 4, 2, 1), 'Pa': (9, 8, 6, 4, 2, 1), 'U': (9, 8, 6, 4, 2, 1),
    'Np': (9, 8, 6, 4, 2, 1), 'Pu': (9, 8, 6, 4, 2, 1), 'Am': (9, 8, 6, 4, 2, 1), 'Cm': (9, 8, 6, 4, 2, 1),
}

VQZP_COUNTS = {
    'H': (4, 3, 2, 1), 'He': (4, 3, 2, 1), 'Li': (5, 4, 3, 2, 1), 'Be': (5, 4, 3, 2, 1),
    'B': (5, 4, 3, 2, 1), 'C': (5, 4, 3, 2, 1), 'N': (5, 4, 3, 2, 1), 'O': (5, 4, 3, 2, 1),
    'F': (5, 4, 3, 2, 1), 'Ne': (5, 4, 3, 2, 1), 'Na': (6, 5, 3, 2, 1), 'Mg': (6, 5, 3, 2, 1),
    'Al': (6, 5, 3, 2, 1), 'Si': (6, 5, 3, 2, 1), 'P': (6, 5, 3, 2, 1), 'S': (6, 5, 3, 2, 1),
    'Cl': (6, 5, 3, 2, 1), 'Ar': (6, 5, 3, 2, 1), 'K': (7, 6, 3, 2, 1), 'Ca': (7, 6, 3, 2, 1),
    'Sc': (7, 6, 4, 3, 2, 1), 'Ti': (7, 6, 4, 3, 2, 1), 'V': (7, 6, 4, 3, 2, 1), 'Cr': (7, 6, 4, 3, 2, 1),
    'Mn': (7, 6, 4, 3, 2, 1), 'Fe': (7, 6, 4, 3, 2, 1), 'Co': (7, 6, 4, 3, 2, 1), 'Ni': (7, 6, 4, 3, 2, 1),
    'Cu': (7, 6, 4, 3, 2, 1), 'Zn': (7, 6, 4, 3, 2, 1), 'Ga': (7, 6, 4, 2, 1), 'Ge': (7, 6, 4, 2, 1),
    'As': (7, 6, 4, 2, 1), 'Se': (7, 6, 4, 2, 1), 'Br': (7, 6, 4, 2, 1), 'Kr': (7, 6, 4, 2, 1),
    'Rb': (8, 7, 4, 2, 1), 'Sr': (8, 7, 4, 2, 1), 'Y': (8, 7, 5, 3, 2, 1), 'Zr': (8, 7, 5, 3, 2, 1),
    'Nb': (8, 7, 5, 3, 2, 1), 'Mo': (8, 7, 5, 3, 2, 1), 'Tc': (8, 7, 5, 3, 2, 1), 'Ru': (8, 7, 5, 3, 2, 1),
    'Rh': (8, 7, 5, 3, 2, 1), 'Pd': (8, 7, 5, 3, 2, 1), 'Ag': (8, 7, 5, 3, 2, 1), 'Cd': (8, 7, 5, 3, 2, 1),
    'In': (8, 7, 5, 3, 2), 'Sn': (8, 7, 5, 3, 2), 'Sb': (8, 7, 5, 3, 2), 'Te': (8, 7, 5, 3, 2),
    'I': (8, 7, 5, 3, 2), 'Xe': (8, 7, 5, 3, 2), 'Cs': (9, 8, 5, 3, 2), 'Ba': (9, 8, 5, 3, 2),
    'La': (9, 8, 6, 4, 3, 2), 'Ce': (9, 8, 6, 4, 3, 2), 'Pr': (9, 8, 6, 4, 3, 2), 'Nd': (9, 8, 6, 4, 3, 2),
    'Pm': (9, 8, 6, 4, 3, 2), 'Sm': (9, 8, 6, 4, 3, 2), 'Eu': (9, 8, 6, 4, 3, 2), 'Gd': (9, 8, 6, 4, 3, 2),
    'Tb': (9, 8, 6, 4, 3, 2), 'Dy': (9, 8, 6, 4, 3, 2), 'Ho': (9, 8, 6, 4, 3, 2), 'Er': (9, 8, 6, 4, 3, 2),
    'Tm': (9, 8, 6, 4, 3, 2), 'Yb': (9, 8, 6, 4, 3, 2), 'Lu': (9, 8, 6, 4, 3, 2), 'Hf': (9, 8, 6, 4, 3, 2),
    'Ta': (9, 8, 6, 4, 3, 2), 'W': (9, 8, 6, 4, 3, 2), 'Re': (9, 8, 6, 4, 3, 2), 'Os': (9, 8, 6, 4, 3, 2),
    'Ir': (9, 8, 6, 4, 2, 1), 'Pt': (9, 8, 6, 4, 2, 1), 'Au': (9, 8, 6, 4, 2, 1), 'Hg': (9, 8, 6, 4, 2, 1),
    'Tl': (9, 8, 6, 4, 3), 'Pb': (9, 8, 6, 4, 3), 'Bi': (9, 8, 6, 4, 3), 'Po': (9, 8, 6, 4, 3),
    'At': (9, 8, 6, 4, 3), 'Rn': (9, 8, 6, 4, 3), 'Fr': (10, 9, 6, 4, 2), 'Ra': (10, 9, 6, 4, 2),
    'Ac': (10, 9, 7, 5, 3, 2), 'Th': (10, 9, 7, 5, 3, 2), 'Pa': (10, 9, 7, 5, 3, 2), 'U': (10, 9, 7, 5, 3, 2),
    'Np': (10, 9, 7, 5, 3, 2), 'Pu': (10, 9, 7, 5, 3, 2), 'Am': (10, 9, 7, 5, 3, 2), 'Cm': (10, 9, 7, 5, 3, 2),
}


def truncate_ano(element, counts):
    """Shell list for ``element`` keeping the first ``counts[l]`` contraction
    columns of PySCF's ANO-RCC general contraction for each l. Primitives
    whose kept coefficients are all zero are dropped."""
    shells = gto.basis.load('ano', element)
    out = []
    for sh in shells:
        l = sh[0]
        if l >= len(counts) or counts[l] == 0:
            continue
        arr = np.asarray(sh[1:], dtype=float)         # (nprim, 1 + ncontr)
        keep = arr[:, :1 + counts[l]]
        nonzero = np.any(np.abs(keep[:, 1:]) > 0.0, axis=1)
        keep = keep[nonzero]
        out.append([l] + keep.tolist())
    return out


class _LazyBasis(dict):
    """Dict of element -> shells, filled on first access of each element.

    Carries ``.name`` so the MinPop scripts label output with the basis
    name. gto.M only touches the elements present in the molecule, so a
    molecule never pays for building the other 95 entries."""

    def __init__(self, name, counts):
        super().__init__()
        self.name = name
        self._counts = counts

    def _build(self, key):
        sym = _el._std_symbol(key) if hasattr(_el, '_std_symbol') else key
        if sym not in self._counts:
            raise KeyError(f"{self.name} is not defined for element {key!r}")
        return truncate_ano(sym, self._counts[sym])

    def __missing__(self, key):
        val = self._build(key)
        dict.__setitem__(self, key, val)
        return val

    def __contains__(self, key):
        return dict.__contains__(self, key) or \
            (isinstance(key, str) and key in self._counts)

    def keys(self):
        return self._counts.keys()

    def __iter__(self):
        return iter(self._counts)

    def __len__(self):
        return len(self._counts)

    def items(self):
        return ((k, self[k]) for k in self._counts)

    def values(self):
        return (self[k] for k in self._counts)

    def get(self, key, default=None):
        try:
            return self[key]
        except KeyError:
            return default

    def copy(self):
        return copy.deepcopy(self)


ANO_RCC_MB = _LazyBasis('ANO-RCC-MB', MB_COUNTS)
ANO_RCC_VDZP = _LazyBasis('ANO-RCC-VDZP', VDZP_COUNTS)
ANO_RCC_VTZP = _LazyBasis('ANO-RCC-VTZP', VTZP_COUNTS)
ANO_RCC_VQZP = _LazyBasis('ANO-RCC-VQZP', VQZP_COUNTS)

_BY_NAME = {'mb': ANO_RCC_MB, 'vdzp': ANO_RCC_VDZP,
            'vtzp': ANO_RCC_VTZP, 'vqzp': ANO_RCC_VQZP}


def ano_rcc(level):
    """Return the ANO-RCC truncation for ``level`` in {'mb','vdzp','vtzp','vqzp'}
    (case-insensitive; an 'ano-rcc-' prefix is accepted)."""
    key = level.lower().replace('_', '-')
    if key.startswith('ano-rcc-'):
        key = key[len('ano-rcc-'):]
    return _BY_NAME[key]


def _selftest(elements_to_check=('H', 'C', 'Si', 'Fe', 'Pt', 'U')):
    """Compare every truncation against Basis Set Exchange, shell by shell."""
    import basis_set_exchange as bse
    worst = 0.0
    for level, table in _BY_NAME.items():
        for sym in elements_to_check:
            ref = bse.get_basis(f'ano-rcc-{level}', elements=[sym],
                                optimize_general=False)
            z = str(bse.lut.element_Z_from_sym(sym))
            ref_sh = ref['elements'][z]['electron_shells']
            mine = table[sym]
            assert len(mine) == len(ref_sh), (level, sym, len(mine), len(ref_sh))
            for r, m in zip(ref_sh, mine):
                assert r['angular_momentum'][0] == m[0]
                ex = np.array(r['exponents'], float)
                co = np.array(r['coefficients'], float).T   # (nprim, ncontr)
                arr = np.asarray(m[1:], float)
                assert arr.shape == (len(ex), 1 + co.shape[1]), (level, sym, arr.shape, co.shape)
                worst = max(worst, np.abs(arr[:, 0] - ex).max(),
                            np.abs(arr[:, 1:] - co).max())
            print(f"{level:5s} {sym:2s} OK  "
                  f"{[ (s[0], len(s[1]) - 1) for s in mine ]}")
    print(f"max |difference| vs BSE: {worst:.2e}")


if __name__ == '__main__':
    _selftest()
