#!/usr/bin/env python3
"""
Minimum Population (MinPop) Localization Analysis for UHF Wavefunctions.

This module projects Unrestricted Hartree-Fock (UHF) molecular orbitals onto a
minimal basis set for chemically intuitive population analysis. Output format
matches Gaussian 16 for direct comparison.

Theory
------
The MinPop method projects extended basis set MO coefficients C onto a minimal
basis using the transformation:

    C' = S_min^(-1) · S_cross · C · M^(-1/2)

where:
    S_min   : Overlap matrix in minimal basis
    S_cross : Cross-overlap matrix ⟨minimal|extended⟩
    M       : Metric matrix ensuring orthonormality, M = C'ᵀ·S_min·C'

For UHF wavefunctions, alpha and beta orbitals have different spatial parts and
must be projected independently.

Minimal Basis Selection
-----------------------
Following Gaussian's convention:
    - First row (H–Ne):       STO-3G
    - Second row (Na–Ar):     STO-3G* (with d-polarization, Cartesian 6D)
    - Third row+ (K–Kr):      STO-3G (spherical 5D for d-orbitals)
    - Fourth row+ (Rb–Xe):    STO-3G with ECP for heavy elements (Z > 36)

Spin Annihilation
-----------------
UHF wavefunctions are contaminated by higher spin states. The first spin
contaminant (S+1) is removed using Löwdin's projection operator method.
For singlets, the annihilated spin density is exactly zero.

ECP Support
-----------
For heavy elements (Z > 36) with def2 family basis sets, effective core
potentials (ECPs) are automatically applied to replace core electrons.

References
----------
[1] Montgomery Jr., J. A. et al. J. Chem. Phys. 110, 2822–2827 (1999).
[2] Montgomery Jr., J. A. et al. J. Chem. Phys. 112, 6532–6542 (2000).
[3] Löwdin, P.-O. Phys. Rev. 97, 1509–1520 (1955).

Author: Barbaro Zulueta (Pitt Quantum Repository)
"""

import argparse
import importlib
import importlib.util
import contextlib
import io
import os
import re
import shlex
import sys
import numpy as np
from pyscf import gto, scf
from pyscf.gto import intor_cross

# Make the sibling modules (bond_symmetry, basis modules, minpop_json_csv, ...)
# importable whether this file runs as a script or as part of the minpop package.
_HERE = os.path.dirname(os.path.abspath(__file__))
for _d in (_HERE, os.path.join(_HERE, "basis_sets")):
    if _d not in sys.path:
        sys.path.insert(0, _d)

__version__ = "1.1.0"
__author__ = "Barbaro Zulueta"
__all__ = ["minpop_uhf", "run_uhf_from_xyz"]


# =============================================================================
# Constants and Configuration
# =============================================================================

# Second-row elements use STO-3G* (with d-polarization) for minimal basis
SECOND_ROW_ELEMENTS = frozenset({'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar'})

# Basis sets that typically use ECPs for heavy elements
ECP_BASIS_SETS = frozenset({
    'def2-svp', 'def2-svpd', 'def2-tzvp', 'def2-tzvpd', 'def2-tzvpp', 'def2-tzvppd',
    'def2-qzvp', 'def2-qzvpd', 'def2-qzvpp', 'def2-qzvppd',
    'def2svp', 'def2svpd', 'def2tzvp', 'def2tzvpd', 'def2tzvpp', 'def2tzvppd',
    'def2qzvp', 'def2qzvpd', 'def2qzvpp', 'def2qzvppd',
    'lanl2dz', 'lanl2tz', 'lanl08', 'sdd', 'stuttgart'
})

# Atomic number ranges for transition metals
TRANSITION_METAL_RANGES = [(21, 30), (39, 48), (57, 80), (89, 112)]

# Short labels for custom Gaussian-derived basis sets stored as PySCF dicts in
# standalone modules (e.g. cbsb7_basis_pyscf.py defines a dict named CBSB7).
CUSTOM_BASIS_MODULES = {
    'cbsb3': ('cbsb3_basis_pyscf', 'CBSB3'),
    'cbsb7': ('cbsb7_basis_pyscf', 'CBSB7'),
    # ANO-RCC truncations, sliced from PySCF's built-in ANO-RCC general
    # contraction by anorcc_basis_pyscf.py (all-electron, DKH2-contracted;
    # pair with -x2c for heavy atoms).
    'anorccmb':   ('anorcc_basis_pyscf', 'ANO_RCC_MB'),
    'anorccvdzp': ('anorcc_basis_pyscf', 'ANO_RCC_VDZP'),
    'anorccvtzp': ('anorcc_basis_pyscf', 'ANO_RCC_VTZP'),
    'anorccvqzp': ('anorcc_basis_pyscf', 'ANO_RCC_VQZP'),
}

# Minimal-basis projection targets accepted by -minimal-basis.
MINIMAL_BASIS_CHOICES = ('sto-3g', 'minao', 'ano-rcc-mb', 'sto-3g-ano')

# Per-element minimal basis: 'LIGHT+HEAVY' puts LIGHT on atoms with
# Z <= MBS_LIGHT_MAX_Z and HEAVY on the rest, e.g. 'sto-3g+ano-rcc-mb' or
# 'sto-3g+minao'. The cutoff follows -light-max-z (default 18, Ar).
MBS_LIGHT_MAX_Z = 18


def _check_minimal_basis(name):
    parts = [s.strip() for s in str(name).split('+')]
    for s in parts:
        if _mbs_key(s) not in [_mbs_key(c) for c in MINIMAL_BASIS_CHOICES]:
            raise ValueError(f"unknown minimal basis '{s}'; choose from "
                             f"{MINIMAL_BASIS_CHOICES} or LIGHT+HEAVY")
    if len(parts) > 2:
        raise ValueError("minimal basis takes at most LIGHT+HEAVY")
    return name

# 'sto-3g-ano' = STO-3G (STO-3G* on Na-Ar) on Z <= MBS_LIGHT_MAX_Z (Ar) and
# ANO-RCC-MB on every heavier atom, i.e. the alias of 'sto-3g+ano-rcc-mb'.

# AO label numbering for the generic (ano-rcc-mb / sto-3g-ano) ordering path.
# 'physical' (default): principal quantum number (Pt 5D, 6S, 4F).
# 'gaussian': running shell counter per atom as Gaussian prints it (9D, 10S,
# 7F), meaningful only up to Xe. The legacy STO-3G path always prints
# Gaussian's labels. Labels only; numbers and ordering are identical.
LABEL_STYLE = None   # None: 'gaussian' for sto-3g / sto-3g-ano, 'physical' for ano-rcc-mb

_SYMMETRY_SYMBOL = {'sigma': '\u03c3', 'pi': '\u03c0', 'delta': '\u03b4', 'phi': '\u03c6'}

# Print the σ/π/δ/φ resolution of every bond population (bond_symmetry.py,
# exact rotation into the diatomic frame). Set by -symmetry / -symmetry-shells.
BOND_SYMMETRY = False
BOND_SYMMETRY_SHELLS = False


def _mbs_key(minimal_basis):
    """Canonical key for a -minimal-basis value: 'sto3g', 'minao', 'anorccmb'."""
    return str(minimal_basis).lower().replace('-', '').replace('_', '')


# =============================================================================
# Basis Resolution (built-in names + custom PySCF-dict modules)
# =============================================================================

def _basis_search_dirs(search_dir=None):
    """Directories to look in for custom basis modules, most specific first."""
    dirs = []
    if search_dir:
        dirs.append(search_dir)
    dirs.append(os.getcwd())
    try:
        here = os.path.dirname(os.path.abspath(__file__))
        dirs.append(here)
        dirs.append(os.path.join(here, "basis_sets"))
    except NameError:  # __file__ undefined in some interactive contexts
        pass
    # de-duplicate while preserving order
    seen, unique = set(), []
    for d in dirs:
        ad = os.path.abspath(d)
        if ad not in seen:
            seen.add(ad)
            unique.append(ad)
    return unique


class _NamedBasis(dict):
    """
    A PySCF basis dict that also remembers the basis-set name.

    Behaves exactly like the underlying {element: shells} dict as far as gto.M
    is concerned, but carries a `.name` attribute so the MinPop tools can label
    output with the real basis name (from the module's BASIS_NAME) instead of a
    generic "custom dict".
    """
    def __init__(self, mapping, name=None):
        super().__init__(mapping)
        self.name = name


def _extract_basis_dict(module, dict_name=None, hint=''):
    """
    Pull the basis dict out of an imported module, tagged with its name.

    The name is taken from the module-level BASIS_NAME - the standard every
    *_basis_pyscf.py module should follow - falling back to the explicit dict
    name or the CBSB3-style stem derived from the filename when BASIS_NAME is
    absent.
    """
    stem = (hint or getattr(module, '__name__', '')).upper()
    stem = stem.replace('_BASIS_PYSCF', '').replace('_BASIS', '')

    if dict_name:
        basis_dict = getattr(module, dict_name)
    elif stem and hasattr(module, stem):
        basis_dict = getattr(module, stem)
    else:
        # fall back to the first module-level dict that looks like a basis table
        basis_dict = None
        for attr, val in vars(module).items():
            if attr.startswith('_') or not isinstance(val, dict) or not val:
                continue
            if all(isinstance(k, str) for k in val) and \
               all(isinstance(v, list) for v in val.values()):
                basis_dict = val
                break
        if basis_dict is None:
            raise ValueError(
                f"could not find a basis dict in module {stem or module!r}; "
                f"specify it explicitly as 'module:DICTNAME'.")

    # Standard convention: the module declares its display name in BASIS_NAME.
    name = (getattr(module, 'BASIS_NAME', None)
            or getattr(basis_dict, 'name', None)
            or dict_name or stem or None)
    return _NamedBasis(basis_dict, name=name)


def _load_basis_from_pyfile(path, dict_name=None):
    """Load a basis dict from a standalone .py file given its path."""
    path = os.path.abspath(path)
    mod_name = os.path.splitext(os.path.basename(path))[0]
    spec = importlib.util.spec_from_file_location(mod_name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return _extract_basis_dict(module, dict_name, hint=mod_name)


def _resolve_basis(basis, search_dir=None, dict_name=None):
    """
    Turn the -basis argument into something gto.M accepts.

    Accepted forms:
      * a dict                 -> returned unchanged (already a PySCF basis)
      * 'cbsb3' / 'cbsb7'      -> loads CBSB3/CBSB7 from <name>_basis_pyscf.py
      * a path to a .py file   -> loads the basis dict from that file
      * 'module:DICTNAME'      -> imports the module, returns its DICTNAME attr
      * any other string       -> returned unchanged (standard PySCF basis name)

    Custom basis modules are searched for in search_dir, the current working
    directory, and the directory holding this script (in that order).
    """
    if not isinstance(basis, str):
        return basis  # dict (or other object) passed programmatically

    for d in _basis_search_dirs(search_dir):
        if d not in sys.path:
            sys.path.insert(0, d)

    spec = basis.strip()

    # explicit "module:DICTNAME"
    if ':' in spec and not spec.lower().endswith('.py'):
        mod_name, _, dname = spec.partition(':')
        module = importlib.import_module(mod_name)
        return _extract_basis_dict(module, dname or dict_name, hint=mod_name)

    # path to a .py file
    if spec.lower().endswith('.py') and os.path.exists(spec):
        return _load_basis_from_pyfile(spec, dict_name)

    # known short label (case/dash/underscore-insensitive)
    key = spec.lower().replace('-', '').replace('_', '')
    if key in CUSTOM_BASIS_MODULES:
        mod_name, dname = CUSTOM_BASIS_MODULES[key]
        try:
            module = importlib.import_module(mod_name)
        except ImportError as exc:
            raise ImportError(
                f"basis '{spec}' requires the module '{mod_name}.py' to be "
                f"importable (place it in the current directory, next to this "
                f"script, or pass -basis-dir). Original error: {exc}")
        return _extract_basis_dict(module, dict_name or dname, hint=mod_name)

    # standard basis name (e.g. '6-31+G', 'cc-pVDZ', 'def2-TZVPP')
    return basis


def _basis_label(basis):
    """Human-readable label for a resolved-or-unresolved basis argument."""
    if isinstance(basis, str):
        return basis
    name = getattr(basis, 'name', None)
    if name:
        return name
    if isinstance(basis, dict):
        return f"custom dict ({len(basis)} elements)"
    return repr(basis)



def _compose_basis(basis, basis_light, atom_str, light_max_z=18, search_dir=None):
    """
    Per-element basis assignment: `basis` on atoms with Z > light_max_z,
    `basis_light` on the rest (H-Ar by default). Returns (basis_obj, ecp, label).

    Either name may be a standard PySCF basis, 'cbsb3'/'cbsb7', an ANO-RCC
    truncation, a .py path or 'module:DICT'. ECPs are auto-detected per side
    from the basis that actually sits on each atom, so 'def2-tzvp' on the
    heavy atoms gets its ECP while '6-31G*' on H-Ar does not.
    With basis_light None the molecule uses `basis` everywhere (old behavior).
    """
    from pyscf.data import elements
    heavy = _resolve_basis(basis, search_dir=search_dir)
    if basis_light is None:
        ecp = _get_ecp_dict(heavy, atom_str) if isinstance(heavy, str) else None
        return heavy, ecp, _basis_label(heavy)
    light = _resolve_basis(basis_light, search_dir=search_dir)

    def pick(resolved, elem):
        if isinstance(resolved, str):
            return resolved
        if elem in resolved:
            return resolved[elem]
        raise KeyError(f"basis '{_basis_label(resolved)}' has no entry for {elem}")

    basis_dict, symbols = {}, []
    for line in atom_str.replace(';', '\n').split('\n'):
        parts = line.split()
        if not parts:
            continue
        sym = parts[0]
        elem = ''.join(c for c in sym if c.isalpha())
        if sym in basis_dict:
            continue
        z = elements.charge(elem)
        basis_dict[sym] = pick(light if z <= light_max_z else heavy, elem)
        symbols.append((sym, elem, z))

    # ECP only for the heavy side, and only if that basis uses one
    ecp = None
    if isinstance(heavy, str):
        heavy_atoms = '\n'.join(f"{s} 0 0 0" for s, e, z in symbols if z > light_max_z)
        ecp = _get_ecp_dict(heavy, heavy_atoms) if heavy_atoms else None
    label = (f"{_basis_label(heavy)} on Z>{light_max_z}, "
             f"{_basis_label(light)} on Z<={light_max_z}")
    return basis_dict, ecp, label


# =============================================================================
# Minimal Basis Construction
# =============================================================================

def _is_transition_metal(atomic_number):
    """Check if atomic number corresponds to a transition metal."""
    return any(low <= atomic_number <= high for low, high in TRANSITION_METAL_RANGES)


def _build_minimal_basis_mol(mol, minimal_basis='sto-3g'):
    """Minimal-basis molecule for the projection (see _build_minimal_basis_mol_impl).
    Also records the ECP core electron count of each atom of `mol` on the
    result (`_minpop_ecp_core`) so valence-only minimal shells can be labelled
    with physical principal quantum numbers."""
    mol_min = _build_minimal_basis_mol_impl(mol, minimal_basis)
    try:
        mol_min._minpop_ecp_core = [int(mol.atom_nelec_core(i)) for i in range(mol.natm)]
    except Exception:
        mol_min._minpop_ecp_core = [0] * mol.natm
    return mol_min


def _build_minimal_basis_mol_impl(mol, minimal_basis='sto-3g'):
    """
    Construct minimal basis molecule for the MinPop projection.
    
    Parameters
    ----------
    mol : pyscf.gto.Mole
        Input molecule in extended basis set
    minimal_basis : str, optional
        Projection target: 'sto-3g' (default; Gaussian's MBS convention,
        with STO-3G* on second-row elements) or 'minao'. For CHNOF the two
        share the same shell structure (1s / 1s2s2p), so all downstream
        Gaussian-ordering logic applies unchanged; MINAO projection with
        transition metals present should be re-validated before use.
        
    Returns
    -------
    mol_min : pyscf.gto.Mole
        Molecule with minimal basis set
        
    Notes
    -----
    - STO-3G branch: second-row elements (Na–Ar) use STO-3G* with Cartesian
      d-orbitals (6D); transition metals use STO-3G (5D); the minimal basis
      itself is always built spherical (Gaussian MinPop convention).
    - MINAO branch: PySCF's MINAO on every atom, spherical.
    """
    from pyscf.data import elements
    
    key = _mbs_key(minimal_basis)
    if '+' in str(minimal_basis):
        # LIGHT+HEAVY: build each side with this same function and merge.
        light_name, heavy_name = [s.strip() for s in minimal_basis.split('+')]
        light_mol = _build_minimal_basis_mol_impl(mol, light_name)
        heavy_mol = _build_minimal_basis_mol_impl(mol, heavy_name)
        basis_dict = {}
        for i in range(mol.natm):
            symbol = mol.atom_symbol(i)
            z = elements.charge(''.join(c for c in symbol if c.isalpha()))
            src = light_mol if z <= MBS_LIGHT_MAX_Z else heavy_mol
            basis_dict[symbol] = src._basis[symbol]
        return gto.M(atom=mol.atom, basis=basis_dict, charge=mol.charge,
                     spin=mol.spin, cart=False)
    if key == 'minao':
        return gto.M(
            atom=mol.atom,
            basis={mol.atom_symbol(i): 'minao' for i in range(mol.natm)},
            charge=mol.charge,
            spin=mol.spin,
            cart=False
        )
    if key == 'sto3gano':
        # STO-3G / STO-3G* on main-group atoms up to Ar, ANO-RCC-MB on every
        # heavier atom (transition metals included). Same as 'sto-3g+ano-rcc-mb'.
        return _build_minimal_basis_mol_impl(mol, 'sto-3g+ano-rcc-mb')
    if key == 'anorccmb':
        # ANO-RCC-MB: one contracted ANO per occupied atomic orbital plus the
        # valence p on metals (Pt: 6s5p3d1f, Fe: 4s3p1d, C: 2s1p), i.e. the
        # same shell structure STO-3G gives for H-Kr, extended all-electron
        # and relativistically contracted through the whole periodic table.
        # Shells are ordered generically by (n, l, m) downstream, so no
        # element-specific Gaussian shell-number hacks are needed.
        anorcc = _resolve_basis('anorccmb')
        return gto.M(
            atom=mol.atom,
            basis={mol.atom_symbol(i): anorcc[''.join(
                c for c in mol.atom_symbol(i) if c.isalpha())]
                   for i in range(mol.natm)},
            charge=mol.charge,
            spin=mol.spin,
            cart=False
        )
    
    has_second_row = False
    has_transition_metal = False
    basis_dict = {}
    
    for i in range(mol.natm):
        symbol = mol.atom_symbol(i)
        elem = ''.join(c for c in symbol if c.isalpha())
        
        if elem in SECOND_ROW_ELEMENTS:
            basis_dict[symbol] = 'STO-3G*'
            has_second_row = True
        else:
            basis_dict[symbol] = 'STO-3G'
            try:
                z = elements.charge(elem)
                if _is_transition_metal(z):
                    has_transition_metal = True
            except (KeyError, ValueError):
                pass
    
    # Gaussian's MinPop uses a spherical (5D) minimal basis throughout,
    # including STO-3G* d-polarization on second-row atoms (verified vs
    # Gaussian: Si2 -> 28 AOs; Cl d-shell labeled 4D 0/4D+-1/4D+-2, not 6D).
    use_cartesian = False
    
    return gto.M(
        atom=mol.atom,
        basis=basis_dict,
        charge=mol.charge,
        spin=mol.spin,
        cart=use_cartesian
    )


# =============================================================================
# Orbital Projection
# =============================================================================

def _project_to_minimal_basis(mo_coeff, S_cross, S_min_inv):
    """
    Project MO coefficients to minimal basis with symmetric orthonormalization.
    
    Parameters
    ----------
    mo_coeff : ndarray (n_ao_ext, n_mo)
        MO coefficients in extended basis
    S_cross : ndarray (n_ao_min, n_ao_ext)
        Cross-overlap matrix ⟨minimal|extended⟩
    S_min_inv : ndarray (n_ao_min, n_ao_min)
        Inverse of minimal basis overlap matrix
        
    Returns
    -------
    mo_min : ndarray (n_ao_min, n_mo)
        Orthonormalized MO coefficients in minimal basis
    """
    # Raw projection: C' = S_min⁻¹ · S_cross · C
    C_proj = S_min_inv @ S_cross @ mo_coeff
    
    # Metric matrix: M = C'ᵀ · S_min · C'
    S_min = np.linalg.inv(S_min_inv)
    M = C_proj.T @ S_min @ C_proj
    
    # Symmetric orthonormalization via M^(-1/2)
    eigvals, eigvecs = np.linalg.eigh(M)
    eigvals = np.maximum(eigvals, 1e-14)  # Numerical stability
    M_invsqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
    
    return C_proj @ M_invsqrt


def _projection_completeness(mo_coeff, S_cross, S_min_inv, mo_energy=None):
    """
    How much of the occupied density the minimal basis can hold.

    The eigenvalues of the metric M = C^T S_cross^T S_min^{-1} S_cross C are
    the captured fractions of the occupied orbitals (1 = fully representable
    in the minimal basis). Their sum is the captured electron count per spin;
    the shortfall is density outside the span of the minimal basis. The
    symmetric orthonormalization that follows rescales the projected orbitals
    back to unit norm, so the reported Mulliken populations always sum to the
    full electron count. This diagnostic is the only place the leak shows.

    Returns
    -------
    dict with n_mo, captured (sum of eigenvalues), eigvals (ascending),
    worst (smallest eigenvalue) and worst_mo (0-based index of the occupied
    MO with the largest leak, from the eigenvector of the smallest eigenvalue).
    """
    C_proj = S_min_inv @ S_cross @ mo_coeff
    S_min = np.linalg.inv(S_min_inv)
    M = C_proj.T @ S_min @ C_proj
    eigvals, eigvecs = np.linalg.eigh(M)
    worst_mo = int(np.argmax(np.abs(eigvecs[:, 0])))
    return {'n_mo': mo_coeff.shape[1], 'captured': float(eigvals.sum()),
            'eigvals': eigvals, 'worst': float(eigvals[0]), 'worst_mo': worst_mo}


def _print_projection_completeness(label, info, n_elec_spin):
    leak = n_elec_spin - info['captured']
    print(f" Projection completeness ({label}): {info['captured']:.5f} of "
          f"{n_elec_spin:.0f} electrons captured, leak {leak:.5f} "
          f"({100.0 * leak / n_elec_spin:.3f}%); least-captured MO #{info['worst_mo'] + 1} "
          f"at {100.0 * info['worst']:.2f}%, {int((info['eigvals'] < 0.99).sum())} MOs below 99%")


# =============================================================================
# Spin Annihilation (Löwdin Projection)
# =============================================================================

def _compute_s2(mo_alpha, mo_beta, S_min):
    """
    Compute ⟨S²⟩ for UHF wavefunction in minimal basis.
    
    Formula: ⟨S²⟩ = S(S+1) + N_β - Σᵢⱼ |⟨αᵢ|βⱼ⟩|²
    
    where S = (N_α - N_β)/2 is the spin quantum number.
    
    Parameters
    ----------
    mo_alpha : ndarray (n_ao, n_alpha)
        Alpha MO coefficients in minimal basis
    mo_beta : ndarray (n_ao, n_beta)
        Beta MO coefficients in minimal basis
    S_min : ndarray (n_ao, n_ao)
        Minimal basis overlap matrix
        
    Returns
    -------
    s2 : float
        Expectation value of S² operator
    """
    n_alpha = mo_alpha.shape[1]
    n_beta = mo_beta.shape[1]
    
    S_exact = (n_alpha - n_beta) / 2.0
    s2_exact = S_exact * (S_exact + 1)
    
    # Alpha-beta overlap matrix
    overlap_ab = mo_alpha.T @ S_min @ mo_beta
    contamination = np.sum(overlap_ab ** 2)
    
    return s2_exact + n_beta - contamination


def _lowdin_annihilate(dm_spin_raw, S_min, n_alpha, n_beta):
    """
    Apply Löwdin spin annihilation to remove first spin contaminant.
    
    For UHF, higher spin states (S+1, S+2, ...) contaminate the wavefunction.
    This removes the first contaminant, projecting toward the pure spin state.
    
    Parameters
    ----------
    dm_spin_raw : ndarray
        Raw spin density matrix (D_α - D_β)
    S_min : ndarray
        Minimal basis overlap matrix
    n_alpha, n_beta : int
        Number of alpha and beta electrons
        
    Returns
    -------
    dm_spin : ndarray
        Spin-annihilated density matrix
    scale : float
        Scaling factor applied
    """
    S_target = (n_alpha - n_beta) / 2.0
    
    # Singlet case: spin density is exactly zero after annihilation
    if S_target == 0:
        return np.zeros_like(dm_spin_raw), 0.0
    
    # Scale spin density to target value
    raw_spin = np.trace(dm_spin_raw @ S_min)
    target_spin = 2 * S_target
    
    if abs(raw_spin) > 1e-10:
        scale = target_spin / raw_spin
    else:
        scale = 1.0
    
    return dm_spin_raw * scale, scale


def _annihilate_first_contaminant(mo_a, mo_b, S, n_a, n_b,
                                  tol=5e-4, max_pairs=6):
    """
    ⟨S²⟩ before and after Löwdin annihilation of the first spin contaminant.

    Reproduces Gaussian's "S**2 before/after annihilation". The contamination is
    read from the corresponding-orbital overlaps (singular values of the alpha-
    beta MO overlap): each pair with overlap λ contributes a spin defect
    t = 1 - λ² and behaves as a mix of a singlet and a triplet (Ms=0) component.

    ⟨S²⟩ before is exact: Sz(Sz+1) + Σ t_i. For the "after" value the same single
    Löwdin annihilator A = (Ŝ² - (Sz+1)(Sz+2)) / (Sz(Sz+1) - (Sz+1)(Sz+2)) that
    Gaussian applies is evaluated as ⟨AΨ|Ŝ²|AΨ⟩ / ⟨AΨ|AΨ⟩, which needs the spin
    moments ⟨Ŝ²⟩, ⟨Ŝ⁴⟩, ⟨Ŝ⁶⟩ of the determinant.

    For a broken-symmetry singlet (Sz = 0) these moments have exact closed forms
    in the pair spin defects t_i = 1 - λ_i² (A = Σt, B = Σt², C = Σt³):
        ⟨Ŝ²⟩ = A
        ⟨Ŝ⁴⟩ = 2A + 2A² - 2B
        ⟨Ŝ⁶⟩ = 4A + 16A² + 6A³ - 16B - 18AB + 12C
    so every pair (including weakly broken ones, which carry the higher-multiplet
    content the "after" value is sensitive to) is included with no truncation.
    For higher multiplicities (Sz > 0) the strongly broken pairs are coupled
    explicitly with the Sz-core and the same annihilator is applied.

    Parameters
    ----------
    mo_a, mo_b : ndarray
        Occupied alpha / beta MO coefficients (in the basis of overlap S).
    S : ndarray
        AO overlap matrix for those coefficients.
    n_a, n_b : int
        Numbers of alpha and beta electrons.
    tol : float, optional
        Minimum spin defect for a pair to be coupled explicitly, used only for
        the Sz > 0 path (default 5e-4).
    max_pairs : int, optional
        Cap on explicitly coupled pairs for the Sz > 0 path (default 6).

    Returns
    -------
    (s2_before, s2_after) : tuple of float
    """
    Sz = (n_a - n_b) / 2.0
    Delta = mo_a.T @ S @ mo_b
    sv = np.linalg.svd(Delta, compute_uv=False)
    lam2 = np.clip(sv ** 2, 0.0, 1.0)
    t = 1.0 - lam2                      # per-pair spin defect
    s2_before = Sz * (Sz + 1) + float(t.sum())

    a = Sz * (Sz + 1)            # target eigenvalue
    b = (Sz + 1) * (Sz + 2)      # first contaminant eigenvalue

    if t.sum() < 1e-12:
        return s2_before, a         # no contamination to annihilate

    # -- Singlet: exact closed-form moments over ALL pairs (matches Gaussian) --
    if abs(Sz) < 1e-9:
        A = float(t.sum()); B = float((t ** 2).sum()); C = float((t ** 3).sum())
        s2 = A
        s4 = 2 * A + 2 * A ** 2 - 2 * B
        s6 = 4 * A + 16 * A ** 2 + 6 * A ** 3 - 16 * B - 18 * A * B + 12 * C
        s2_after = (s6 - 2 * b * s4 + b * b * s2) / (s4 - 2 * b * s2 + b * b)
        return s2_before, float(s2_after)

    # -- Sz > 0: couple strongly broken pairs with the Sz-core, apply annihilator --
    sig = np.sort(t[t > tol])[::-1][:max_pairs]
    if sig.size == 0:
        return s2_before, a

    def _spin_ops(two_s):
        d = two_s + 1
        ms = np.array([two_s / 2.0 - i for i in range(d)])
        sz = np.diag(ms)
        sp = np.zeros((d, d))
        for i in range(1, d):
            m = ms[i]
            sp[i - 1, i] = np.sqrt((two_s / 2.0) * (two_s / 2.0 + 1) - m * (m + 1))
        return sz, sp

    ops_z, ops_p, vecs = [], [], []
    two_s_core = int(round(2 * Sz))
    cz, cp = _spin_ops(two_s_core)
    ops_z.append(cz); ops_p.append(cp)
    cvec = np.zeros(two_s_core + 1); cvec[0] = 1.0
    vecs.append(cvec)
    sq2 = np.sqrt(2.0)
    pair_z = np.diag([0.0, -1.0, 0.0, 1.0])
    pair_p = np.zeros((4, 4)); pair_p[2, 1] = sq2; pair_p[3, 2] = sq2
    for ti in sig:
        qi = ti / 2.0
        ops_z.append(pair_z); ops_p.append(pair_p)
        v = np.zeros(4); v[0] = np.sqrt(max(1 - qi, 0.0)); v[2] = np.sqrt(qi)
        vecs.append(v)

    def _kron(mats):
        out = np.array([[1.0]])
        for m in mats:
            out = np.kron(out, m)
        return out

    dims = [o.shape[0] for o in ops_z]
    n_sub = len(ops_z)
    Sz_tot = np.zeros((int(np.prod(dims)),) * 2)
    Sp_tot = np.zeros_like(Sz_tot)
    for i in range(n_sub):
        mats = [np.eye(dims[j]) for j in range(n_sub)]; mats[i] = ops_z[i]
        Sz_tot += _kron(mats)
        mats = [np.eye(dims[j]) for j in range(n_sub)]; mats[i] = ops_p[i]
        Sp_tot += _kron(mats)
    S2 = Sz_tot @ Sz_tot + 0.5 * (Sp_tot @ Sp_tot.T + Sp_tot.T @ Sp_tot)

    psi = np.array([1.0])
    for v in vecs:
        psi = np.kron(psi, v)
    psi /= np.linalg.norm(psi)

    Aop = (S2 - b * np.eye(S2.shape[0])) / (a - b)
    Apsi = Aop @ psi
    s2_after = float((Apsi @ S2 @ Apsi) / (Apsi @ Apsi))
    return s2_before, s2_after


# =============================================================================
# AO Reordering: PySCF → Gaussian Convention
# =============================================================================

# ECP core shells in the order standard cores are built (cumulative electrons:
# 2, 4, 10, 12, 18, 28, 30, 36, 46, 60, 62, 68, 78, 92, 94, 100, 110).
_CORE_SHELL_ORDER = [(1, 0), (2, 0), (2, 1), (3, 0), (3, 1), (3, 2), (4, 0), (4, 1),
                     (4, 2), (4, 3), (5, 0), (5, 1), (5, 2), (5, 3), (6, 0), (6, 1), (6, 2)]


def _ecp_core_shells(n_core):
    """Number of core shells per l removed by an ECP with n_core electrons
    (28 -> {0: 3, 1: 2, 2: 1}: 1s2s3s, 2p3p, 3d)."""
    removed = {}
    left = int(n_core)
    for n, l in _CORE_SHELL_ORDER:
        size = 2 * (2 * l + 1)
        if left < size:
            break
        left -= size
        removed[l] = removed.get(l, 0) + 1
    return removed


def _label_offsets(mol_min):
    """Per-atom {l: offset} to add to PySCF's shell numbering so that a
    valence-only minimal basis (MINAO on an ECP atom) is labelled with the
    physical principal quantum numbers (Rh with a 28-electron core: 4S, 5S,
    4P, 4D instead of 1S, 2S, 2P, 3D). The offset is the number of core shells
    of that l removed by the ECP, capped by the number of shells the atom's
    basis actually lacks relative to the all-electron minimal set, so an
    all-electron minimal basis on an ECP atom is left unchanged."""
    cores = getattr(mol_min, '_minpop_ecp_core', None)
    if not cores or not any(cores):
        return {}
    try:
        from anorcc_basis_pyscf import MB_COUNTS
    except ImportError:
        return {}
    from pyscf.data import elements
    ao = mol_min.ao_labels(fmt=None)
    offsets = {}
    for i in range(mol_min.natm):
        if not cores[i]:
            continue
        elem = ''.join(c for c in mol_min.atom_symbol(i) if c.isalpha())
        ae = MB_COUNTS.get(elem)
        if ae is None:
            continue
        present = {}
        for lab in ao:
            if lab[0] != i:
                continue
            name = lab[2].strip()
            k = 0
            while k < len(name) and name[k].isdigit():
                k += 1
            present.setdefault(name[:k + 1], 0)
        counts = {}
        for sh in present:
            l = {'s': 0, 'p': 1, 'd': 2, 'f': 3, 'g': 4}[sh[-1]]
            counts[l] = counts.get(l, 0) + 1
        core = _ecp_core_shells(cores[i])
        offsets[i] = {l: max(0, min(core.get(l, 0), (ae[l] if l < len(ae) else 0) - counts.get(l, 0)))
                      for l in counts}
    return offsets


def _generic_ao_order(mol_min, label_style='gaussian'):
    """
    Order AOs by atom, then principal quantum number n, then l, then m in
    Gaussian's spherical sequence (0, +1, -1, +2, -2, ...), with labels in
    Gaussian's shell-counter convention (counter advances on every s, d, f
    shell; a p shell shares its counter with the s of the same n).

    This reproduces Gaussian's STO-3G order and labels for every case the
    hand-written rules in _parse_ao_label cover (checked on H2O, CH2, CO, ScH)
    and extends them to any minimal basis whose PySCF labels carry the
    physical n, which a minimal basis with no shell gaps does (ANO-RCC-MB:
    Pt 1s..6s, 2p..6p, 3d..5d, 4f). Used for ANO-RCC-MB; the STO-3G path
    keeps the validated legacy routine.

    Returns
    -------
    reorder : list of int
    labels : list of tuple
        (lbl, shell, ang, subtype, m_val) as _get_gaussian_ao_order returns.
    """
    ao_labels = mol_min.ao_labels(fmt=None)
    offsets = _label_offsets(mol_min)
    lmap = {'s': 0, 'p': 1, 'd': 2, 'f': 3, 'g': 4, 'h': 5}
    p_m = {'x': 1, 'y': -1, 'z': 0}
    p_rank = {'x': 0, 'y': 1, 'z': 2}

    def m_rank(m):
        return 0 if m == 0 else 2 * abs(m) - (1 if m > 0 else 0)

    parsed = []
    for i, lbl in enumerate(ao_labels):
        orb = lbl[2].lower().strip()
        cart = (lbl[3] or '').lower().strip()
        n = int(re.match(r'(\d+)', orb).group(1))
        l = lmap[orb[len(str(n))]]
        n += offsets.get(lbl[0], {}).get(l, 0)
        if l == 0:
            m, sub, rank = 0, '', 0
        elif l == 1:
            m, sub, rank = p_m[cart], cart, p_rank[cart]
        elif l == 2:
            m, sub = _parse_spherical_d(cart)
            rank = m_rank(m)
        else:
            m = int(cart.replace(' ', '')) if cart not in ('', '0') else 0
            sub, rank = 'sph', m_rank(m)
        parsed.append((lbl[0], n, l, rank, i, lbl, m, sub))
    parsed.sort(key=lambda t: t[:4])

    # Gaussian shell counter per atom: every distinct (n, l) block gets the
    # next number, except a p block, which reuses the number of the s block
    # with the same n (1S, 2S 2P, 3S 3P, 4D, 5S 5P, ...).
    shell_of = {}
    counter, prev_atom = 0, None
    for atom, n, l, _rank, _i, _lbl, _m, _sub in parsed:
        if atom != prev_atom:
            counter, prev_atom = 0, atom
        if (atom, n, l) in shell_of:
            continue
        if l == 1 and (atom, n, 0) in shell_of:
            shell_of[(atom, n, l)] = shell_of[(atom, n, 0)]
        else:
            counter += 1
            shell_of[(atom, n, l)] = counter

    if label_style == 'physical':
        shell_of = {k: k[1] for k in shell_of}

    new_order = [t[4] for t in parsed]
    new_labels = [(lbl, shell_of[(atom, n, l)], l, sub, m)
                  for atom, n, l, _rank, _i, lbl, m, sub in parsed]
    return new_order, new_labels


def _get_gaussian_ao_order(mol_min, minimal_basis='sto-3g'):
    """
    Generate permutation to reorder AOs from PySCF to Gaussian convention.
    
    Gaussian orders AOs by increasing shell number within each atom:
        1S, 2S, 2PX, 2PY, 2PZ, 3S, 3PX, 3PY, 3PZ, 4D, ...
    
    For d-orbitals:
        - Cartesian (STO-3G*): XX, YY, ZZ, XY, XZ, YZ
        - Spherical (STO-3G):  D0, D+1, D-1, D+2, D-2
    """
    from pyscf.data import elements
    
    if _mbs_key(minimal_basis) in ('anorccmb', 'sto3gano') or '+' in str(minimal_basis):
        style = LABEL_STYLE or ('gaussian' if _mbs_key(minimal_basis) == 'sto3gano'
                                else 'physical')
        return _generic_ao_order(mol_min, style)
    
    ao_labels = mol_min.ao_labels(fmt=None)
    
    # Detect if using Cartesian d-orbitals
    has_cartesian_d = any(
        lbl[3].lower() in ('xx', 'yy', 'zz')
        for lbl in ao_labels if 'd' in lbl[2].lower()
    )
    
    # Group AOs by atom
    atom_aos = {}
    for i, lbl in enumerate(ao_labels):
        atom_idx = lbl[0]
        atom_aos.setdefault(atom_idx, []).append((i, lbl))
    
    new_order, new_labels = [], []
    
    for atom_idx in sorted(atom_aos.keys()):
        aos = atom_aos[atom_idx]
        orbitals = [lbl[2].lower() for _, lbl in aos]
        
        # Determine atom properties
        elem = ''.join(c for c in aos[0][1][1] if c.isalpha())
        try:
            z = elements.charge(elem)
        except (KeyError, ValueError):
            z = 0
        
        is_transition_metal = '3d' in orbitals and '4s' in orbitals
        is_5th_period = z > 36
        is_second_row = elem in SECOND_ROW_ELEMENTS  # Na-Ar with STO-3G*
        
        # Track orbital block indices for 5th period elements
        d_orbital_index = 0
        s_count, p_count = {}, {}
        
        parsed = []
        for i, lbl in aos:
            orb = lbl[2].lower().strip()
            cart = (lbl[3] or '').lower().strip()
            
            # Determine d-block index
            d_block_idx = 0
            if is_5th_period and 'd' in orb:
                d_block_idx = 1 if d_orbital_index >= 5 else 0
                d_orbital_index += 1
            
            # Track s/p shell occurrences
            sp_block_idx = 0
            if is_5th_period:
                if 's' in orb and not cart:
                    s_count[orb] = s_count.get(orb, 0) + 1
                    sp_block_idx = s_count[orb] - 1
                elif 'p' in orb and cart in ('x', 'y', 'z'):
                    p_count[orb] = p_count.get(orb, 0) + 1
                    sp_block_idx = (p_count[orb] - 1) // 3
            
            shell, ang, subtype, m_val = _parse_ao_label(
                orb, cart, is_transition_metal, has_cartesian_d,
                is_5th_period, d_block_idx, sp_block_idx, is_second_row
            )
            parsed.append((i, lbl, shell, ang, subtype, m_val))
        
        parsed.sort(key=lambda x: _ao_sort_key(x, has_cartesian_d))
        
        for item in parsed:
            new_order.append(item[0])
            new_labels.append(item[1:])
    
    return new_order, new_labels


def _parse_ao_label(orb, cart, is_transition_metal, has_cartesian_d,
                    is_5th_period=False, d_block_idx=0, sp_block_idx=0, is_second_row=False):
    """
    Parse PySCF orbital label to Gaussian shell number and angular type.
    
    Shell Numbering Convention
    --------------------------
    For 5th period elements (e.g., Sn), Gaussian uses:
        1S, 2S, 2P, 3S, 3P, 4D, 5S, 5P, 6D, 7S, 7P
    
    For second-row elements with STO-3G* (d-polarization):
        d-orbitals are shell 4 (4XX, 4YY, etc.)
    
    Returns
    -------
    tuple : (shell, angular_momentum, subtype, m_value)
    """
    # S orbitals
    if 's' in orb and not cart:
        shell = int(orb[0]) if orb[0].isdigit() else 1
        
        if is_transition_metal:
            if orb.startswith('4s'): shell = 5
            elif orb.startswith('5s'): shell = 6
        
        if is_5th_period:
            if orb.startswith('6s') or shell == 6:
                shell = 7
            elif sp_block_idx > 0:
                shell += 3
        
        return (shell, 0, '', 0)
    
    # P orbitals
    if 'p' in orb and cart in ('x', 'y', 'z'):
        shell = int(orb[0]) if orb[0].isdigit() else 2
        
        if is_transition_metal:
            if orb.startswith('4p'): shell = 5
            elif orb.startswith('5p'): shell = 6
        
        if is_5th_period:
            if orb.startswith('6p') or shell == 6:
                shell = 7
            elif sp_block_idx > 0:
                shell += 3
        
        m_val = {'x': 1, 'y': -1, 'z': 0}[cart]
        return (shell, 1, cart, m_val)
    
    # D orbitals (Cartesian)
    if has_cartesian_d and cart in ('xx', 'xy', 'xz', 'yy', 'yz', 'zz'):
        shell = int(orb[0]) if orb[0].isdigit() else 4
        
        # For second-row elements with STO-3G*, d-polarization is always shell 4
        if is_second_row:
            shell = 4
        elif is_transition_metal:
            if orb == '3d': shell = 4
            elif orb == '4d': shell = 5
        elif is_5th_period:
            shell = 4 if d_block_idx == 0 else 6
        
        return (shell, 2, cart, 0)
    
    # D orbitals (Spherical)
    if 'd' in orb:
        shell = int(orb[0]) if orb[0].isdigit() else 4
        
        # STO-3G* d-polarization on second-row atoms is shell 4 in Gaussian
        # (PySCF labels it '3d'); mirror the Cartesian branch above.
        if is_second_row:
            shell = 4
        elif is_transition_metal:
            if orb == '3d': shell = 4
            elif orb == '4d': shell = 5
        elif is_5th_period:
            shell = 4 if d_block_idx == 0 else 6
        
        m_val, subtype = _parse_spherical_d(cart)
        return (shell, 2, subtype, m_val)
    
    return (1, 0, '', 0)


def _parse_spherical_d(cart):
    """Parse spherical d-orbital label to (m_value, subtype)."""
    cart_clean = cart.lower().strip()
    
    if cart_clean in ('0', '+0', '-0'):
        return (0, 'sph')
    if cart_clean in ('+1', '1'):
        return (1, 'sph')
    if cart_clean == '-1':
        return (-1, 'sph')
    if cart_clean in ('+2', '2'):
        return (2, 'sph')
    if cart_clean == '-2':
        return (-2, 'sph')
    
    label_map = {
        'z^2': 0, 'z2': 0, '3z2-r2': 0, 'd0': 0, 'dz2': 0,
        'xz': 1, 'd+1': 1, 'dxz': 1,
        'yz': -1, 'd-1': -1, 'dyz': -1,
        'x2-y2': 2, 'x2y2': 2, 'd+2': 2, 'dx2y2': 2,
        'xy': -2, 'd-2': -2, 'dxy': -2,
    }
    
    return (label_map.get(cart_clean, 0), 'sph')


def _ao_sort_key(parsed_ao, has_cartesian_d):
    """Generate sort key for Gaussian AO ordering."""
    _, _, shell, ang, subtype, m_val = parsed_ao
    
    if ang == 1:
        return (shell, ang, {'x': 0, 'y': 1, 'z': 2}.get(subtype, 0))
    
    if ang == 2:
        if has_cartesian_d:
            cart_order = {'xx': 0, 'yy': 1, 'zz': 2, 'xy': 3, 'xz': 4, 'yz': 5}
            return (shell, ang, cart_order.get(subtype, 0))
        else:
            sph_order = {0: 0, 1: 1, -1: 2, 2: 3, -2: 4}
            return (shell, ang, sph_order.get(m_val, 0))
    
    return (shell, ang, 0)


def _convert_label_to_gaussian(lbl_info):
    """Convert parsed AO label to Gaussian format string."""
    lbl, shell, ang, subtype, m_val = lbl_info
    atom_idx, elem = lbl[0], lbl[1]
    
    if ang == 0:
        return (atom_idx, elem, f'{shell}S', '')
    
    if ang == 1:
        return (atom_idx, elem, f'{shell}P{subtype.upper()}', '')
    
    if ang == 2:
        if subtype in ('xx', 'xy', 'xz', 'yy', 'yz', 'zz'):
            return (atom_idx, elem, f'{shell}{subtype.upper()}', '')
        m_str = ' 0' if m_val == 0 else (f'+{m_val}' if m_val > 0 else str(m_val))
        return (atom_idx, elem, f'{shell}D{m_str}', '')
    
    if ang >= 3:  # F, G, H: always spherical, Gaussian prints nF 0, nF+1, ...
        letter = 'FGHI'[ang - 3]
        m_str = ' 0' if m_val == 0 else (f'+{m_val}' if m_val > 0 else str(m_val))
        return (atom_idx, elem, f'{shell}{letter}{m_str}', '')
    
    return (atom_idx, elem, lbl[2].upper(), '')


# =============================================================================
# Matrix Operations
# =============================================================================

def _reorder_matrix(matrix, order):
    """Reorder rows and columns of a symmetric matrix."""
    return matrix[np.ix_(order, order)]


def _mulliken_pop_matrix(dm, S):
    """Compute Mulliken population matrix: P = D ⊙ S (element-wise)."""
    return dm * S


def _condense_to_atoms(pop_matrix, ao_labels):
    """Sum orbital populations to atom-atom matrix."""
    n_atoms = max(lbl[0] for lbl in ao_labels) + 1
    condensed = np.zeros((n_atoms, n_atoms))
    
    for i, li in enumerate(ao_labels):
        for j, lj in enumerate(ao_labels):
            condensed[li[0], lj[0]] += pop_matrix[i, j]
    
    return condensed


# =============================================================================
# Output Formatting (Gaussian-Compatible)
# =============================================================================

def _format_value(v):
    """Format float in Gaussian's style: 10 chars total, 5 decimal places."""
    if abs(v) < 0.000005:
        # Near-zero values: use explicit sign formatting (10 chars total)
        sign = '-' if v < 0 else ' '
        return f"  {sign}0.00000"
    return f"{v:10.5f}"


def _print_density_matrix(dm, ao_labels, title, prefix="     "):
    """Print density matrix in Gaussian format with column blocks of 5."""
    n = len(ao_labels)
    print(f"{prefix}{title}:")
    
    for col_start in range(0, n, 5):
        col_end = min(col_start + 5, n)
        
        # Column headers - 18 spaces + column numbers in 10-char fields
        header = " " * 18 + "".join(f"{c+1:10d}" for c in range(col_start, col_end))
        print(header)
        
        # Matrix rows
        for row in range(col_start, n):
            lbl = ao_labels[row]
            
            # Format: [row#:4][space:1][atom#:4 or spaces][elem:3 or spaces][orb:9]
            # Total = 21 chars before values
            if row == col_start or lbl[0] != ao_labels[row-1][0]:
                # Row with atom number: "   1 1   C  1S       "
                atom_num = f"{lbl[0]+1:<4d}"  # left-aligned, 4 chars
                elem = f"{lbl[1]:<3s}"        # left-aligned, 3 chars
                orb = f"{lbl[2]:<9s}"         # left-aligned, 9 chars
                row_str = f"{row+1:4d} {atom_num}{elem}{orb}"
            else:
                # Continuation row: "   2        2S       "
                orb = f"{lbl[2]:<9s}"
                row_str = f"{row+1:4d}        {orb}"
            
            values = "".join(
                _format_value(dm[row, c]) for c in range(col_start, min(row + 1, col_end))
            )
            print(f"{row_str}{values}")


def _print_gross_populations(gross, ao_labels):
    """Print gross orbital populations table."""
    print("     MBS Gross orbital populations:")
    print("                         Total     Alpha     Beta      Spin")
    
    for i, lbl in enumerate(ao_labels):
        # Same 21-char row label format as density matrix
        if i == 0 or lbl[0] != ao_labels[i-1][0]:
            atom_num = f"{lbl[0]+1:<4d}"
            elem = f"{lbl[1]:<3s}"
            orb = f"{lbl[2]:<9s}"
            row_str = f"{i+1:4d} {atom_num}{elem}{orb}"
        else:
            orb = f"{lbl[2]:<9s}"
            row_str = f"{i+1:4d}        {orb}"
        
        vals = "".join(f"{v:10.5f}" for v in gross[i])
        print(f"{row_str}{vals}")


def _print_atomic_matrix(matrix, mol, title):
    """Print atom-condensed matrix in Gaussian format."""
    n = mol.natm
    print(f"          MBS {title}:")
    
    for col_start in range(0, n, 6):
        col_end = min(col_start + 6, n)
        
        # Header: 5 spaces + column numbers in 11-char fields
        header = " " * 5 + "".join(f"{c+1:11d}" for c in range(col_start, col_end))
        print(header)
        
        for row in range(n):
            sym = mol.atom_symbol(row)
            vals = "".join(f"{matrix[row, c]:11.6f}" for c in range(col_start, col_end))
            # Row format: 6-char number + 2 spaces + 2-char element + values (no extra space)
            print(f"{row+1:6d}  {sym:2s}{vals}")


def _print_mulliken_summary(charges, spins, mol):
    """Print Mulliken charges and spin populations summary."""
    print(" MBS Mulliken charges and spin densities:")
    print("               1          2")
    
    for i in range(mol.natm):
        sym = mol.atom_symbol(i)
        print(f"{i+1:6d}  {sym:2s}{charges[i]:11.6f}{spins[i]:11.6f}")
    
    total_charge = np.sum(charges)
    total_spin = np.sum(spins)
    print(f" Sum of MBS Mulliken charges = {total_charge:10.5f} {total_spin:10.5f}")



def _print_symmetry_section(results, mol_min, shells=False):
    """Sigma/pi/delta/phi resolution of the MBS bond populations, printed in the
    same formats as the matrices above. Called after the Mulliken summary."""
    from bond_symmetry import class_matrices
    reorder = results['ao_reorder']
    inv = np.argsort(reorder)                      # Gaussian order -> PySCF order
    S_min = mol_min.intor('int1e_ovlp')
    ao_raw = results['ao_labels_raw']
    ao_lab = results['ao_labels']

    # Shell label as printed above (e.g. '9D' or '5D' depending on -label-style),
    # keyed by (atom index, PySCF shell name such as '5d').
    shell_name = {}
    for raw, lab in zip(ao_raw, ao_lab):
        pyscf_shell = raw[0][2].strip()
        i = 0
        while i < len(pyscf_shell) and pyscf_shell[i].isdigit():
            i += 1
        pyscf_shell = pyscf_shell[:i + 1]
        printed = re.match(r'(\d+[A-Z])', lab[2]).group(1)
        shell_name.setdefault((raw[0][0], pyscf_shell), printed)

    print("MBS bond populations resolved by symmetry (\u03c3, \u03c0, \u03b4, \u03c6)")
    print("=" * 60)

    for dm_key, what, spin in (('dm_total', 'Condensed to atoms (all electrons)', False),
                               ('dm_spin', 'Atomic-Atomic Spin Densities', True)):
        dm = results[dm_key][np.ix_(inv, inv)]
        if spin and np.abs(dm).max() < 1e-8:
            print(" Spin density is zero; no spin-density bond populations.")
            continue
        cm = class_matrices(mol_min, dm, S_min)
        # Put shells in the aufbau order used by the orbital matrices above
        # (first appearance of each (atom, shell) in the reordered AO list).
        printed_order = []
        for raw in ao_raw:
            name = raw[0][2].strip()
            i = 0
            while i < len(name) and name[i].isdigit():
                i += 1
            key = f"{mol_min.atom_symbol(raw[0][0])}{raw[0][0] + 1} {name[:i + 1]}"
            if key not in printed_order:
                printed_order.append(key)
        perm = [cm['shell_labels'].index(k) for k in printed_order]
        cm['shell_labels'] = [cm['shell_labels'][p] for p in perm]
        for c in cm['classes']:
            cm['shell'][c] = cm['shell'][c][np.ix_(perm, perm)]
        labels = []
        for lab in cm['shell_labels']:
            sym_idx, sh = lab.split()
            atom_idx = int(re.sub(r'[A-Za-z]', '', sym_idx)) - 1
            labels.append((atom_idx, mol_min.atom_symbol(atom_idx),
                           shell_name.get((atom_idx, sh), sh.upper()), ''))
        for c in cm['classes']:
            cname = _SYMMETRY_SYMBOL[c]
            if np.abs(cm['atom'][c]).max() < 5e-7:
                print(f"          MBS {cname} {what}: all entries below 1e-6")
                continue
            if shells:
                _print_density_matrix(cm['shell'][c], labels,
                                      f"{cname} bond populations condensed to shells"
                                      + (" (spin density)" if spin else " (all electrons)"),
                                      prefix="    ")
            _print_atomic_matrix(cm['atom'][c], mol_min, f"{cname} {what}")
        print()


def _print_results(results, mol_min, n_alpha, n_beta):
    """Print complete analysis in Gaussian format."""
    ao_labels = results['ao_labels']
    
    print(f"UHF orbital structure: {n_alpha} alpha, {n_beta} beta")
    print(" Annihilation of the first spin contaminant:")
    print(f" S**2 before annihilation {results['s2_before_annihilation']:8.4f},"
          f"   after {results['s2_after_annihilation']:8.4f}")
    print("=" * 60)
    print("MinPop Analysis (UHF)")
    print("=" * 60)
    
    _print_density_matrix(results['dm_alpha'], ao_labels, "Alpha  MBS Density Matrix")
    _print_density_matrix(results['dm_beta'], ao_labels, "Beta  MBS Density Matrix")
    _print_density_matrix(results['pop_total'], ao_labels,
                         "Full MBS Mulliken population analysis", prefix="    ")
    
    _print_gross_populations(results['gross_orbital_pop'], ao_labels)
    _print_atomic_matrix(results['condensed_to_atoms'], mol_min,
                        "Condensed to atoms (all electrons)")
    _print_atomic_matrix(results['spin_atomic'], mol_min,
                        "Atomic-Atomic Spin Densities")
    _print_mulliken_summary(results['mulliken_charges'],
                           results['spin_populations'], mol_min)
    
    print("=" * 60)
    if BOND_SYMMETRY:
        _print_symmetry_section(results, mol_min, shells=BOND_SYMMETRY_SHELLS)
        print("=" * 60)


# =============================================================================
# File I/O
# =============================================================================

def _read_xyz(filename):
    """Parse XYZ file to PySCF atom specification string."""
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    n_atoms = int(lines[0].strip())
    atoms = []
    
    for line in lines[2:2 + n_atoms]:
        parts = line.split()
        atoms.append(f"{parts[0]} {parts[1]} {parts[2]} {parts[3]}")
    
    return "; ".join(atoms)


# Import the standard-orientation module at most once.
_STD_ORIENT_MOD = None


def _load_std_orientation(search_dir=None):
    """
    Import gaussian_standard_orientation (Gaussian's 'Standard orientation'
    frame). A normal import is tried first, then a fallback that loads
    gaussian_standard_orientation.py from the current directory or next to this
    script, mirroring how the custom basis modules are located.
    """
    global _STD_ORIENT_MOD
    if _STD_ORIENT_MOD is not None:
        return _STD_ORIENT_MOD
    try:
        import gaussian_standard_orientation as _gso
        _STD_ORIENT_MOD = _gso
        return _gso
    except ImportError:
        pass
    for d in _basis_search_dirs(search_dir):
        path = os.path.join(d, "gaussian_standard_orientation.py")
        if os.path.isfile(path):
            spec = importlib.util.spec_from_file_location(
                "gaussian_standard_orientation", path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            _STD_ORIENT_MOD = module
            return module
    raise ImportError(
        "gaussian_standard_orientation.py not found. Place it next to this "
        "script or in the current directory, or pass -no-std-orient to skip "
        "standard reorientation.")


def _format_standard_orientation(Z, coords):
    """
    Render coordinates as Gaussian 16's "Standard orientation:" table.

    Reproduces Gaussian's fixed layout - center number (I7), atomic number
    (I11), atomic type (I12, always 0), then 4X and X/Y/Z in Angstroms as F12.6
    (rounded to 6 decimals) - so the block can be diffed directly against a
    Gaussian log. Only the printed values are rounded; the coordinates handed to
    PySCF keep full precision. Components that round to zero are shown unsigned.
    """
    sep = " " + "-" * 69
    lines = [
        " " * 25 + "Standard orientation:" + " " * 24,
        sep,
        " Center     Atomic      Atomic             Coordinates (Angstroms)",
        " Number     Number       Type             X           Y           Z",
        sep,
    ]
    for center, (z, r) in enumerate(zip(Z, coords), start=1):
        xyz = [0.0 if abs(float(v)) < 5e-7 else float(v) for v in r]
        lines.append(f"{center:7d}{int(round(z)):11d}{0:12d}    "
                     f"{xyz[0]:12.6f}{xyz[1]:12.6f}{xyz[2]:12.6f}")
    lines.append(sep)
    return "\n".join(lines)


def _standardize_atom_str(atom_str, search_dir=None):
    """
    Rotate a PySCF atom-spec string into Gaussian's standard orientation.

    The population analysis reproduces Gaussian's minimal-basis density-matrix
    and per-AO gross-population printouts, which are frame-dependent: p/d axes
    rotate with the molecule, so the same wavefunction prints different matrix
    elements in different orientations. Running PySCF in Gaussian's
    standard-orientation frame - center of nuclear charge, charge-weighted
    principal axes - makes those printouts line up. Atomic charges and spins are
    rotationally invariant, so the condensed-to-atoms numbers are unchanged; only
    the AO-resolved output is affected.

    Parameters
    ----------
    atom_str : str
        PySCF atom spec "El x y z; El x y z; ..." in Angstrom (from _read_xyz).
    search_dir : str, optional
        Extra directory to look in for gaussian_standard_orientation.py.

    Returns
    -------
    atom_str_std : str
        The same atoms, order preserved, in Gaussian's standard orientation.
    orient_table : str
        The geometry rendered as Gaussian's "Standard orientation:" table
        (F12.6), ready to print; the caller decides where/whether to show it.
    """
    from pyscf.data import elements

    gso = _load_std_orientation(search_dir)

    symbols, Z, coords = [], [], []
    for entry in atom_str.replace(';', '\n').split('\n'):
        parts = entry.split()
        if not parts:
            continue
        sym = parts[0]
        symbols.append(sym)
        elem = ''.join(c for c in sym if c.isalpha())
        try:
            Z.append(elements.charge(elem))
        except (KeyError, ValueError):
            Z.append(int(sym))          # numeric atomic number in the xyz
        coords.append([float(parts[1]), float(parts[2]), float(parts[3])])

    coords_std, _used_on_axis = gso.to_standard_orientation(
        np.asarray(Z, dtype=float), np.asarray(coords, dtype=float),
        warn=False)

    orient_table = _format_standard_orientation(Z, coords_std)
    atoms = [f"{sym} {r[0]:.10f} {r[1]:.10f} {r[2]:.10f}"
             for sym, r in zip(symbols, coords_std)]
    return "; ".join(atoms), orient_table


def _get_ecp_dict(basis, atom_str):
    """Build element-specific ECP dictionary for heavy elements (Z > 36)."""
    from pyscf.data import elements
    
    basis_normalized = basis.lower().replace('-', '').replace('_', '')
    
    uses_ecp = any(
        ecp_basis.replace('-', '') in basis_normalized
        for ecp_basis in ECP_BASIS_SETS
    )
    
    if not uses_ecp:
        return None
    
    ecp_dict = {}
    for line in atom_str.replace(';', '\n').split('\n'):
        parts = line.split()
        if parts:
            elem = ''.join(c for c in parts[0] if c.isalpha())
            try:
                z = elements.charge(elem)
                if z > 36:
                    ecp_dict[elem] = 'def2-ecp'   # one ECP for every def2 basis (Rh: ECP28MWB)
            except (KeyError, ValueError):
                pass
    
    return ecp_dict if ecp_dict else None


# =============================================================================
# Main Analysis Functions
# =============================================================================

_AZ_P_RE = re.compile(r'^(\d+)P([XYZ])$')
_AZ_D_RE = re.compile(r'^(\d+)([DFGH])([ +\-]\d)$')  # D and higher


def _az_build_z_rotation(ao_labels, theta):
    """AO-basis operator for a rotation of the molecule by ``theta`` about Z.

    Pairs (nPX,nPY) and (nD+m,nD-m) rotate by m*theta; S, PZ and D 0 are
    invariant. Validated against Gaussian for Si2 (full-matrix residual ~5e-5).
    """
    n = len(ao_labels)
    U = np.eye(n)
    plus, minus = {}, {}
    for i, lab in enumerate(ao_labels):
        atom, _elem, ao, _ = lab
        mP = _AZ_P_RE.match(ao)
        if mP:
            shell, axis = int(mP.group(1)), mP.group(2)
            if axis == 'X':
                plus[(atom, shell, 1)] = i
            elif axis == 'Y':
                minus[(atom, shell, 1)] = i
            continue
        mD = _AZ_D_RE.match(ao)
        if mD:
            # key carries l so a 4D+1 and a 4F+1 never pair with each other
            shell = (int(mD.group(1)), mD.group(2))
            mv = int(mD.group(3).replace(' ', ''))
            if mv > 0:
                plus[(atom, shell, mv)] = i
            elif mv < 0:
                minus[(atom, shell, -mv)] = i
    for key, a in plus.items():
        b = minus.get(key)
        if b is None:
            continue
        m = key[2]
        c, s = np.cos(m * theta), np.sin(m * theta)
        U[a, a] = c
        U[a, b] = -s
        U[b, a] = s
        U[b, b] = c
    return U


def _az_px_indices(ao_labels):
    return [i for i, lab in enumerate(ao_labels)
            if _AZ_P_RE.match(lab[2]) and lab[2].endswith('X')]


def _az_golden_max(dm, ao_labels, pxi, a, b, tol=1e-13, maxiter=200):
    """Golden-section search for the theta that maximizes total PX pi content.

    The coarse grid argmax is quantized to the scan spacing, which leaves up to
    half a step of misalignment. That is small in angle but puts ~1e-3 into
    individual density-matrix elements and breaks idempotency of the gauge. This
    refines the bracketing interval to machine precision.
    """
    def f(t):
        U = _az_build_z_rotation(ao_labels, t)
        M = U @ dm @ U.T
        return M[np.ix_(pxi, pxi)].sum()
    invphi = (np.sqrt(5.0) - 1.0) / 2.0
    c = b - invphi * (b - a)
    d = a + invphi * (b - a)
    fc, fd = f(c), f(d)
    for _ in range(maxiter):
        if (b - a) < tol:
            break
        if fc > fd:
            b, d, fd = d, c, fc
            c = b - invphi * (b - a)
            fc = f(c)
        else:
            a, c, fc = c, d, fd
            d = a + invphi * (b - a)
            fd = f(d)
    return 0.5 * (a + b)


def _apply_azimuthal_gauge(dm_list, ao_labels, coords_bohr,
                           tol_linear=1e-3, tol_aniso=1e-2, verbose=False):
    """Canonicalize the azimuth about the molecular axis, LINEAR molecules only.

    A linear molecule has every atom on the symmetry axis, so Gaussian's
    standard-orientation rules (which fix x/y from an off-axis "key atom") leave
    the azimuth undetermined. The frame-dependent MBS density/population printouts
    then differ from Gaussian by a rotation about the molecular axis. This rotates
    to a deterministic frame (maximum PX pi content).

    STRICT NO-OP for non-linear molecules (their standard orientation is fully
    fixed) and for cylindrically symmetric linear molecules (no pi anisotropy to
    canonicalize). Returns (dm_list_out, theta); theta is None when no gauge was
    applied.
    """
    dm_total = dm_list[2] if len(dm_list) >= 3 else dm_list[0]
    coords = np.asarray(coords_bohr)
    # (1) linear? In standard orientation the axis is Z: x,y ~ 0 for every atom.
    if coords.shape[0] < 2 or np.abs(coords[:, :2]).max() > tol_linear:
        return dm_list, None
    pxi = _az_px_indices(ao_labels)
    if not pxi:
        return dm_list, None
    # (2) scan the PX content and require genuine pi anisotropy. A rigid rotation
    # turns every spin channel alike, so any channel pins the same angle, but a
    # nearly cylindrical one pins it badly. Weigh the total density against the
    # spin density and canonicalize on whichever is more anisotropic: in a 2-Pi
    # radical such as NO the unpaired electron sits in the spin density, leaving
    # the total density almost isotropic (PX variation ~5e-3, under the gate)
    # while the spin density varies strongly.
    candidates = [dm_total]
    if len(dm_list) >= 4 and dm_list[3] is not None:
        candidates.append(dm_list[3])
    ths = np.linspace(0.0, np.pi, 1441, endpoint=False)
    curves = np.empty((len(ths), len(candidates)))
    for k, t in enumerate(ths):
        U = _az_build_z_rotation(ao_labels, t)
        for c, D in enumerate(candidates):
            M = U @ D @ U.T
            curves[k, c] = M[np.ix_(pxi, pxi)].sum()
    spans = curves.max(axis=0) - curves.min(axis=0)
    best = int(np.argmax(spans))
    if spans[best] < tol_aniso:
        return dm_list, None
    dm_obj = candidates[best]
    g = curves[:, best]
    k = int(np.argmax(g))
    step = ths[1] - ths[0]
    theta = _az_golden_max(dm_obj, ao_labels, pxi,
                           ths[k] - step, ths[k] + step)
    # (3) sign convention: largest inter-center PX-PX coupling >= 0 (fixes 180deg).
    U = _az_build_z_rotation(ao_labels, theta)
    Mt = U @ dm_obj @ U.T
    off = Mt[np.ix_(pxi, pxi)].copy()
    np.fill_diagonal(off, 0.0)
    if off.size and off.flat[np.argmax(np.abs(off))] < 0:
        theta = (theta + np.pi) % (2 * np.pi)
        U = _az_build_z_rotation(ao_labels, theta)
    out = [U @ D @ U.T for D in dm_list]
    if verbose:
        print("Azimuthal gauge (linear molecule): rotated "
              f"{np.degrees(theta):.2f} deg about the molecular axis to the "
              "canonical PX-max frame.")
    return out, theta


def minpop_uhf(mf, verbose=True, azimuthal_gauge=True,
               minimal_basis='sto-3g'):
    """
    Perform MinPop population analysis on a converged UHF calculation.
    
    Parameters
    ----------
    mf : pyscf.scf.uhf.UHF
        Converged UHF mean-field object
    verbose : bool, optional
        Print Gaussian-formatted output (default: True)
    
    Returns
    -------
    results : dict
        Analysis results containing:
        - dm_alpha, dm_beta : Alpha/beta density matrices in minimal basis
        - dm_total, dm_spin : Total and spin density matrices
        - gross_orbital_pop : Per-orbital populations [total, α, β, spin]
        - condensed_to_atoms : Atom-atom population matrix
        - mulliken_charges : Atomic partial charges
        - spin_populations : Atomic spin densities (after annihilation)
        - ao_labels : Minimal basis AO labels
        - s2_before_annihilation : ⟨S²⟩ before projection
        - s2_after_annihilation : ⟨S²⟩ after projection
    
    Examples
    --------
    >>> from pyscf import gto, scf
    >>> mol = gto.M(atom='C 0 0 0; H 0 1 0; H 0 0 1', basis='6-31G*', spin=2)
    >>> mf = scf.UHF(mol).run()
    >>> results = minpop_uhf(mf)
    >>> print(f"Carbon spin: {results['spin_populations'][0]:.4f}")
    """
    mol = mf.mol
    mol_min = _build_minimal_basis_mol(mol, minimal_basis=minimal_basis)
    if verbose:
        tag = {'minao': 'MINAO',
               'anorccmb': 'ANO-RCC-MB (all-electron, generic (n,l,m) order)',
               'sto3gano': f'STO-3G on Z<={MBS_LIGHT_MAX_Z}, ANO-RCC-MB on Z>{MBS_LIGHT_MAX_Z}',
               }.get(_mbs_key(minimal_basis), 'STO-3G (Gaussian MBS convention)')
        if '+' in str(minimal_basis):
            light_name, heavy_name = [s.strip() for s in minimal_basis.split('+')]
            tag = (f"{light_name.upper()} on Z<={MBS_LIGHT_MAX_Z}, "
                   f"{heavy_name.upper()} on Z>{MBS_LIGHT_MAX_Z}")
        print(f" Minimal-basis projection: {tag}")
    
    # Compute overlap matrices
    S_cross = intor_cross('int1e_ovlp', mol_min, mol)
    S_min = mol_min.intor('int1e_ovlp')
    S_min_inv = np.linalg.inv(S_min)
    
    # UHF orbital structure
    mo_occ_a, mo_occ_b = mf.mo_occ
    n_alpha = int(np.sum(mo_occ_a > 0))
    n_beta = int(np.sum(mo_occ_b > 0))
    
    mo_coeff_a, mo_coeff_b = mf.mo_coeff
    
    # Project alpha and beta orbitals independently
    mo_alpha = _project_to_minimal_basis(mo_coeff_a[:, :n_alpha], S_cross, S_min_inv)
    mo_beta = _project_to_minimal_basis(mo_coeff_b[:, :n_beta], S_cross, S_min_inv)
    if verbose:
        _print_projection_completeness('alpha', _projection_completeness(
            mo_coeff_a[:, :n_alpha], S_cross, S_min_inv), float(n_alpha))
        _print_projection_completeness('beta', _projection_completeness(
            mo_coeff_b[:, :n_beta], S_cross, S_min_inv), float(n_beta))
    
    # ⟨S²⟩ before/after annihilation from the FULL wavefunction. Gaussian reports
    # the full-basis values here (not the minimal-basis projection), and single
    # Löwdin annihilation leaves residual higher contaminants (so "after" is not
    # the ideal S(S+1) for a spin-broken singlet).
    S_full = mol.intor('int1e_ovlp')
    s2_before, s2_after = _annihilate_first_contaminant(
        mo_coeff_a[:, :n_alpha], mo_coeff_b[:, :n_beta], S_full, n_alpha, n_beta)
    
    # Build density matrices
    dm_alpha = mo_alpha @ mo_alpha.T
    dm_beta = mo_beta @ mo_beta.T
    dm_total = dm_alpha + dm_beta
    # Spin density is the raw UHF D_alpha - D_beta. For a broken-symmetry singlet
    # this is nonzero (locally spin-polarized) even though the net spin integrates
    # to zero, so it must NOT be zeroed out. The alpha<->beta labeling of an Sz=0
    # singlet is arbitrary, but the deterministic guess-mix reproduces Gaussian's
    # orientation, so the raw sign is kept as-is.
    dm_spin = dm_alpha - dm_beta
    
    # Mulliken population matrices
    pop_alpha = _mulliken_pop_matrix(dm_alpha, S_min)
    pop_beta = _mulliken_pop_matrix(dm_beta, S_min)
    pop_total = _mulliken_pop_matrix(dm_total, S_min)
    pop_spin = _mulliken_pop_matrix(dm_spin, S_min)
    
    
    # Reorder to Gaussian convention
    reorder, ao_labels_raw = _get_gaussian_ao_order(mol_min, minimal_basis)
    
    dm_alpha = _reorder_matrix(dm_alpha, reorder)
    dm_beta = _reorder_matrix(dm_beta, reorder)
    dm_total = _reorder_matrix(dm_total, reorder)
    dm_spin = _reorder_matrix(dm_spin, reorder)
    pop_alpha = _reorder_matrix(pop_alpha, reorder)
    pop_beta = _reorder_matrix(pop_beta, reorder)
    pop_total = _reorder_matrix(pop_total, reorder)
    pop_spin = _reorder_matrix(pop_spin, reorder)
    
    ao_labels = [_convert_label_to_gaussian(lbl) for lbl in ao_labels_raw]

    # Canonical azimuthal gauge for LINEAR molecules. Their standard orientation
    # leaves the azimuth about the molecular axis undetermined (no off-axis key
    # atom), so the frame-dependent MBS density/population printouts otherwise
    # differ from Gaussian by an arbitrary rotation about that axis. Rotate to a
    # deterministic frame; strict no-op for non-linear or cylindrically symmetric
    # systems. When applied, populations are rebuilt from the rotated density
    # (Mulliken is D (x) S with the fixed lab-frame S, so this is exact).
    if azimuthal_gauge:
        dm_list, _gauge_theta = _apply_azimuthal_gauge(
            [dm_alpha, dm_beta, dm_total, dm_spin], ao_labels,
            mol.atom_coords(), verbose=verbose)
    else:
        dm_list, _gauge_theta = [dm_alpha, dm_beta, dm_total, dm_spin], None
    if _gauge_theta is not None:
        dm_alpha, dm_beta, dm_total, dm_spin = dm_list
        S_min_g = _reorder_matrix(S_min, reorder)
        pop_alpha = _mulliken_pop_matrix(dm_alpha, S_min_g)
        pop_beta = _mulliken_pop_matrix(dm_beta, S_min_g)
        pop_total = _mulliken_pop_matrix(dm_total, S_min_g)
        pop_spin = _mulliken_pop_matrix(dm_spin, S_min_g)

    # Gross orbital populations
    gross = np.column_stack([
        np.sum(pop_total, axis=0),
        np.sum(pop_alpha, axis=0),
        np.sum(pop_beta, axis=0),
        np.sum(pop_spin, axis=0)
    ])
    
    # Atomic properties
    condensed = _condense_to_atoms(pop_total, ao_labels)
    spin_atomic = _condense_to_atoms(pop_spin, ao_labels)
    nuclear_charges = np.array([mol.atom_charge(i) for i in range(mol.natm)])
    mulliken_charges = nuclear_charges - np.sum(condensed, axis=1)
    spin_populations = np.sum(spin_atomic, axis=1)
    
    results = {
        'dm_alpha': dm_alpha,
        'dm_beta': dm_beta,
        'dm_total': dm_total,
        'dm_spin': dm_spin,
        'pop_total': pop_total,
        'pop_spin': pop_spin,
        'gross_orbital_pop': gross,
        'condensed_to_atoms': condensed,
        'spin_atomic': spin_atomic,
        'mulliken_charges': mulliken_charges,
        'spin_populations': spin_populations,
        'ao_labels': ao_labels,
        'ao_labels_raw': ao_labels_raw,
        'ao_reorder': reorder,
        's2_before_annihilation': s2_before,
        's2_after_annihilation': s2_after,
    }
    
    if verbose:
        _print_results(results, mol_min, n_alpha, n_beta)
    
    return results


def _init_guess_triplet(mol, mode="density", conv_tol=1e-9, max_cycle=128,
                        verbose=False):
    """Broken-symmetry singlet guess from a converged triplet ROHF.

    For an open-shell singlet (a diradical, or a stretched bond near a
    transition state) the restricted solution is a poor place to start: HOMO
    and LUMO are nearly degenerate, so a mixed guess can relax straight back to
    the symmetric stationary point and the SCF settles on an answer that is not
    the UHF minimum. A triplet ROHF in the SAME basis converges readily and
    puts the two unpaired electrons in two distinct spatial orbitals, so alpha
    and beta differ from the first iteration and the SCF starts inside the
    broken-symmetry basin without any orbital rotation. That is why this needs
    no -guessmix.

    Both modes hand scf.UHF a density, as every other initial guess does.

    mode='density' (default)
        Use the triplet's own alpha and beta density matrices. Simple and
        consistent with the -guess family, but the density carries the
        triplet's spin populations, N/2+1 alpha against N/2-1 beta, so the
        opening Fock describes a configuration with the wrong number of
        electrons in each channel. The singlet UHF reoccupies to N/2 each on
        its first diagonalization.

    mode='orbitals'
        Reoccupy the triplet's spatial orbitals for the singlet: both spins
        take the doubly occupied core, then alpha takes the first singly
        occupied orbital and beta the second. This starts from the correct
        N/2 electrons per spin while still placing the two unpaired electrons
        in different spatial orbitals. It searches a different part of the
        surface and can find broken-symmetry minima the density mode misses.

    Returns
    -------
    dm0 : ndarray, shape (2, nao, nao)
    """
    if mol.spin != 0:
        raise ValueError("triplet guess applies to singlets only "
                         f"(got spin={mol.spin}, i.e. multiplicity "
                         f"{mol.spin + 1})")
    if mode not in ("density", "orbitals"):
        raise ValueError(f"unknown triplet guess mode {mode!r}")
    mol_t = mol.copy()
    mol_t.spin = 2                      # 2S, i.e. a triplet
    mol_t.build(False, False)           # same basis, charge and ECP
    # Keep the scaffold silent. This ROHF is a guess generator, not a result,
    # and letting PySCF print its own convergence verdict would put
    # "SCF not converged." into the report for a run whose real singlet SCF
    # converged perfectly well, which downstream status checks read as a
    # failure. The explicit line below records the triplet's outcome instead.
    mol_t.verbose = 0
    mf_t = scf.ROHF(mol_t)
    mf_t.verbose = 0
    mf_t.max_cycle = max_cycle
    mf_t.conv_tol = conv_tol
    mf_t.kernel()
    if not mf_t.converged:
        # Stop here rather than downstream. An unconverged triplet gives
        # meaningless orbitals and a meaningless density, so whatever the
        # singlet UHF then does with them is not a controlled starting point,
        # and a failure several steps later reads as a singlet problem when the
        # guess was the problem.
        raise MinPopSCFError(
            f"triplet ROHF guess did not converge in {max_cycle} cycles "
            f"(E = {mf_t.e_tot:.9f}). Its orbitals and density are "
            f"meaningless, so the singlet UHF is not attempted. Use "
            f"-guessmix instead, or raise the triplet cycle limit.")
    state = "converged"

    if mode == "density":
        dm0 = np.asarray(mf_t.make_rdm1())
        if dm0.ndim != 3 or dm0.shape[0] != 2:
            raise RuntimeError("triplet ROHF did not return an (alpha, beta) "
                               f"density (got shape {dm0.shape})")
        if verbose:
            print(f"Triplet ROHF guess [density]: triplet ROHF in the same "
                  f"basis ({mol_t.nao} AOs, charge {mol_t.charge}); "
                  f"E(triplet) = {mf_t.e_tot:.9f} ({state})")
            print(f"Triplet ROHF guess [density]: handing its spin-polarized "
                  f"density to the singlet UHF")
        return dm0

    mo = np.asarray(mf_t.mo_coeff)      # one spatial set: (nao, nmo)
    occ_t = np.asarray(mf_t.mo_occ)
    doubly = np.where(occ_t == 2)[0]
    somos = np.where(occ_t == 1)[0]
    if len(somos) != 2:
        raise RuntimeError("triplet ROHF did not give two singly occupied "
                           f"orbitals (found {len(somos)}); cannot build a "
                           "broken-symmetry singlet guess from it")
    nocc = mol.nelectron // 2
    if len(doubly) + 1 != nocc:
        raise RuntimeError(f"triplet ROHF has {len(doubly)} doubly occupied "
                           f"orbitals, expected {nocc - 1} for a singlet with "
                           f"{mol.nelectron} electrons")
    occ_a = np.zeros(mo.shape[1])
    occ_b = np.zeros(mo.shape[1])
    occ_a[doubly] = 1.0
    occ_b[doubly] = 1.0
    occ_a[somos[0]] = 1.0               # one unpaired electron to alpha
    occ_b[somos[1]] = 1.0               # the other to beta
    dm0 = scf.uhf.make_rdm1((mo, mo), (occ_a, occ_b))
    if verbose:
        print(f"Triplet ROHF guess [orbitals]: triplet ROHF in the same "
              f"basis ({mol_t.nao} AOs, charge {mol_t.charge}); "
              f"E(triplet) = {mf_t.e_tot:.9f} ({state})")
        print(f"Triplet ROHF guess [orbitals]: {len(doubly)} doubly occupied "
              f"plus MO {somos[0] + 1} to alpha and MO {somos[1] + 1} to beta "
              f"({nocc} electrons per spin)")
    return dm0


def _init_guess_mixed(mol, mixing_angle_deg=45.0, verbose=False,
                      seed='minao'):
    """
    Broken-symmetry UHF initial guess by HOMO-LUMO mixing (Gaussian Guess=Mix).

    The *initial-guess* Fock (from the atomic-superposition / minao guess) is
    diagonalized ONCE - without converging an SCF - and its frontier orbitals are
    rotated within the HOMO-LUMO space:

        alpha_HOMO = cos(q) * phi_HOMO + sin(q) * phi_LUMO
        beta_HOMO  = cos(q) * phi_HOMO - sin(q) * phi_LUMO

    (the LUMO columns are counter-rotated to keep the set orthonormal). The
    default q = 45 degrees reproduces Gaussian's Guess=Mix coefficients
    (cos45 = sin45 = 0.7071).

    Mixing the *unconverged* guess is essential. If a restricted reference is
    converged first and its HOMO/LUMO are mixed, the density starts at the
    symmetric stationary point and DIIS relaxes straight back to it (the mixing
    is undone). Mixing the crude guess instead - as Gaussian does - starts the
    SCF far from that stationary point, so it descends into the broken-symmetry
    basin on its own, usually without needing a separate stability step.

    Parameters
    ----------
    mol : pyscf.gto.Mole
        Target molecule (charge and spin already set).
    mixing_angle_deg : float, optional
        HOMO-LUMO mixing angle in degrees (default: 45).
    verbose : bool, optional
        Print a one-line note about the mixing (default: False).

    Returns
    -------
    dm0 : ndarray
        Broken-symmetry (alpha, beta) initial density matrix for scf.UHF.
    """
    q = np.deg2rad(mixing_angle_deg)

    # Single diagonalization of the initial-guess Fock (no SCF convergence).
    ref = scf.RHF(mol) if mol.spin == 0 else scf.ROHF(mol)
    s1e = ref.get_ovlp()
    h1e = ref.get_hcore()
    # The seed decides which frontier orbitals get mixed, so it changes
    # which basin the broken-symmetry SCF falls into, not just how fast it
    # gets there. 'huckel' is the useful alternative when the minao-seeded
    # mix stalls or lands on a higher stationary point.
    dm_guess = ref.get_init_guess(mol, key=seed)
    fock = ref.get_fock(h1e=h1e, s1e=s1e,
                        vhf=ref.get_veff(mol, dm_guess), dm=dm_guess)
    mo_energy, mo = ref.eig(fock, s1e)

    n_alpha, n_beta = mol.nelec
    homo = n_alpha - 1
    lumo = n_alpha
    if lumo >= mo.shape[1]:
        raise ValueError("Guess=Mix needs a virtual orbital, but the basis has "
                         "no LUMO for this system (fully occupied).")

    Ca, Cb = mo.copy(), mo.copy()
    c, s = np.cos(q), np.sin(q)
    phi_h, phi_l = mo[:, homo].copy(), mo[:, lumo].copy()
    Ca[:, homo] =  c * phi_h + s * phi_l
    Cb[:, homo] =  c * phi_h - s * phi_l
    Ca[:, lumo] = -s * phi_h + c * phi_l
    Cb[:, lumo] =  s * phi_h + c * phi_l

    occ_a = np.zeros(mo.shape[1]); occ_a[:n_alpha] = 1.0
    occ_b = np.zeros(mo.shape[1]); occ_b[:n_beta] = 1.0

    if verbose:
        print(f"Guess=Mix: rotating HOMO (MO {homo+1}) with LUMO (MO {lumo+1}) "
              f"by {mixing_angle_deg:g} deg to break spin symmetry")

    return scf.uhf.make_rdm1((Ca, Cb), (occ_a, occ_b))


class MinPopSCFError(RuntimeError):
    """The SCF cannot be trusted, so the run stops instead of reporting.

    Raised when an SCF fails to converge, or when Stable=Opt exhausts its
    cycles with the wavefunction still internally unstable. Both conditions
    otherwise produce a complete, plausible-looking MinPop report built on a
    wavefunction that is not a converged minimum, which is worse than no report
    at all for reference data: the numbers look usable and are silently wrong.
    """


def _require_converged(mf, what):
    """Stop the run unless this SCF converged."""
    if not getattr(mf, "converged", False):
        raise MinPopSCFError(
            f"{what} did not converge in {mf.max_cycle} cycles "
            f"(E = {mf.e_tot:.9f}). Populations from an unconverged "
            f"wavefunction are not reference quality. Try a different "
            f"-guess, -guess-triplet for an open-shell singlet, or raise "
            f"the cycle limit.")


def _stabilize_uhf(mf, max_cycles=10, verbose=False,
                   newton_fallback=True):
    """
    Follow internal UHF instabilities until the solution is internally stable
    (analogous to Gaussian's Stable=Opt).

    A HOMO-LUMO mixed guess seeds a broken-symmetry state, but plain DIIS can
    relax back to the symmetric solution. This repeatedly runs the internal
    stability analysis and, whenever an instability is found, rotates along it
    and reconverges, descending to the genuine (often broken-symmetry) minimum.

    Parameters
    ----------
    mf : pyscf.scf.uhf.UHF
        A converged UHF mean-field object.
    max_cycles : int, optional
        Maximum stability-follow reoptimizations (default: 10).
    verbose : bool, optional
        Print progress (default: False).

    Returns
    -------
    mf : pyscf.scf.uhf.UHF
        UHF object at an internally stable solution (or the last one reached).
    """
    for i in range(1, max_cycles + 1):
        mo1, _, stable_i, _ = mf.stability(return_status=True)
        if stable_i:
            if verbose and i > 1:
                print(f"Stable=Opt: internally stable after "
                      f"{i - 1} reoptimization(s)")
            return mf
        if verbose:
            print(f"Stable=Opt: internal instability found; reoptimizing "
                  f"(cycle {i})")
        dm1 = mf.make_rdm1(mo1, mf.mo_occ)
        mf.kernel(dm0=dm1)
        if not mf.converged and newton_fallback:
            # The surface just past an instability rotation is flat, and DIIS
            # can crawl along it for hundreds of iterations without meeting the
            # tolerance. The second-order solver descends it in a handful of
            # steps, so try that before declaring the run lost.
            if verbose:
                print(f"Stable=Opt: DIIS did not converge on cycle {i}; "
                      f"retrying with the second-order (Newton) solver")
            conv_tol = mf.conv_tol
            mf = mf.newton()
            mf.conv_tol = conv_tol
            mf.kernel(dm0=dm1)
        _require_converged(mf, f"Stable=Opt reoptimization (cycle {i})")
    raise MinPopSCFError(
        f"still internally unstable after {max_cycles} Stable=Opt cycle(s). "
        f"The SCF is sitting on a saddle point of the orbital-rotation "
        f"surface, so a lower broken-symmetry solution exists that this "
        f"starting point cannot reach. Try -guess-triplet orbitals for an "
        f"open-shell singlet, or raise -stable-cycles.")


def run_uhf_from_xyz(xyz_file, charge=0, multiplicity=1, basis='6-31+G',
                     ecp=None, verbose=True, basis_dir=None,
                     standard_orientation=True,
                     azimuthal_gauge=True,
                     guess='minao',
                     guess_triplet=None,
                     guessmix=False, guessmix_angle=45.0,
                     stable=False, stable_cycles=5, max_cycle=128,
                     soscf=False, newton_fallback=True,
                     newton_conv_tol=1e-11,
                     cartesian=False, x2c=False, basis_light=None, light_max_z=18,
                     minimal_basis='sto-3g'):
    """
    Run UHF calculation and MinPop analysis from an XYZ file.
    
    Parameters
    ----------
    xyz_file : str
        Path to XYZ geometry file
    charge : int, optional
        Molecular charge (default: 0)
    multiplicity : int, optional
        Spin multiplicity 2S+1 (default: 1)
    basis : str, optional
        Basis set (default: '6-31+G')
    basis_light : str, optional
        Basis for atoms with Z <= light_max_z (H-Ar by default); `basis`
        then covers only the heavier atoms. Lets an expensive all-electron
        basis such as ANO-RCC-VDZP sit on the metals only while H-Ar use
        6-31G*, CBSB3, def2-SVP and so on. Default None: `basis` everywhere.
    light_max_z : int, optional
        Highest atomic number that takes basis_light (default 18, Ar).
    x2c : bool, optional
        Use the spin-free exact-two-component (sfX2C-1e) scalar-relativistic
        one-electron Hamiltonian. Required for sensible all-electron results
        on heavy atoms; ANO-RCC is contracted for a scalar-relativistic
        (DKH2) Hamiltonian, so use it with every ANO-RCC run. The MinPop
        projection is unchanged: X2C orbitals live in the same
        non-relativistic AO basis, so S_cross and S_min are the usual
        overlaps. Default False.
    cartesian : bool, optional
        Use Cartesian (6D/10F) angular functions for the COMPUTATIONAL basis
        instead of spherical (5D/7F). Default False (spherical), matching the
        original behaviour. Gaussian's convention is per-basis: 6-31G* and the
        6-31G family are defined 6D, while the 6-311G family, Gen input and
        cc-pVnZ default to 5D 7F. Set True to match a Gaussian 6-31G* job.
        The MBS minimal basis stays spherical regardless: Gaussian's MinPop
        uses a spherical minimal basis throughout (verified vs Gaussian).
    minimal_basis : str, optional
        Minimal-basis projection target: 'sto-3g' (default, Gaussian's MBS
        convention) or 'minao'.
    ecp : str or dict, optional
        ECP specification. If None, auto-detects for def2 basis sets
        with heavy elements (Z > 36)
    verbose : bool, optional
        Print output (default: True)
    
    Returns
    -------
    results : dict
        MinPop analysis results (see minpop_uhf)
    
    Examples
    --------
    >>> results = run_uhf_from_xyz("ch2.xyz", charge=0, multiplicity=3)
    >>> print(f"Carbon spin: {results['spin_populations'][0]:.4f}")
    """
    atom_str = _read_xyz(xyz_file)

    # Rotate into Gaussian's standard orientation so the frame-dependent MBS
    # density-matrix / per-AO population printouts match Gaussian (charges and
    # spins are rotation-invariant and unaffected). Disable with -no-std-orient
    # when the input geometry is already in Gaussian's standard orientation. The
    # "Standard orientation:" table is printed below the molecule summary.
    orient_table = None
    if standard_orientation:
        atom_str, orient_table = _standardize_atom_str(atom_str,
                                                       search_dir=basis_dir)

    # Resolve the computational basis: a standard name is passed through, while
    # 'cbsb3'/'cbsb7' (or a .py path / 'module:DICT') is loaded as a PySCF dict.
    basis_obj, auto_ecp, basis_name = _compose_basis(
        basis, basis_light, atom_str, light_max_z=light_max_z, search_dir=basis_dir)
    if ecp is None and auto_ecp:
        ecp = auto_ecp
        if verbose:
            print(f"Auto-detected ECP for heavy elements: {ecp}")
    
    mol = gto.M(
        atom=atom_str,
        basis=basis_obj,
        charge=charge,
        spin=multiplicity - 1,
        ecp=ecp,
        cart=cartesian
    )
    
    if verbose:
        print(f"Molecule: {xyz_file}")
        print(f"Charge: {charge}, Multiplicity: {multiplicity}")
        print(f"Basis: {basis_name} "
              f"({'6D/10F Cartesian' if cartesian else '5D/7F spherical'})")
        print(f"Atoms: {mol.natm}, Electrons: {mol.nelectron}")
        if orient_table:
            print()
            print(orient_table)
        print()
    
    # Match Gaussian's default linear-dependence handling (IOp(3/59)=6): discard
    # overlap-matrix eigenvectors whose eigenvalue is below 1e-6. This is already
    # PySCF's built-in default, but it is pinned here explicitly so the basis
    # culling matches Gaussian exactly and independently of the PySCF version.
    scf.hf.remove_overlap_zero_eigenvalue = True
    scf.hf.overlap_zero_eigenvalue_threshold = 1e-6

    # SCF with Gaussian-like defaults
    mf = scf.UHF(mol)
    if x2c:
        mf = mf.x2c()
        if verbose:
            print("Hamiltonian: spin-free X2C-1e (scalar relativistic)")
    elif ecp is None and max(mol.atom_charge(i) for i in range(mol.natm)) > 36 \
            and verbose:
        print("WARNING: all-electron basis on an atom beyond Kr without -x2c; "
              "non-relativistic orbitals for such atoms are not physical.")
    mf.max_cycle = max_cycle
    mf.conv_tol = 1e-9
    mf.conv_tol_grad = 1e-6
    mf.diis_space = 8
    mf.level_shift = 0.0

    # Optional broken-symmetry HOMO-LUMO mixed initial guess (Gaussian Guess=Mix)
    dm0 = None
    if guess_triplet:
        dm0 = _init_guess_triplet(mol, mode=guess_triplet,
                                  verbose=verbose)
    elif guessmix:
        dm0 = _init_guess_mixed(mol, mixing_angle_deg=guessmix_angle,
                                verbose=verbose, seed=guess)
    elif mol.spin == 0:
        # Plain UHF on a closed-shell singlet has to collapse onto the
        # restricted solution, and Gaussian duly prints spin densities of
        # exactly zero. PySCF's stock UHF guess is not exactly spin symmetric,
        # and the residual survives the SCF: about 1e-6 in the AO density,
        # which amplifies to ~1e-4 in the MBS spin densities and fails a 1e-4
        # comparison. Newton tightening barely touches it, because the
        # asymmetry costs no energy. Seeding alpha and beta with the identical
        # density keeps them identical through every iteration, so the spin
        # density comes out identically zero. -guessmix takes precedence, since
        # breaking that symmetry on purpose is the whole point of Guess=Mix.
        _dm_restricted = scf.hf.get_init_guess(mol, guess)
        dm0 = np.array([_dm_restricted * 0.5, _dm_restricted * 0.5])
    elif guess != 'minao':
        dm0 = mf.get_init_guess(key=guess)
    if verbose and guess != 'minao':
        print(f"Initial guess: {guess}")
    if soscf:
        # Second-order from the outset instead of only as a rescue. Costs more
        # per iteration than DIIS but descends flat broken-symmetry surfaces
        # that DIIS oscillates on, so it can be the difference between a run
        # that finishes and one that stalls at the right energy.
        if verbose:
            print("SCF: second-order (Newton) solver from the start (-soscf)")
        _conv, _mx = mf.conv_tol, mf.max_cycle
        mf = mf.newton()
        mf.conv_tol, mf.max_cycle = _conv, _mx

    mf.kernel(dm0=dm0)
    if not mf.converged and newton_fallback and not soscf:
        # DIIS can oscillate around a broken-symmetry solution without ever
        # meeting the tolerance: the energy sits at the right value while the
        # gradient refuses to fall. The second-order solver closes that gap in
        # a few steps, so try it before giving up on the run.
        if verbose:
            print("SCF: DIIS did not converge; retrying with the "
                  "second-order (Newton) solver")
        _conv = mf.conv_tol
        mf = mf.newton()
        mf.conv_tol = _conv
        mf.kernel(dm0=dm0)
    _require_converged(mf, "SCF")

    # Follow internal instabilities to the lower (broken-symmetry) solution
    if stable:
        mf = _stabilize_uhf(mf, max_cycles=stable_cycles,
                            verbose=verbose,
                            newton_fallback=newton_fallback)

    # Second-order (Newton) tightening. Broken-symmetry singlets sit on a very
    # flat surface, and plain DIIS at a loose gradient leaves spin-sensitive
    # properties (the spin density = D_alpha - D_beta) under-converged even when
    # the energy looks converged. Newton pushes the gradient down cheaply so the
    # MBS spin densities reproduce Gaussian.
    mo_coeff, mo_occ = mf.mo_coeff, mf.mo_occ    # converged DIIS/stability orbitals
    mf = mf.newton()
    mf.conv_tol = newton_conv_tol
    mf.kernel(mo_coeff, mo_occ)                   # seed Newton with the MOs, not a
    # density matrix. Passing make_rdm1() forces Newton to rebuild the orbitals by
    # diagonalizing F against the near-singular overlap (dm -> MO), which is what
    # emitted "Newton solver ... treated as density matrix" and the
    # "Singularity detected in overlap matrix (condition number = ...)" warning.
    # Seeding with the already-converged MOs skips that redundant step entirely.
    
    if verbose:
        print()
    
    return minpop_uhf(mf, verbose=verbose, azimuthal_gauge=azimuthal_gauge,
                      minimal_basis=minimal_basis)


# =============================================================================
# Command Line Interface
# =============================================================================


# --------------------------------------------------------------------------- #
# Live export (-json / -csv-long / -csv-features)
# --------------------------------------------------------------------------- #
class _TeeStream:
    """Write-through to several streams: the MinPop report reaches stdout
    unchanged while a copy accumulates for export."""

    def __init__(self, *streams):
        self._streams = streams

    def write(self, s):
        for st in self._streams:
            st.write(s)
        return len(s)

    def flush(self):
        for st in self._streams:
            st.flush()

    def __getattr__(self, name):
        return getattr(self._streams[0], name)


@contextlib.contextmanager
def _capture_report():
    """Tee stdout into a buffer while the analysis runs.

    PySCF binds lib.StreamObject.stdout to sys.stdout at import time, so BOTH
    must be redirected; with sys.stdout alone the 'converged SCF energy' line
    never reaches the buffer and every exported record loses its energy.
    """
    buf = io.StringIO()
    tee = _TeeStream(sys.stdout, buf)
    old_sys = sys.stdout
    from pyscf import lib
    old_lib = lib.StreamObject.stdout
    sys.stdout = tee
    lib.StreamObject.stdout = tee
    try:
        yield buf
    finally:
        sys.stdout = old_sys
        lib.StreamObject.stdout = old_lib


def _load_export_module():
    """Find minpop_json_csv: import path, next to this script, then the cwd.
    Returns None with a stderr note when unavailable; export is optional and
    must never fail a run whose analysis already succeeded."""
    try:
        import minpop_json_csv as mod
        return mod
    except ImportError:
        pass
    here = os.path.dirname(os.path.abspath(__file__))
    for d in (here, os.getcwd()):
        path = os.path.join(d, "minpop_json_csv.py")
        if os.path.isfile(path):
            try:
                spec = importlib.util.spec_from_file_location(
                    "minpop_json_csv", path)
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
                return mod
            except Exception as exc:
                print(f"[export] failed to load {path}: {exc}",
                      file=sys.stderr)
                return None
    print("[export] minpop_json_csv.py not found (import path, next to this "
          "script, current directory); skipping export", file=sys.stderr)
    return None


def _export_outputs(text, args):
    """Write the requested JSON/CSV products from the captured report.

    All notes go to stderr so a redirected .out stays parseable, and every
    failure is reported and swallowed: the export is a convenience layered on
    an analysis that already succeeded."""
    mod = _load_export_module()
    if mod is None:
        return
    src = args.xyz_file
    if args.json_path:
        try:
            mod.export_json(text, args.json_path, source=src,
                            xyz_dirs=args.json_xyz_dir)
            print(f"[export] json -> {args.json_path}", file=sys.stderr)
        except Exception as exc:
            print(f"[export] json failed: {type(exc).__name__}: {exc}",
                  file=sys.stderr)
    if args.csv_long:
        try:
            parsed = mod._parse_for_csv(text, source=src)
        except Exception as exc:
            print(f"[export] csv parse failed: {type(exc).__name__}: {exc}",
                  file=sys.stderr)
            return
        if args.csv_long:
            try:
                n = mod.write_long_csv([parsed], args.csv_long,
                                       sparse_tol=args.csv_sparse_tol)
                print(f"[export] csv-long ({n} rows) -> {args.csv_long}",
                      file=sys.stderr)
            except Exception as exc:
                print(f"[export] csv-long failed: "
                      f"{type(exc).__name__}: {exc}", file=sys.stderr)


def main():
    """Command-line entry point."""
    parser = argparse.ArgumentParser(
        description="MinPop analysis for UHF wavefunctions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python minpop_uhf.py -xyz ch2.xyz -charge 0 -mult 3
  python minpop_uhf.py -xyz radical.xyz -mult 2 -basis cc-pVDZ
  python minpop_uhf.py -xyz snh4.xyz -basis def2-TZVPP     # Auto-detects ECP
  python minpop_uhf.py -xyz ch2.xyz -mult 3 -basis cbsb7   # Gaussian CBSB7 (H-Ar)
  python minpop_uhf.py -xyz ch2.xyz -mult 3 -basis cbsb3 -basis-dir ./basis
  python minpop_uhf.py -xyz singlet_diradical.xyz -mult 1 -guessmix -stable  # broken-symmetry

Notes:
  Output matches Gaussian 16's Pop=(Full) IOp(6/27=122,6/12=3) format.
  Geometries are rotated into Gaussian's standard orientation by default (via
  gaussian_standard_orientation.py) so the frame-dependent MBS density-matrix /
  per-AO population output matches Gaussian; pass -no-std-orient to keep the
  input frame. Atomic charges and spins are rotation-invariant either way.
  ECP auto-detected for def2 basis sets with heavy elements (Z > 36).
  For open-shell-singlet / antiferromagnetic states use -guessmix (Gaussian
  Guess=Mix) together with -stable (Gaussian Stable=Opt): the mix seeds the
  broken-symmetry state and -stable keeps DIIS from relaxing back to symmetric.
  ANO-RCC: -basis ano-rcc-vdzp (also -mb/-vtzp/-vqzp) with -minimal-basis
  ano-rcc-mb and -x2c gives an all-electron relativistic MinPop for any
  element H-Cm (needs anorcc_basis_pyscf.py next to this script).
  -basis accepts a standard name, 'cbsb3'/'cbsb7', a path to a *_basis_pyscf.py
  file, or 'module:DICTNAME'. Custom modules are found in -basis-dir, the current
  directory, or next to this script.
"""
    )
    parser.add_argument("-xyz", required=True, dest="xyz_file",
                        help="Path to XYZ geometry file")
    parser.add_argument("-charge", type=int, default=0,
                        help="Molecular charge (default: 0)")
    parser.add_argument("-mult", type=int, default=1,
                        help="Spin multiplicity 2S+1 (default: 1)")
    parser.add_argument("-basis", default="6-31+G",
                        help="Computational basis: a standard name (e.g. cc-pVDZ), "
                             "'cbsb3'/'cbsb7', a path to a *_basis_pyscf.py file, "
                             "or 'module:DICTNAME' (default: 6-31+G)")
    parser.add_argument("-basis-dir", dest="basis_dir", default=None,
                        help="Directory to search for custom basis modules "
                             "(cbsb3_basis_pyscf.py / cbsb7_basis_pyscf.py)")
    parser.add_argument("-cartesian", "-cart", dest="cartesian",
                        action="store_true", default=False,
                        help="Use Cartesian (6D/10F) angular functions for the "
                             "computational basis; matches Gaussian for the "
                             "6-31G family (6-31G* is defined 6D)")
    parser.add_argument("-spherical", "-sph", dest="cartesian",
                        action="store_false",
                        help="Use spherical (5D/7F) angular functions "
                             "(default; matches Gen input, the 6-311G family "
                             "and cc-pVnZ)")
    parser.add_argument("-minimal-basis", "-mbs", dest="minimal_basis",
                        type=_check_minimal_basis, default="sto-3g",
                        help="Minimal-basis projection target (default: "
                             "sto-3g, Gaussian's MBS convention; "
                             "ano-rcc-mb = all-electron relativistic ANO "
                             "minimal basis, any element H-Cm; sto-3g-ano = "
                             "STO-3G on H-Ar and ANO-RCC-MB on heavier atoms; "
                             "LIGHT+HEAVY, e.g. sto-3g+ano-rcc-mb or sto-3g+minao, "
                             "puts LIGHT on Z<=-light-max-z and HEAVY on the rest)")
    parser.add_argument("-label-style", dest="label_style",
                        choices=["gaussian", "physical"], default=None,
                        help="AO label numbering: gaussian = running shell "
                             "counter as Gaussian prints it (Si 3d = 4D, Pt 5d "
                             "= 9D), physical = principal quantum number (3D, "
                             "5D). Default: gaussian for sto-3g and sto-3g-ano, "
                             "physical for ano-rcc-mb. Labels only.")
    parser.add_argument("-symmetry", "-sigma-pi", dest="symmetry", action="store_true",
                        default=False,
                        help="Resolve each bond population into σ/π/δ/φ "
                             "(exact for any l and orientation); matrices condensed to atoms")
    parser.add_argument("-symmetry-shells", "-sigma-pi-shells", dest="symmetry_shells",
                        action="store_true", default=False,
                        help="With -symmetry: also the matrices condensed to shells")
    parser.add_argument("-basis-light", dest="basis_light", default=None,
                        help="Basis for atoms up to -light-max-z (default Ar); -basis "
                             "then applies to the heavier atoms only, e.g. "
                             "-basis ano-rcc-vdzp -basis-light 6-31G*")
    parser.add_argument("-light-max-z", dest="light_max_z", type=int, default=18,
                        help="Highest Z that takes -basis-light (default 18)")
    parser.add_argument("-x2c", dest="x2c", action="store_true",
                        default=False,
                        help="Scalar-relativistic sfX2C-1e Hamiltonian. Use "
                             "with all-electron bases on heavy atoms "
                             "(ANO-RCC-VDZP on Pt, etc.).")
    parser.add_argument("-ecp", default=None,
                        help="ECP (auto-detected for def2 + heavy elements)")
    parser.add_argument("-no-azimuthal-gauge", dest="azimuthal_gauge",
                        action="store_false", default=True,
                        help="Do not canonicalize the azimuth of linear "
                             "molecules. Use together with -no-std-orient "
                             "when the input geometry is already Gaussian's "
                             "standard orientation, so the frame is adopted "
                             "verbatim and nothing rotates it afterwards.")
    parser.add_argument("-no-std-orient", dest="standard_orientation",
                        action="store_false",
                        help="Skip Gaussian standard reorientation and use the "
                             "input geometry as-is (e.g. it is already in "
                             "Gaussian's standard orientation). By default the "
                             "geometry is rotated into Gaussian's frame via "
                             "gaussian_standard_orientation.py")
    parser.add_argument("-guessmix", action="store_true",
                        help="Break spin symmetry via a HOMO-LUMO mixed initial "
                             "guess (Gaussian Guess=Mix) for open-shell singlets / "
                             "antiferromagnetic broken-symmetry states")
    parser.add_argument("-guessmix-angle", dest="guessmix_angle", type=float,
                        default=45.0,
                        help="HOMO-LUMO mixing angle in degrees (default: 45, "
                             "matching Gaussian's 0.7071 coefficients)")
    parser.add_argument("-stable", action="store_true",
                        help="Follow internal UHF instabilities to the lower "
                             "solution (Gaussian Stable=Opt); needed to keep a "
                             "broken-symmetry state that DIIS would otherwise "
                             "relax back to the symmetric solution")
    parser.add_argument("-soscf", dest="soscf", action="store_true",
                        help="Use the second-order (Newton) solver for the "
                             "main SCF from the start, rather than only as a "
                             "rescue after DIIS fails. Slower per iteration "
                             "but far steadier on the flat surfaces of "
                             "broken-symmetry singlets.")
    parser.add_argument("-no-newton-fallback", dest="newton_fallback",
                        action="store_false", default=True,
                        help="Do not retry a stalled DIIS solve with the "
                             "second-order solver. The run then stops on the "
                             "first non-convergence, which is what you want "
                             "when reproducing an older run exactly or when "
                             "testing whether DIIS alone can do the job.")
    parser.add_argument("-newton-conv-tol", dest="newton_conv_tol", type=float,
                        default=1e-11,
                        help="Convergence tolerance for the final Newton "
                             "tightening pass that settles the spin densities "
                             "(default: 1e-11).")
    parser.add_argument("-max-cycle", dest="max_cycle", type=int, default=128,
                        help="Max SCF iterations per solve (default: 128). "
                             "Raise it when a run stops with 'did not "
                             "converge' while the energy and <S^2> are still "
                             "drifting toward a sensible answer, which means "
                             "it needed more iterations rather than a "
                             "different starting point.")
    parser.add_argument("-stable-cycles", dest="stable_cycles", type=int,
                        default=5,
                        help="Max stability-follow reoptimizations (default: "
                             "5). The run STOPS if the wavefunction is still "
                             "internally unstable after them, rather than "
                             "reporting populations from a saddle point.")
    parser.add_argument("-guess-triplet", dest="guess_triplet", nargs="?",
                        const="density", default=None,
                        choices=["density", "orbitals"],
                        help="Singlets only: converge a triplet ROHF in the "
                             "same basis and start the UHF from it, so no "
                             "-guessmix is needed. 'density' (the default when "
                             "the flag is given bare) passes the triplet's own "
                             "alpha and beta densities, consistent with the "
                             "-guess family. 'orbitals' instead reoccupies the "
                             "triplet's spatial orbitals for the singlet, "
                             "which starts from the correct electron count per "
                             "spin and can reach broken-symmetry minima the "
                             "density mode misses.")
    parser.add_argument("-guess", dest="guess", default="minao",
                        choices=["minao", "huckel", "vsap", "sap", "atom",
                                 "1e"],
                        help="SCF initial guess (default: minao). With "
                             "-guessmix this seeds the density whose frontier "
                             "orbitals are mixed, so it selects the "
                             "broken-symmetry basin; 'huckel' is worth trying "
                             "when a minao-seeded mix fails to converge.")
    parser.add_argument("-json", dest="json_path", metavar="PATH",
                        default=None,
                        help="Also serialize this run to a JSON record via "
                             "minpop_json_csv.py (never fatal: a failed export "
                             "warns on stderr and leaves the analysis intact)")
    parser.add_argument("-json-xyz-dir", dest="json_xyz_dir", action="append",
                        metavar="DIR",
                        help="Directory with the input-orientation .xyz for "
                             "the JSON record (repeatable)")
    parser.add_argument("-csv-long", dest="csv_long", metavar="PATH",
                        default=None,
                        help="Also write this run's nonzero MBS elements as a "
                             "tidy sparse CSV (one file per run; aggregate a "
                             "dataset afterwards with minpop_json_csv.py "
                             "--runs)")
    parser.add_argument("-csv-sparse-tol", dest="csv_sparse_tol", type=float,
                        default=0.0,
                        help="Drop |value| <= this from -csv-long "
                             "(default 0: exact zeros only)")
    parser.add_argument("-q", "--quiet", action="store_true",
                        help="Suppress output")
    parser.add_argument("--version", action="version",
                        version=f"%(prog)s {__version__}")
    
    args = parser.parse_args()
    global LABEL_STYLE, BOND_SYMMETRY, BOND_SYMMETRY_SHELLS, MBS_LIGHT_MAX_Z
    MBS_LIGHT_MAX_Z = args.light_max_z
    LABEL_STYLE = args.label_style
    BOND_SYMMETRY = args.symmetry or args.symmetry_shells
    BOND_SYMMETRY_SHELLS = args.symmetry_shells

    if args.guess_triplet is not None:
        if args.mult != 1:
            sys.exit("-guess-triplet applies to singlets only "
                     f"(-mult 1); got -mult {args.mult}")
        if args.guessmix:
            sys.exit("-guess-triplet and -guessmix are two different ways to "
                     "reach the broken-symmetry solution; pick one")

    exporting = bool(args.json_path or args.csv_long)
    if exporting and args.quiet:
        print("[export] note: -q suppresses the report the exports are parsed "
              "from; exported records will be mostly empty", file=sys.stderr)

    # The capture starts before the command echo so the exported record hashes
    # the same bytes a redirected .out receives (source.sha256 == sha of .out).
    ctx = _capture_report() if exporting else contextlib.nullcontext()
    failure = None
    with ctx as buf:
        try:
            # Echo the exact command as the very first line so the run is
            # reproducible.
            if not args.quiet:
                print("Command line: python "
                      + " ".join(shlex.quote(a) for a in sys.argv))

            run_uhf_from_xyz(
                args.xyz_file,
                charge=args.charge,
                multiplicity=args.mult,
                basis=args.basis,
                ecp=args.ecp,
                verbose=not args.quiet,
                basis_dir=args.basis_dir,
                standard_orientation=args.standard_orientation,
                azimuthal_gauge=args.azimuthal_gauge,
                guess=args.guess,
                guess_triplet=args.guess_triplet,
                guessmix=args.guessmix,
                guessmix_angle=args.guessmix_angle,
                stable=args.stable,
                stable_cycles=args.stable_cycles,
            max_cycle=args.max_cycle,
            soscf=args.soscf,
            newton_fallback=args.newton_fallback,
            newton_conv_tol=args.newton_conv_tol,
            cartesian=args.cartesian,
            x2c=args.x2c,
            basis_light=args.basis_light,
            light_max_z=args.light_max_z,
            minimal_basis=args.minimal_basis
            )
        except MinPopSCFError as exc:
            failure = exc


    if failure is not None:
        # No export: there is no trustworthy result to serialize. The partial
        # report still reaches stdout so the redirected .out shows how far the
        # run got.
        print(f"MinPop ERROR: {failure}", file=sys.stderr)
        sys.exit(1)

    if exporting:
        _export_outputs(buf.getvalue(), args)


if __name__ == "__main__":
    main()
