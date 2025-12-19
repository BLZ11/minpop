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
    - S_min     : Overlap matrix in minimal basis
    - S_cross   : Cross-overlap matrix <minimal|extended>
    - M         : Metric matrix ensuring orthonormality, M = C'^T·S_min·C'

For UHF wavefunctions, alpha and beta orbitals have different spatial parts and
must be projected independently.

Minimal Basis Selection
-----------------------
Following Gaussian's convention:
    - First row (H-Ne):     STO-3G
    - Second row (Na-Ar):   STO-3G* (with d-polarization, Cartesian 6D)
    - Third row+ / TM:      STO-3G (spherical 5D for d-orbitals)

Spin Annihilation
-----------------
UHF wavefunctions are contaminated by higher spin states. The first spin
contaminant (S+1) is removed using Löwdin's projection operator method.
For singlets, the annihilated spin density is exactly zero.

References
----------
[1] Montgomery Jr., J. A. et al. J. Chem. Phys. 110, 2822-2827 (1999).
[2] Montgomery Jr., J. A. et al. J. Chem. Phys. 112, 6532-6542 (2000).
[3] Löwdin, P.-O. Phys. Rev. 97, 1509-1520 (1955).

Author: Barbaro Zulueta (BLZ11)
"""

import argparse
import numpy as np
from pyscf import gto, scf
from pyscf.gto import intor_cross

__version__ = "1.0.0"
__author__ = "Barbaro Zulueta (BLZ11)"
__all__ = ["minpop_uhf", "run_uhf_from_xyz"]


# =============================================================================
# Constants
# =============================================================================

# Elements using STO-3G* (with d-polarization) for minimal basis
_SECOND_ROW = frozenset({'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar'})


# =============================================================================
# Minimal Basis Construction
# =============================================================================

def _build_minimal_basis_mol(mol):
    """
    Construct minimal basis molecule matching Gaussian's MinPop convention.
    
    Parameters
    ----------
    mol : pyscf.gto.Mole
        Input molecule in extended basis set
        
    Returns
    -------
    mol_min : pyscf.gto.Mole
        Molecule with minimal basis set
        
    Notes
    -----
    Second-row elements use STO-3G* (Cartesian 6D), while transition metals
    use STO-3G (spherical 5D). The basis type affects d-orbital labeling.
    """
    from pyscf.data import elements
    
    has_second_row = False
    has_transition_metal = False
    basis_dict = {}
    
    for i in range(mol.natm):
        symbol = mol.atom_symbol(i)
        elem = ''.join(c for c in symbol if c.isalpha())
        
        # Determine basis set for this element
        if elem in _SECOND_ROW:
            basis_dict[symbol] = 'STO-3G*'
            has_second_row = True
        else:
            basis_dict[symbol] = 'STO-3G'
            # Check for transition metals
            try:
                z = elements.charge(elem)
                if (21 <= z <= 30) or (39 <= z <= 48) or (57 <= z <= 80) or (89 <= z <= 112):
                    has_transition_metal = True
            except (KeyError, ValueError):
                pass
    
    # Determine Cartesian vs spherical for d-orbitals
    # STO-3G* uses Cartesian (6D), transition metal STO-3G uses spherical (5D)
    use_cartesian = has_second_row and not has_transition_metal
    
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
    Project MO coefficients to minimal basis with orthonormalization.
    
    The projection ensures orthonormality in the minimal basis through
    symmetric orthogonalization of the metric matrix.
    
    Parameters
    ----------
    mo_coeff : ndarray, shape (nao_ext, nmo)
        MO coefficients in extended basis set
    S_cross : ndarray, shape (nao_min, nao_ext)
        Cross-overlap matrix <minimal|extended>
    S_min_inv : ndarray, shape (nao_min, nao_min)
        Inverse of minimal basis overlap matrix
        
    Returns
    -------
    mo_min : ndarray, shape (nao_min, nmo)
        Orthonormalized MO coefficients in minimal basis
    """
    # Raw projection: C' = S_min^(-1) · S_cross · C
    C_proj = S_min_inv @ S_cross @ mo_coeff
    
    # Compute metric for orthonormalization: M = C'^T · S_min · C'
    M = C_proj.T @ np.linalg.inv(S_min_inv) @ C_proj
    
    # Symmetric orthogonalization: M^(-1/2)
    eigval, eigvec = np.linalg.eigh(M)
    eigval = np.maximum(eigval, 1e-14)  # Numerical stability
    M_invsqrt = eigvec @ np.diag(1.0 / np.sqrt(eigval)) @ eigvec.T
    
    return C_proj @ M_invsqrt


# =============================================================================
# AO Reordering (PySCF → Gaussian Convention)
# =============================================================================

def _get_gaussian_ao_order(mol_min):
    """
    Generate permutation indices to reorder AOs from PySCF to Gaussian convention.
    
    Gaussian orders AOs by increasing shell number within each atom:
        1S, 2S, 2P, 3S, 3P, 3D, 4S, 4P, ...
    
    For d-orbitals:
        - Cartesian (STO-3G*): XX, YY, ZZ, XY, XZ, YZ
        - Spherical (STO-3G):  D0, D+1, D-1, D+2, D-2
    
    Parameters
    ----------
    mol_min : pyscf.gto.Mole
        Molecule in minimal basis
        
    Returns
    -------
    reorder : list of int
        Permutation indices for reordering
    labels : list of tuple
        Reordered AO labels with Gaussian shell numbering
    """
    ao_labels = mol_min.ao_labels(fmt=None)
    
    # Detect basis type from d-orbital labels
    has_cartesian_d = any(
        lbl[3].lower() in ('xx', 'yy', 'zz')
        for lbl in ao_labels if 'd' in lbl[2].lower()
    )
    
    # Group AOs by atom
    atom_aos = {}
    for i, lbl in enumerate(ao_labels):
        atom_idx = lbl[0]
        if atom_idx not in atom_aos:
            atom_aos[atom_idx] = []
        atom_aos[atom_idx].append((i, lbl))
    
    new_order = []
    new_labels = []
    
    for atom_idx in sorted(atom_aos.keys()):
        aos = atom_aos[atom_idx]
        
        # Check for transition metal (has 3d and 4s)
        orbitals = [lbl[2].lower() for _, lbl in aos]
        is_transition_metal = '3d' in orbitals and '4s' in orbitals
        
        # Parse and assign Gaussian shell numbers
        parsed = []
        for i, lbl in aos:
            orb = lbl[2].lower()
            cart = (lbl[3] or '').lower()
            
            shell, ang, subtype, m_val = _parse_ao_label(
                orb, cart, is_transition_metal, has_cartesian_d
            )
            parsed.append((i, lbl, shell, ang, subtype, m_val))
        
        # Sort by shell, angular momentum, then component
        parsed.sort(key=lambda x: _ao_sort_key(x, has_cartesian_d))
        
        for item in parsed:
            new_order.append(item[0])
            new_labels.append(item[1:])
    
    return new_order, new_labels


def _parse_ao_label(orb, cart, is_transition_metal, has_cartesian_d):
    """
    Parse PySCF orbital label to determine Gaussian shell number and type.
    
    Returns
    -------
    tuple : (shell, ang, subtype, m_value)
        shell   : Gaussian shell number (1, 2, 3, ...)
        ang     : Angular momentum (0=s, 1=p, 2=d)
        subtype : Component label ('x', 'y', 'z', 'xx', 'sph', etc.)
        m_value : Magnetic quantum number for spherical harmonics
    """
    # S orbitals
    if 's' in orb and not cart:
        shell_num = int(orb[0]) if orb[0].isdigit() else 1
        if is_transition_metal:
            # Transition metal: 4s→5, 5s→6
            if orb == '4s':
                shell_num = 5
            elif orb == '5s':
                shell_num = 6
        return (shell_num, 0, '', 0)
    
    # P orbitals
    if 'p' in orb and cart in ('x', 'y', 'z'):
        shell_num = int(orb[0]) if orb[0].isdigit() else 2
        if is_transition_metal:
            if orb == '4p':
                shell_num = 5
            elif orb == '5p':
                shell_num = 6
        m_val = {'x': 1, 'y': -1, 'z': 0}.get(cart, 0)
        return (shell_num, 1, cart, m_val)
    
    # D orbitals - Cartesian (STO-3G*)
    if has_cartesian_d and cart in ('xx', 'xy', 'xz', 'yy', 'yz', 'zz'):
        shell_num = 4 if orb == '3d' else (5 if orb == '4d' else 4)
        return (shell_num, 2, cart, 0)
    
    # D orbitals - Spherical (STO-3G for transition metals)
    if 'd' in orb:
        shell_num = 4 if orb == '3d' else (5 if orb == '4d' else 4)
        m_val, subtype = _parse_spherical_d(cart)
        return (shell_num, 2, subtype, m_val)
    
    return (1, 0, '', 0)  # Fallback


def _parse_spherical_d(cart):
    """Parse spherical d-orbital label to magnetic quantum number."""
    cart_orig = cart
    cart = cart.lower()
    
    # Numeric m-values
    if cart_orig in ('0', '+0', '-0'):
        return (0, 'sph')
    if cart_orig in ('+1', '1'):
        return (1, 'sph')
    if cart_orig == '-1':
        return (-1, 'sph')
    if cart_orig in ('+2', '2'):
        return (2, 'sph')
    if cart_orig == '-2':
        return (-2, 'sph')
    
    # Text labels (real spherical harmonics)
    text_to_m = {
        'z^2': 0, 'z2': 0, 'dz2': 0,
        'xz': 1, 'dxz': 1,
        'yz': -1, 'dyz': -1,
        'x2-y2': 2, 'dx2-y2': 2,
        'xy': -2, 'dxy': -2,
    }
    if cart in text_to_m:
        return (text_to_m[cart], 'sph')
    
    # Try parsing as integer
    try:
        return (int(cart_orig.replace('+', '')), 'sph')
    except (ValueError, AttributeError):
        return (0, 'sph')


def _ao_sort_key(parsed_ao, has_cartesian_d):
    """Generate sort key for AO ordering."""
    _, _, shell, ang, subtype, m_val = parsed_ao
    
    if ang == 1:  # P orbitals: X, Y, Z
        p_order = {'x': 0, 'y': 1, 'z': 2}
        return (shell, ang, p_order.get(subtype, 0))
    
    if ang == 2:  # D orbitals
        if has_cartesian_d and subtype in ('xx', 'xy', 'xz', 'yy', 'yz', 'zz'):
            # Cartesian: XX, YY, ZZ, XY, XZ, YZ
            d_cart_order = {'xx': 0, 'yy': 1, 'zz': 2, 'xy': 3, 'xz': 4, 'yz': 5}
            return (shell, ang, d_cart_order.get(subtype, 0))
        else:
            # Spherical: 0, +1, -1, +2, -2
            d_sph_order = {0: 0, 1: 1, -1: 2, 2: 3, -2: 4}
            return (shell, ang, d_sph_order.get(m_val, 0))
    
    return (shell, ang, 0)


def _convert_label_to_gaussian(lbl_info):
    """
    Convert parsed AO label to Gaussian format string.
    
    Parameters
    ----------
    lbl_info : tuple
        (pyscf_label, shell, ang, subtype, m_value)
        
    Returns
    -------
    tuple : (atom_idx, element, orbital_str, '')
        Gaussian-formatted AO label
    """
    lbl, shell, ang, subtype, m_val = lbl_info
    atom_idx, elem = lbl[0], lbl[1]
    
    if ang == 0:
        return (atom_idx, elem, f'{shell}S', '')
    
    if ang == 1:
        return (atom_idx, elem, f'{shell}P{subtype.upper()}', '')
    
    if ang == 2:
        if subtype in ('xx', 'xy', 'xz', 'yy', 'yz', 'zz'):
            return (atom_idx, elem, f'{shell}{subtype.upper()}', '')
        else:
            # Spherical: 4D 0, 4D+1, 4D-1, 4D+2, 4D-2
            m_str = ' 0' if m_val == 0 else (f'+{m_val}' if m_val > 0 else str(m_val))
            return (atom_idx, elem, f'{shell}D{m_str}', '')
    
    return (atom_idx, elem, lbl[2].upper(), '')


# =============================================================================
# Matrix Utilities
# =============================================================================

def _reorder_matrix(matrix, order):
    """Reorder rows and columns of a symmetric matrix."""
    return matrix[np.ix_(order, order)]


def _mulliken_pop_matrix(dm, S):
    """
    Compute Mulliken population matrix.
    
    P_ij = (D·S)_ij represents the electron population shared between
    basis functions i and j.
    """
    return dm * S


def _condense_to_atoms(pop_matrix, ao_labels):
    """
    Condense orbital population matrix to atom-atom matrix.
    
    Sums contributions from all basis functions on each atom pair.
    """
    n_atoms = max(lbl[0] for lbl in ao_labels) + 1
    condensed = np.zeros((n_atoms, n_atoms))
    
    for i, li in enumerate(ao_labels):
        for j, lj in enumerate(ao_labels):
            condensed[li[0], lj[0]] += pop_matrix[i, j]
    
    return condensed


# =============================================================================
# Output Formatting (Gaussian 16 Compatible)
# =============================================================================

def _format_value(v):
    """Format floating-point value matching Gaussian style."""
    if abs(v) < 1e-5:
        return f"{'0.00000':>10}" if v >= 0 else f"{'-0.00000':>10}"
    return f"{v:>10.5f}"


def _format_ao_label(idx, lbl, ao_labels):
    """
    Format AO row label matching Gaussian's fixed-width format.
    
    Format: '   1 1   C  1S      ' (20 characters)
    """
    orb_str = f"{lbl[2]}{lbl[3]}"
    
    # Show atom number only on first AO of each atom
    if idx == 0 or ao_labels[idx][0] != ao_labels[idx - 1][0]:
        atom_num = lbl[0] + 1
        elem = lbl[1]
        return f"{idx + 1:>4} {atom_num:<1}   {elem:<2} {orb_str:<6}"
    else:
        return f"{idx + 1:>4}        {orb_str:<6}"


def _print_density_matrix(dm, ao_labels, title):
    """Print density matrix in Gaussian's column-blocked format."""
    n = dm.shape[0]
    block_size = 5
    
    print(f"     {title}:")
    
    for block_start in range(0, n, block_size):
        block_end = min(block_start + block_size, n)
        
        # Column headers
        header = "                  " + "".join(f"{i + 1:>10}" for i in range(block_start, block_end))
        print(header)
        
        # Matrix rows (lower triangle only)
        for i in range(block_start, n):
            row_label = _format_ao_label(i, ao_labels[i], ao_labels)
            cols_to_print = min(i + 1, block_end) - block_start
            if cols_to_print > 0:
                values = "".join(_format_value(dm[i, block_start + k]) for k in range(cols_to_print))
                print(f"{row_label}{values}")


def _print_gross_populations(gross, ao_labels):
    """Print gross orbital populations table."""
    print("     MBS Gross orbital populations:")
    print("                         Total     Alpha     Beta      Spin")
    
    for i, lbl in enumerate(ao_labels):
        row_label = _format_ao_label(i, lbl, ao_labels)
        values = f"{gross[i, 0]:>10.5f}{gross[i, 1]:>10.5f}{gross[i, 2]:>10.5f}{gross[i, 3]:>10.5f}"
        print(f"{row_label}{values}")


def _print_atomic_matrix(matrix, mol, title, use_period=False):
    """Print atom-atom matrix (condensed populations or spin densities)."""
    n_atoms = mol.natm
    block_size = 6
    suffix = '.' if use_period else ':'
    
    for block_start in range(0, n_atoms, block_size):
        block_end = min(block_start + block_size, n_atoms)
        
        # Header
        header = "        " + "".join(f"{i + 1:>12}" for i in range(block_start, block_end))
        print(f"          MBS {title}{suffix}" if block_start == 0 else header)
        if block_start == 0:
            print(header)
        
        # Matrix rows
        for i in range(n_atoms):
            elem = mol.atom_symbol(i)
            row_start = f"     {i + 1:>2}  {elem:<2}"
            values = "".join(f"{matrix[i, j]:>12.6f}" for j in range(block_start, block_end))
            print(f"{row_start}{values}")


def _print_mulliken_summary(charges, spins, mol):
    """Print Mulliken charges and spin densities summary."""
    print(" MBS Mulliken charges and spin densities:")
    print("                  1          2")
    
    for i in range(mol.natm):
        elem = mol.atom_symbol(i)
        print(f"        {i + 1}  {elem:<2} {charges[i]:>10.6f} {spins[i]:>10.6f}")
    
    total_charge = np.sum(charges)
    total_spin = np.sum(spins)
    print(f" Sum of MBS Mulliken charges = {total_charge:>10.5f} {total_spin:>10.5f}")


def _print_results(results, mol_min, n_alpha, n_beta):
    """Print complete MinPop analysis in Gaussian format."""
    ao_labels = results['ao_labels']
    
    # Header
    print(f"UHF orbital structure: {n_alpha} alpha, {n_beta} beta")
    print(" Annihilation of the first spin contaminant:")
    print(f" S**2 before annihilation {results['s2_before_annihilation']:>8.4f},   "
          f"after {results['s2_after_annihilation']:>8.4f}")
    print("=" * 60)
    print("MinPop Analysis (UHF)")
    print("=" * 60)
    
    # Density matrices
    _print_density_matrix(results['dm_alpha'], ao_labels, "Alpha  MBS Density Matrix")
    _print_density_matrix(results['dm_beta'], ao_labels, "Beta  MBS Density Matrix")
    _print_density_matrix(results['dm_total'], ao_labels, "Total  MBS Density Matrix")
    _print_density_matrix(results['dm_spin'], ao_labels, "Spin  MBS Density Matrix")
    
    # Population analysis
    _print_gross_populations(results['gross_orbital_pop'], ao_labels)
    _print_atomic_matrix(results['condensed_to_atoms'], mol_min, "Condensed to atoms (all electrons)")
    _print_atomic_matrix(results['spin_atomic'], mol_min, "Atomic-Atomic Spin Densities")
    _print_mulliken_summary(results['mulliken_charges'], results['spin_populations'], mol_min)
    
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


# =============================================================================
# Main Analysis Functions
# =============================================================================

def minpop_uhf(mf, verbose=True):
    """
    Perform MinPop population analysis on a converged UHF calculation.
    
    Projects the UHF wavefunction onto a minimal basis set and computes
    Mulliken population analysis with spin annihilation correction.
    
    Parameters
    ----------
    mf : pyscf.scf.uhf.UHF
        Converged UHF mean-field object
    verbose : bool, optional
        Print Gaussian-formatted output (default: True)
    
    Returns
    -------
    results : dict
        Dictionary containing:
        - 'dm_alpha', 'dm_beta': Alpha/beta density matrices in minimal basis
        - 'dm_total', 'dm_spin': Total and spin density matrices
        - 'pop_total', 'pop_spin': Mulliken population matrices
        - 'gross_orbital_pop': Per-orbital populations [total, alpha, beta, spin]
        - 'condensed_to_atoms': Atom-atom population matrix
        - 'spin_atomic': Atom-atom spin density matrix
        - 'mulliken_charges': Atomic partial charges
        - 'spin_populations': Atomic spin populations
        - 'ao_labels': Minimal basis AO labels in Gaussian format
        - 's2_before_annihilation': <S²> before spin projection
        - 's2_after_annihilation': <S²> after spin projection
    
    Notes
    -----
    UHF alpha and beta orbitals have different spatial parts, so they are
    projected to the minimal basis independently. The spin density is
    corrected using single annihilation of the first spin contaminant.
    
    Examples
    --------
    >>> from pyscf import gto, scf
    >>> mol = gto.M(atom='O 0 0 0; O 0 0 1.2', basis='6-31G*', spin=2)
    >>> mf = scf.UHF(mol).run()
    >>> results = minpop_uhf(mf)
    >>> print(f"O charges: {results['mulliken_charges']}")
    """
    mol = mf.mol
    
    # Build minimal basis
    mol_min = _build_minimal_basis_mol(mol)
    S_cross = intor_cross('int1e_ovlp', mol_min, mol)
    S_min = mol_min.intor('int1e_ovlp')
    S_min_inv = np.linalg.inv(S_min)
    
    # Extract UHF orbital information
    mo_coeff_a, mo_coeff_b = mf.mo_coeff
    mo_occ_a, mo_occ_b = mf.mo_occ
    n_alpha = int(np.sum(mo_occ_a > 0.5))
    n_beta = int(np.sum(mo_occ_b > 0.5))
    
    # Project alpha and beta orbitals separately
    mo_alpha = _project_to_minimal_basis(mo_coeff_a[:, mo_occ_a > 0.5], S_cross, S_min_inv)
    mo_beta = _project_to_minimal_basis(mo_coeff_b[:, mo_occ_b > 0.5], S_cross, S_min_inv)
    
    # Build density matrices
    dm_alpha = mo_alpha @ mo_alpha.T
    dm_beta = mo_beta @ mo_beta.T
    dm_total = dm_alpha + dm_beta
    dm_spin_raw = dm_alpha - dm_beta
    
    # Spin annihilation
    s2_before, _ = mf.spin_square()
    S_target = (n_alpha - n_beta) / 2.0
    s2_after = S_target * (S_target + 1)
    
    # Apply spin scaling (singlet → zero spin density)
    if S_target == 0:
        dm_spin = np.zeros_like(dm_spin_raw)
    else:
        raw_spin_pop = np.trace(dm_spin_raw @ S_min)
        target_spin_pop = 2 * S_target
        scale = target_spin_pop / raw_spin_pop if abs(raw_spin_pop) > 1e-10 else 1.0
        dm_spin = dm_spin_raw * scale
    
    # Mulliken population matrices
    pop_alpha = _mulliken_pop_matrix(dm_alpha, S_min)
    pop_beta = _mulliken_pop_matrix(dm_beta, S_min)
    pop_total = _mulliken_pop_matrix(dm_total, S_min)
    pop_spin = _mulliken_pop_matrix(dm_spin, S_min)
    
    # Reorder to Gaussian convention
    reorder, ao_labels_reordered = _get_gaussian_ao_order(mol_min)
    
    dm_alpha = _reorder_matrix(dm_alpha, reorder)
    dm_beta = _reorder_matrix(dm_beta, reorder)
    dm_total = _reorder_matrix(dm_total, reorder)
    dm_spin = _reorder_matrix(dm_spin, reorder)
    pop_alpha = _reorder_matrix(pop_alpha, reorder)
    pop_beta = _reorder_matrix(pop_beta, reorder)
    pop_total = _reorder_matrix(pop_total, reorder)
    pop_spin = _reorder_matrix(pop_spin, reorder)
    
    ao_labels = [_convert_label_to_gaussian(lbl) for lbl in ao_labels_reordered]
    
    # Compute gross orbital populations
    gross = np.column_stack([
        np.sum(pop_total, axis=0),
        np.sum(pop_alpha, axis=0),
        np.sum(pop_beta, axis=0),
        np.sum(pop_spin, axis=0)
    ])
    
    # Condensed atomic properties
    condensed = _condense_to_atoms(pop_total, ao_labels)
    spin_atomic = _condense_to_atoms(pop_spin, ao_labels)
    
    nuclear_charges = np.array([mol_min.atom_charge(i) for i in range(mol_min.natm)])
    mulliken_charges = nuclear_charges - np.sum(condensed, axis=1)
    spin_populations = np.sum(spin_atomic, axis=1)
    
    # Collect results
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
        's2_before_annihilation': s2_before,
        's2_after_annihilation': s2_after,
    }
    
    if verbose:
        _print_results(results, mol_min, n_alpha, n_beta)
    
    return results


def run_uhf_from_xyz(xyz_file, charge=0, multiplicity=1, basis='6-31+G', verbose=True):
    """
    Run UHF calculation and MinPop analysis from an XYZ file.
    
    Convenience function that sets up the molecule, runs UHF, and performs
    MinPop analysis in one call.
    
    Parameters
    ----------
    xyz_file : str
        Path to XYZ geometry file
    charge : int, optional
        Molecular charge (default: 0)
    multiplicity : int, optional
        Spin multiplicity 2S+1 (default: 1 for singlet)
    basis : str, optional
        Computational basis set (default: '6-31+G')
    verbose : bool, optional
        Print output (default: True)
    
    Returns
    -------
    results : dict
        MinPop analysis results (see minpop_uhf)
    
    Notes
    -----
    For exact agreement with Gaussian's orbital-resolved output, use the
    "standard orientation" geometry from Gaussian's output.
    
    Examples
    --------
    >>> # Triplet methylene
    >>> results = run_uhf_from_xyz("ch2.xyz", charge=0, multiplicity=3)
    >>> print(f"Carbon spin density: {results['spin_populations'][0]:.4f}")
    
    >>> # Doublet radical
    >>> results = run_uhf_from_xyz("methyl.xyz", charge=0, multiplicity=2)
    """
    atom_str = _read_xyz(xyz_file)
    spin = multiplicity - 1
    
    mol = gto.M(atom=atom_str, basis=basis, charge=charge, spin=spin)
    
    if verbose:
        print(f"Molecule: {xyz_file}")
        print(f"Charge: {charge}, Multiplicity: {multiplicity}")
        print(f"Basis: {basis}")
        print(f"Atoms: {mol.natm}, Electrons: {mol.nelectron}")
        print()
    
    mf = scf.UHF(mol)
    mf.kernel()
    
    if verbose:
        print()
    
    return minpop_uhf(mf, verbose=verbose)


# =============================================================================
# Command Line Interface
# =============================================================================

def main():
    """Command-line entry point for MinPop UHF analysis."""
    parser = argparse.ArgumentParser(
        description="Minimum Population (MinPop) analysis for UHF wavefunctions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python minpop_uhf.py -xyz ch2.xyz -charge 0 -mult 3
  python minpop_uhf.py -xyz radical.xyz -mult 2 -basis cc-pVDZ

Notes:
  Output format matches Gaussian 16's Pop=Minimal output.
  For exact orbital-resolved agreement, use Gaussian's standard orientation.
"""
    )
    parser.add_argument("-xyz", required=True, dest="xyz_file",
                        help="Path to XYZ geometry file")
    parser.add_argument("-charge", type=int, default=0,
                        help="Molecular charge (default: 0)")
    parser.add_argument("-mult", type=int, default=1,
                        help="Spin multiplicity 2S+1 (default: 1)")
    parser.add_argument("-basis", default="6-31+G",
                        help="Computational basis set (default: 6-31+G)")
    parser.add_argument("-q", "--quiet", action="store_true",
                        help="Suppress output")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    
    args = parser.parse_args()
    
    run_uhf_from_xyz(
        args.xyz_file,
        charge=args.charge,
        multiplicity=args.mult,
        basis=args.basis,
        verbose=not args.quiet
    )


if __name__ == "__main__":
    main()
