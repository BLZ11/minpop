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
import numpy as np
from pyscf import gto, scf
from pyscf.gto import intor_cross

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


# =============================================================================
# Minimal Basis Construction
# =============================================================================

def _is_transition_metal(atomic_number):
    """Check if atomic number corresponds to a transition metal."""
    return any(low <= atomic_number <= high for low, high in TRANSITION_METAL_RANGES)


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
        Molecule with minimal basis set (STO-3G or STO-3G*)
        
    Notes
    -----
    - Second-row elements (Na–Ar) use STO-3G* with Cartesian d-orbitals (6D)
    - Transition metals use STO-3G with spherical d-orbitals (5D)
    - Mixed systems default to spherical when transition metals are present
    """
    from pyscf.data import elements
    
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
    
    # Use Cartesian d-orbitals only for pure second-row systems
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


# =============================================================================
# AO Reordering: PySCF → Gaussian Convention
# =============================================================================

def _get_gaussian_ao_order(mol_min):
    """
    Generate permutation to reorder AOs from PySCF to Gaussian convention.
    
    Gaussian orders AOs by increasing shell number within each atom:
        1S, 2S, 2PX, 2PY, 2PZ, 3S, 3PX, 3PY, 3PZ, 4D, ...
    
    For d-orbitals:
        - Cartesian (STO-3G*): XX, YY, ZZ, XY, XZ, YZ
        - Spherical (STO-3G):  D0, D+1, D-1, D+2, D-2
    """
    from pyscf.data import elements
    
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
        
        if is_transition_metal:
            if orb == '3d': shell = 4
            elif orb == '4d': shell = 5
        
        if is_5th_period:
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
    
    for col_start in range(0, n, 5):
        col_end = min(col_start + 5, n)
        
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
                    ecp_dict[elem] = 'def2svp'
            except (KeyError, ValueError):
                pass
    
    return ecp_dict if ecp_dict else None


# =============================================================================
# Main Analysis Functions
# =============================================================================

def minpop_uhf(mf, verbose=True):
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
    mol_min = _build_minimal_basis_mol(mol)
    
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
    
    # Compute S² before annihilation
    s2_before = _compute_s2(mo_alpha, mo_beta, S_min)
    
    # Build density matrices
    dm_alpha = mo_alpha @ mo_alpha.T
    dm_beta = mo_beta @ mo_beta.T
    dm_total = dm_alpha + dm_beta
    dm_spin_raw = dm_alpha - dm_beta
    
    # Apply Löwdin spin annihilation
    dm_spin, _ = _lowdin_annihilate(dm_spin_raw, S_min, n_alpha, n_beta)
    
    S_target = (n_alpha - n_beta) / 2.0
    s2_after = S_target * (S_target + 1)
    
    # Mulliken population matrices
    pop_alpha = _mulliken_pop_matrix(dm_alpha, S_min)
    pop_beta = _mulliken_pop_matrix(dm_beta, S_min)
    pop_total = _mulliken_pop_matrix(dm_total, S_min)
    pop_spin = _mulliken_pop_matrix(dm_spin, S_min)
    
    # Reorder to Gaussian convention
    reorder, ao_labels_raw = _get_gaussian_ao_order(mol_min)
    
    dm_alpha = _reorder_matrix(dm_alpha, reorder)
    dm_beta = _reorder_matrix(dm_beta, reorder)
    dm_total = _reorder_matrix(dm_total, reorder)
    dm_spin = _reorder_matrix(dm_spin, reorder)
    pop_alpha = _reorder_matrix(pop_alpha, reorder)
    pop_beta = _reorder_matrix(pop_beta, reorder)
    pop_total = _reorder_matrix(pop_total, reorder)
    pop_spin = _reorder_matrix(pop_spin, reorder)
    
    ao_labels = [_convert_label_to_gaussian(lbl) for lbl in ao_labels_raw]
    
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
        's2_before_annihilation': s2_before,
        's2_after_annihilation': s2_after,
    }
    
    if verbose:
        _print_results(results, mol_min, n_alpha, n_beta)
    
    return results


def run_uhf_from_xyz(xyz_file, charge=0, multiplicity=1, basis='6-31+G',
                     ecp=None, verbose=True):
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
    
    # Auto-detect ECP for heavy elements
    if ecp is None:
        ecp = _get_ecp_dict(basis, atom_str)
        if ecp and verbose:
            print(f"Auto-detected ECP for heavy elements: {ecp}")
    
    mol = gto.M(
        atom=atom_str,
        basis=basis,
        charge=charge,
        spin=multiplicity - 1,
        ecp=ecp
    )
    
    if verbose:
        print(f"Molecule: {xyz_file}")
        print(f"Charge: {charge}, Multiplicity: {multiplicity}")
        print(f"Basis: {basis}")
        print(f"Atoms: {mol.natm}, Electrons: {mol.nelectron}")
        print()
    
    # SCF with Gaussian-like defaults
    mf = scf.UHF(mol)
    mf.max_cycle = 128
    mf.conv_tol = 1e-8
    mf.conv_tol_grad = 1e-6
    mf.diis_space = 8
    mf.level_shift = 0.0
    mf.kernel()
    
    if verbose:
        print()
    
    return minpop_uhf(mf, verbose=verbose)


# =============================================================================
# Command Line Interface
# =============================================================================

def main():
    """Command-line entry point."""
    parser = argparse.ArgumentParser(
        description="MinPop analysis for UHF wavefunctions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python minpop_uhf.py -xyz ch2.xyz -charge 0 -mult 3
  python minpop_uhf.py -xyz radical.xyz -mult 2 -basis cc-pVDZ
  python minpop_uhf.py -xyz snh4.xyz -basis def2-TZVPP  # Auto-detects ECP

Notes:
  Output matches Gaussian 16's Pop=(Full) IOp(6/27=122,6/12=3) format.
  ECP auto-detected for def2 basis sets with heavy elements (Z > 36).
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
    parser.add_argument("-ecp", default=None,
                        help="ECP (auto-detected for def2 + heavy elements)")
    parser.add_argument("-q", "--quiet", action="store_true",
                        help="Suppress output")
    parser.add_argument("--version", action="version",
                        version=f"%(prog)s {__version__}")
    
    args = parser.parse_args()
    
    run_uhf_from_xyz(
        args.xyz_file,
        charge=args.charge,
        multiplicity=args.mult,
        basis=args.basis,
        ecp=args.ecp,
        verbose=not args.quiet
    )


if __name__ == "__main__":
    main()
