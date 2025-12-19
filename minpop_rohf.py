#!/usr/bin/env python3
"""
Minimum Population (MinPop) Localization Analysis for ROHF Wavefunctions.

This module projects Restricted Open-shell Hartree-Fock (ROHF) molecular orbitals
onto a minimal basis set for chemically intuitive population analysis. Output
format matches Gaussian 16 for direct comparison.

Theory
------
The MinPop method projects extended basis set MO coefficients C onto a minimal
basis using the transformation:

    C' = S_min^(-1) · S_cross · C · M^(-1/2)

where:
    - S_min     : Overlap matrix in minimal basis
    - S_cross   : Cross-overlap matrix <minimal|extended>
    - M         : Metric matrix ensuring orthonormality, M = C'^T·S_min·C'

For ROHF wavefunctions, alpha and beta orbitals share the same spatial functions
(doubly occupied), with additional singly occupied orbitals having only alpha
electrons.

Minimal Basis Selection
-----------------------
Following Gaussian's convention:
    - First row (H-Ne):     STO-3G
    - Second row (Na-Ar):   STO-3G* (with d-polarization, Cartesian 6D)
    - Third row+ / TM:      STO-3G (spherical 5D for d-orbitals)

References
----------
[1] Montgomery Jr., J. A. et al. J. Chem. Phys. 110, 2822-2827 (1999).
[2] Montgomery Jr., J. A. et al. J. Chem. Phys. 112, 6532-6542 (2000).

Author: Barbaro Zulueta (Pitt Quantum Repository)
"""

import argparse
import numpy as np
from pyscf import gto, scf
from pyscf.gto import intor_cross

__version__ = "1.0.0"
__author__ = "Barbaro Zulueta"
__all__ = ["minpop_rohf", "run_rohf_from_xyz"]

# Elements using STO-3G* (with d-polarization) for minimal basis
_SECOND_ROW = frozenset({'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar'})


def _build_minimal_basis_mol(mol):
    """Construct minimal basis molecule matching Gaussian's MinPop convention."""
    from pyscf.data import elements
    
    has_second_row = False
    has_transition_metal = False
    basis_dict = {}
    
    for i in range(mol.natm):
        symbol = mol.atom_symbol(i)
        elem = ''.join(c for c in symbol if c.isalpha())
        
        if elem in _SECOND_ROW:
            basis_dict[symbol] = 'STO-3G*'
            has_second_row = True
        else:
            basis_dict[symbol] = 'STO-3G'
            try:
                z = elements.charge(elem)
                if (21 <= z <= 30) or (39 <= z <= 48) or (57 <= z <= 80) or (89 <= z <= 112):
                    has_transition_metal = True
            except (KeyError, ValueError):
                pass
    
    use_cartesian = has_second_row and not has_transition_metal
    return gto.M(atom=mol.atom, basis=basis_dict, charge=mol.charge, spin=mol.spin, cart=use_cartesian)


def _project_to_minimal_basis(mo_coeff, S_cross, S_min_inv):
    """Project MO coefficients to minimal basis with orthonormalization."""
    C_proj = S_min_inv @ S_cross @ mo_coeff
    M = C_proj.T @ np.linalg.inv(S_min_inv) @ C_proj
    eigval, eigvec = np.linalg.eigh(M)
    eigval = np.maximum(eigval, 1e-14)
    M_invsqrt = eigvec @ np.diag(1.0 / np.sqrt(eigval)) @ eigvec.T
    return C_proj @ M_invsqrt


def _get_gaussian_ao_order(mol_min):
    """Generate permutation indices to reorder AOs from PySCF to Gaussian convention."""
    ao_labels = mol_min.ao_labels(fmt=None)
    has_cartesian_d = any(lbl[3].lower() in ('xx', 'yy', 'zz') for lbl in ao_labels if 'd' in lbl[2].lower())
    
    atom_aos = {}
    for i, lbl in enumerate(ao_labels):
        atom_idx = lbl[0]
        if atom_idx not in atom_aos:
            atom_aos[atom_idx] = []
        atom_aos[atom_idx].append((i, lbl))
    
    new_order, new_labels = [], []
    
    for atom_idx in sorted(atom_aos.keys()):
        aos = atom_aos[atom_idx]
        orbitals = [lbl[2].lower() for _, lbl in aos]
        is_transition_metal = '3d' in orbitals and '4s' in orbitals
        
        parsed = []
        for i, lbl in aos:
            orb, cart = lbl[2].lower(), (lbl[3] or '').lower()
            shell, ang, subtype, m_val = _parse_ao_label(orb, cart, is_transition_metal, has_cartesian_d)
            parsed.append((i, lbl, shell, ang, subtype, m_val))
        
        parsed.sort(key=lambda x: _ao_sort_key(x, has_cartesian_d))
        for item in parsed:
            new_order.append(item[0])
            new_labels.append(item[1:])
    
    return new_order, new_labels


def _parse_ao_label(orb, cart, is_transition_metal, has_cartesian_d):
    """Parse PySCF orbital label to Gaussian shell number and type."""
    if 's' in orb and not cart:
        shell = int(orb[0]) if orb[0].isdigit() else 1
        if is_transition_metal and orb == '4s': shell = 5
        elif is_transition_metal and orb == '5s': shell = 6
        return (shell, 0, '', 0)
    
    if 'p' in orb and cart in ('x', 'y', 'z'):
        shell = int(orb[0]) if orb[0].isdigit() else 2
        if is_transition_metal and orb == '4p': shell = 5
        elif is_transition_metal and orb == '5p': shell = 6
        return (shell, 1, cart, {'x': 1, 'y': -1, 'z': 0}.get(cart, 0))
    
    if has_cartesian_d and cart in ('xx', 'xy', 'xz', 'yy', 'yz', 'zz'):
        shell = 4 if orb == '3d' else (5 if orb == '4d' else 4)
        return (shell, 2, cart, 0)
    
    if 'd' in orb:
        shell = 4 if orb == '3d' else (5 if orb == '4d' else 4)
        m_val, subtype = _parse_spherical_d(cart)
        return (shell, 2, subtype, m_val)
    
    return (1, 0, '', 0)


def _parse_spherical_d(cart):
    """Parse spherical d-orbital label to magnetic quantum number."""
    cart_orig = cart
    cart = cart.lower()
    
    numeric = {'0': 0, '+0': 0, '-0': 0, '+1': 1, '1': 1, '-1': -1, '+2': 2, '2': 2, '-2': -2}
    if cart_orig in numeric:
        return (numeric[cart_orig], 'sph')
    
    text_to_m = {'z^2': 0, 'z2': 0, 'dz2': 0, 'xz': 1, 'dxz': 1, 'yz': -1, 'dyz': -1,
                 'x2-y2': 2, 'dx2-y2': 2, 'xy': -2, 'dxy': -2}
    if cart in text_to_m:
        return (text_to_m[cart], 'sph')
    
    try:
        return (int(cart_orig.replace('+', '')), 'sph')
    except (ValueError, AttributeError):
        return (0, 'sph')


def _ao_sort_key(parsed_ao, has_cartesian_d):
    """Generate sort key for AO ordering."""
    _, _, shell, ang, subtype, m_val = parsed_ao
    if ang == 1:
        return (shell, ang, {'x': 0, 'y': 1, 'z': 2}.get(subtype, 0))
    if ang == 2:
        if has_cartesian_d and subtype in ('xx', 'xy', 'xz', 'yy', 'yz', 'zz'):
            return (shell, ang, {'xx': 0, 'yy': 1, 'zz': 2, 'xy': 3, 'xz': 4, 'yz': 5}.get(subtype, 0))
        return (shell, ang, {0: 0, 1: 1, -1: 2, 2: 3, -2: 4}.get(m_val, 0))
    return (shell, ang, 0)


def _convert_label_to_gaussian(lbl_info):
    """Convert parsed AO label to Gaussian format."""
    lbl, shell, ang, subtype, m_val = lbl_info
    atom_idx, elem = lbl[0], lbl[1]
    
    if ang == 0: return (atom_idx, elem, f'{shell}S', '')
    if ang == 1: return (atom_idx, elem, f'{shell}P{subtype.upper()}', '')
    if ang == 2:
        if subtype in ('xx', 'xy', 'xz', 'yy', 'yz', 'zz'):
            return (atom_idx, elem, f'{shell}{subtype.upper()}', '')
        m_str = ' 0' if m_val == 0 else (f'+{m_val}' if m_val > 0 else str(m_val))
        return (atom_idx, elem, f'{shell}D{m_str}', '')
    return (atom_idx, elem, lbl[2].upper(), '')


def _reorder_matrix(matrix, order):
    """Reorder rows and columns of a symmetric matrix."""
    return matrix[np.ix_(order, order)]


def _mulliken_pop_matrix(dm, S):
    """Compute Mulliken population matrix."""
    return dm * S


def _condense_to_atoms(pop_matrix, ao_labels):
    """Condense orbital population matrix to atom-atom matrix."""
    n_atoms = max(lbl[0] for lbl in ao_labels) + 1
    condensed = np.zeros((n_atoms, n_atoms))
    for i, li in enumerate(ao_labels):
        for j, lj in enumerate(ao_labels):
            condensed[li[0], lj[0]] += pop_matrix[i, j]
    return condensed


def _format_value(v):
    """Format floating-point value matching Gaussian style."""
    if abs(v) < 1e-5:
        return f"{'0.00000':>10}" if v >= 0 else f"{'-0.00000':>10}"
    return f"{v:>10.5f}"


def _format_ao_label(idx, lbl, ao_labels):
    """Format AO row label matching Gaussian's fixed-width format."""
    orb_str = f"{lbl[2]}{lbl[3]}"
    if idx == 0 or ao_labels[idx][0] != ao_labels[idx - 1][0]:
        return f"{idx + 1:>4} {lbl[0] + 1:<1}   {lbl[1]:<2} {orb_str:<6}"
    return f"{idx + 1:>4}        {orb_str:<6}"


def _print_density_matrix(dm, ao_labels, title):
    """Print density matrix in Gaussian's column-blocked format."""
    n, block_size = dm.shape[0], 5
    print(f"     {title}:")
    
    for block_start in range(0, n, block_size):
        block_end = min(block_start + block_size, n)
        print("                  " + "".join(f"{i + 1:>10}" for i in range(block_start, block_end)))
        for i in range(block_start, n):
            cols = min(i + 1, block_end) - block_start
            if cols > 0:
                values = "".join(_format_value(dm[i, block_start + k]) for k in range(cols))
                print(f"{_format_ao_label(i, ao_labels[i], ao_labels)}{values}")


def _print_gross_populations(gross, ao_labels):
    """Print gross orbital populations table."""
    print("     MBS Gross orbital populations:")
    print("                         Total     Alpha     Beta      Spin")
    for i, lbl in enumerate(ao_labels):
        print(f"{_format_ao_label(i, lbl, ao_labels)}{gross[i,0]:>10.5f}{gross[i,1]:>10.5f}{gross[i,2]:>10.5f}{gross[i,3]:>10.5f}")


def _print_atomic_matrix(matrix, mol, title):
    """Print atom-atom matrix."""
    n_atoms, block_size = mol.natm, 6
    for block_start in range(0, n_atoms, block_size):
        block_end = min(block_start + block_size, n_atoms)
        header = "        " + "".join(f"{i + 1:>12}" for i in range(block_start, block_end))
        print(f"          MBS {title}:" if block_start == 0 else header)
        if block_start == 0: print(header)
        for i in range(n_atoms):
            print(f"     {i + 1:>2}  {mol.atom_symbol(i):<2}" + 
                  "".join(f"{matrix[i, j]:>12.6f}" for j in range(block_start, block_end)))


def _print_mulliken_summary(charges, spins, mol):
    """Print Mulliken charges and spin densities summary."""
    print(" MBS Mulliken charges and spin densities:")
    print("                  1          2")
    for i in range(mol.natm):
        print(f"        {i + 1}  {mol.atom_symbol(i):<2} {charges[i]:>10.6f} {spins[i]:>10.6f}")
    print(f" Sum of MBS Mulliken charges = {np.sum(charges):>10.5f} {np.sum(spins):>10.5f}")


def _print_results(results, mol_min, n_doubly, n_singly):
    """Print complete MinPop analysis in Gaussian format."""
    ao_labels = results['ao_labels']
    print(f"ROHF orbital structure: {n_doubly} doubly occupied, {n_singly} singly occupied")
    print("=" * 60)
    print("MinPop Analysis (ROHF)")
    print("=" * 60)
    
    _print_density_matrix(results['dm_alpha'], ao_labels, "Alpha  MBS Density Matrix")
    _print_density_matrix(results['dm_beta'], ao_labels, "Beta  MBS Density Matrix")
    _print_density_matrix(results['dm_total'], ao_labels, "Total  MBS Density Matrix")
    _print_density_matrix(results['dm_spin'], ao_labels, "Spin  MBS Density Matrix")
    _print_gross_populations(results['gross_orbital_pop'], ao_labels)
    _print_atomic_matrix(results['condensed_to_atoms'], mol_min, "Condensed to atoms (all electrons)")
    _print_atomic_matrix(results['spin_atomic'], mol_min, "Atomic-Atomic Spin Densities")
    _print_mulliken_summary(results['mulliken_charges'], results['spin_populations'], mol_min)
    print("=" * 60)


def _read_xyz(filename):
    """Parse XYZ file to PySCF atom specification string."""
    with open(filename, 'r') as f:
        lines = f.readlines()
    n_atoms = int(lines[0].strip())
    return "; ".join(f"{p[0]} {p[1]} {p[2]} {p[3]}" for p in (l.split() for l in lines[2:2 + n_atoms]))


def minpop_rohf(mf, verbose=True):
    """
    Perform MinPop population analysis on a converged ROHF calculation.
    
    Parameters
    ----------
    mf : pyscf.scf.rohf.ROHF
        Converged ROHF mean-field object
    verbose : bool, optional
        Print Gaussian-formatted output (default: True)
    
    Returns
    -------
    results : dict
        Analysis results including density matrices, populations, and charges
    """
    mol = mf.mol
    mol_min = _build_minimal_basis_mol(mol)
    S_cross = intor_cross('int1e_ovlp', mol_min, mol)
    S_min = mol_min.intor('int1e_ovlp')
    S_min_inv = np.linalg.inv(S_min)
    
    # ROHF orbital structure
    mo_occ = mf.mo_occ
    n_doubly = int(np.sum(mo_occ == 2))
    n_singly = int(np.sum(mo_occ == 1))
    n_occ = n_doubly + n_singly
    
    mo_min = _project_to_minimal_basis(mf.mo_coeff[:, :n_occ], S_cross, S_min_inv)
    mo_alpha = mo_min
    mo_beta = mo_min[:, :n_doubly]
    
    dm_alpha = mo_alpha @ mo_alpha.T
    dm_beta = mo_beta @ mo_beta.T
    dm_total = dm_alpha + dm_beta
    dm_spin = dm_alpha - dm_beta
    
    pop_alpha = _mulliken_pop_matrix(dm_alpha, S_min)
    pop_beta = _mulliken_pop_matrix(dm_beta, S_min)
    pop_total = _mulliken_pop_matrix(dm_total, S_min)
    pop_spin = _mulliken_pop_matrix(dm_spin, S_min)
    
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
    
    gross = np.column_stack([
        np.sum(pop_total, axis=0), np.sum(pop_alpha, axis=0),
        np.sum(pop_beta, axis=0), np.sum(pop_alpha, axis=0) - np.sum(pop_beta, axis=0)
    ])
    
    condensed = _condense_to_atoms(pop_total, ao_labels)
    spin_atomic = _condense_to_atoms(pop_spin, ao_labels)
    nuclear_charges = np.array([mol_min.atom_charge(i) for i in range(mol_min.natm)])
    mulliken_charges = nuclear_charges - np.sum(condensed, axis=1)
    spin_populations = np.sum(spin_atomic, axis=1)
    
    results = {
        'dm_alpha': dm_alpha, 'dm_beta': dm_beta, 'dm_total': dm_total, 'dm_spin': dm_spin,
        'pop_total': pop_total, 'pop_spin': pop_spin, 'gross_orbital_pop': gross,
        'condensed_to_atoms': condensed, 'spin_atomic': spin_atomic,
        'mulliken_charges': mulliken_charges, 'spin_populations': spin_populations,
        'ao_labels': ao_labels,
    }
    
    if verbose:
        _print_results(results, mol_min, n_doubly, n_singly)
    return results


def run_rohf_from_xyz(xyz_file, charge=0, multiplicity=1, basis='6-31+G', verbose=True):
    """Run ROHF calculation and MinPop analysis from an XYZ file."""
    atom_str = _read_xyz(xyz_file)
    mol = gto.M(atom=atom_str, basis=basis, charge=charge, spin=multiplicity - 1)
    
    if verbose:
        print(f"Molecule: {xyz_file}\nCharge: {charge}, Multiplicity: {multiplicity}")
        print(f"Basis: {basis}\nAtoms: {mol.natm}, Electrons: {mol.nelectron}\n")
    
    mf = scf.ROHF(mol)
    mf.kernel()
    if verbose: print()
    
    return minpop_rohf(mf, verbose=verbose)


def main():
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description="MinPop analysis for ROHF wavefunctions")
    parser.add_argument("-xyz", required=True, dest="xyz_file", help="XYZ geometry file")
    parser.add_argument("-charge", type=int, default=0, help="Molecular charge")
    parser.add_argument("-mult", type=int, default=1, help="Spin multiplicity 2S+1")
    parser.add_argument("-basis", default="6-31+G", help="Basis set")
    parser.add_argument("-q", "--quiet", action="store_true", help="Suppress output")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    
    args = parser.parse_args()
    run_rohf_from_xyz(args.xyz_file, args.charge, args.mult, args.basis, not args.quiet)


if __name__ == "__main__":
    main()
