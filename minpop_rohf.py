#!/usr/bin/env python3
"""
Minimum Population (MinPop) Localization Analysis for ROHF Wavefunctions

Projects Restricted Open-Shell Hartree-Fock (ROHF) molecular orbitals onto a 
minimal basis set (STO-3G) for chemically intuitive population analysis.
Output format matches Gaussian 16.

Theory
------
MinPop projects MO coefficients C onto a minimal basis via:

    C' = S⁻¹ · S̄ · C · M^(-1/2)

where S is the minimal basis overlap, S̄ is the cross-overlap <min|ex>, 
and M = Cᵀ·S̄ᵀ·S⁻¹·S̄·C ensures orthonormality.

For ROHF, alpha and beta orbitals share spatial functions, so all occupied 
orbitals are projected together before splitting.

References
----------
[1] Montgomery et al., J. Chem. Phys. 110, 2822 (1999)
[2] Montgomery et al., J. Chem. Phys. 112, 6532 (2000)

Usage
-----
Command line:
    python minpop_rohf.py -xyz molecule.xyz -charge 0 -mult 3 -basis 6-31+G

Python API:
    from minpop_rohf import minpop_rohf, run_rohf_from_xyz
    results = run_rohf_from_xyz("molecule.xyz", charge=0, multiplicity=3)
    results = minpop_rohf(mf)  # From existing PySCF calculation

Note: For exact Gaussian agreement, use Gaussian's standard orientation geometry.
"""

import argparse
import numpy as np
import scipy.linalg
from pyscf import gto, scf
from pyscf.gto import intor_cross

__version__ = "1.0.0"
__all__ = ["minpop_rohf", "run_rohf_from_xyz"]


# =============================================================================
# Core Analysis Functions
# =============================================================================

def _project_to_minimal_basis(mo_coeff, S_cross, S_min_inv):
    """
    Project MO coefficients to minimal basis with symmetric orthogonalization.
    
    C' = S⁻¹ · S̄ · C · M^(-1/2), where M ensures orthonormality in minimal basis.
    """
    P = S_cross @ mo_coeff                          # Project to minimal basis
    M = mo_coeff.T @ S_cross.T @ S_min_inv @ P      # Orthonormalization metric
    M_inv_sqrt = scipy.linalg.sqrtm(np.linalg.inv(M))
    return np.real(S_min_inv @ P @ M_inv_sqrt)


def _mulliken_pop_matrix(D, S):
    """Mulliken population matrix: P_ij = D_ij · S_ji"""
    return np.einsum('ij,ji->ij', D, S).real


def _condense_to_atoms(P, ao_labels):
    """Sum AO population matrix elements to atomic contributions."""
    n_atoms = max(lbl[0] for lbl in ao_labels) + 1
    A = np.zeros((n_atoms, n_atoms))
    for i, li in enumerate(ao_labels):
        for j, lj in enumerate(ao_labels):
            A[li[0], lj[0]] += P[i, j]
    return A


# =============================================================================
# File I/O
# =============================================================================

def _read_xyz(filename):
    """Parse XYZ file to PySCF atom string."""
    with open(filename, 'r') as f:
        lines = f.readlines()
    n_atoms = int(lines[0].strip())
    atoms = []
    for line in lines[2:2 + n_atoms]:
        parts = line.split()
        atoms.append(f"{parts[0]} {parts[1]} {parts[2]} {parts[3]}")
    return "; ".join(atoms)


# =============================================================================
# Output Formatting (Gaussian 16 Compatible)
# =============================================================================

def _format_value(v):
    """Format number matching Gaussian's output style."""
    if abs(v) < 1e-5:
        return f"{'0.00000':>10}" if v >= 0 else f"{'-0.00000':>10}"
    return f"{v:>10.5f}"


def _format_ao_label(idx, lbl, ao_labels):
    """Format AO row label matching Gaussian style."""
    if idx == 0 or ao_labels[idx][0] != ao_labels[idx-1][0]:
        return f"{idx+1:>4} {lbl[0]+1}   {lbl[1]:<2} {lbl[2]}{lbl[3]:<2}"
    return f"{idx+1:>4}        {lbl[2]}{lbl[3]:<2}"


def _print_lower_triangular(matrix, ao_labels, title):
    """Print matrix in Gaussian's lower triangular format (5 columns per block)."""
    print(f"     {title}:")
    n = len(ao_labels)
    
    for col_start in range(0, n, 5):
        col_end = min(col_start + 5, n)
        
        # Column headers
        print(" " * 16 + "".join(f"{c+1:>10}" for c in range(col_start, col_end)))
        
        # Data rows
        for row in range(n):
            if row < col_start:
                continue
            row_str = f"{_format_ao_label(row, ao_labels[row], ao_labels):<16}"
            for col in range(col_start, min(col_end, row + 1)):
                row_str += _format_value(matrix[row, col])
            print(row_str)


def _print_atomic_matrix(matrix, mol, title):
    """Print atom-atom matrix in Gaussian format."""
    n = mol.natm
    print(f"          {title}:")
    print("          " + "".join(f"{i+1:>11}" for i in range(n)))
    for i in range(n):
        row = f"     {i+1:>4}  {mol.atom_symbol(i):<2}"
        row += "".join(f"{matrix[i,j]:>11.6f}" for j in range(n))
        print(row)


def _print_results(results, mol, n_doubly, n_singly):
    """Print complete MinPop analysis in Gaussian-compatible format."""
    ao_labels = results['ao_labels']
    gross = results['gross_orbital_pop']
    charges = results['mulliken_charges']
    spins = results['spin_populations']
    
    print(f"ROHF orbital structure: {n_doubly} doubly occupied, {n_singly} singly occupied")
    print("=" * 60)
    print("MinPop Analysis (ROHF)")
    print("=" * 60)
    
    # Density matrices
    _print_lower_triangular(results['dm_alpha'], ao_labels, "Alpha  MBS Density Matrix")
    _print_lower_triangular(results['dm_beta'], ao_labels, "Beta  MBS Density Matrix")
    _print_lower_triangular(results['pop_total'], ao_labels, "Full MBS Mulliken population analysis")
    
    # Gross orbital populations
    print("     MBS Gross orbital populations:")
    print(f"{'':>16}{'Total':>10}{'Alpha':>10}{'Beta':>10}{'Spin':>10}")
    for i, lbl in enumerate(ao_labels):
        row = f"{_format_ao_label(i, lbl, ao_labels):<16}"
        print(f"{row}{gross[i,0]:>10.5f}{gross[i,1]:>10.5f}{gross[i,2]:>10.5f}{gross[i,3]:>10.5f}")
    
    # Atomic matrices
    _print_atomic_matrix(results['condensed_to_atoms'], mol, "MBS Condensed to atoms (all electrons)")
    _print_atomic_matrix(results['spin_atomic'], mol, "MBS Atomic-Atomic Spin Densities")
    
    # Mulliken charges and spin densities
    print(" MBS Mulliken charges and spin densities:")
    print(f"{'':>13}{'1':>11}{'2':>11}")
    for i in range(mol.natm):
        print(f"     {i+1:>4}  {mol.atom_symbol(i):<2}{charges[i]:>11.6f}{spins[i]:>11.6f}")
    print(f" Sum of MBS Mulliken charges = {charges.sum():>9.5f} {spins.sum():>9.5f}")
    print("=" * 60)


# =============================================================================
# Main Analysis Functions
# =============================================================================

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
    dict
        Analysis results with keys:
        - dm_alpha, dm_beta: Spin density matrices in minimal basis
        - dm_total, dm_spin: Total and spin density matrices
        - pop_total, pop_spin: Mulliken population matrices
        - gross_orbital_pop: Per-orbital populations [total, alpha, beta, spin]
        - condensed_to_atoms: Atom-atom population matrix
        - spin_atomic: Atom-atom spin density matrix
        - mulliken_charges: Atomic partial charges
        - spin_populations: Atomic spin populations
        - ao_labels: Minimal basis AO labels
    
    Notes
    -----
    ROHF alpha/beta orbitals share spatial functions, so all occupied orbitals
    are projected together to ensure consistent orthogonalization before
    splitting into alpha (all occupied) and beta (doubly occupied only).
    """
    mol = mf.mol
    
    # === Setup minimal basis ===
    mol_min = gto.M(atom=mol.atom, basis='STO-3G', charge=mol.charge, spin=mol.spin)
    S_cross = intor_cross('int1e_ovlp', mol_min, mol)  # <minimal|extended>
    S_min = mol_min.intor('int1e_ovlp')
    S_min_inv = np.linalg.inv(S_min)
    
    # === Extract orbital information ===
    # ROHF occupation: 2 = doubly occupied, 1 = singly occupied, 0 = virtual
    mo_occ = mf.mo_occ
    n_doubly = int(np.sum(mo_occ >= 1.5))
    n_singly = int(np.sum((mo_occ >= 0.5) & (mo_occ < 1.5)))
    
    # === Project orbitals to minimal basis ===
    # Key: Project ALL occupied orbitals together, then split
    # This ensures consistent orthogonalization since alpha/beta share spatial functions
    occupied_mask = mo_occ > 0.5
    mo_min = _project_to_minimal_basis(mf.mo_coeff[:, occupied_mask], S_cross, S_min_inv)
    
    mo_alpha = mo_min                    # All occupied (doubly + singly)
    mo_beta = mo_min[:, :n_doubly]       # Only doubly occupied
    
    # === Build density matrices ===
    dm_alpha = mo_alpha @ mo_alpha.T
    dm_beta = mo_beta @ mo_beta.T
    dm_total = dm_alpha + dm_beta
    dm_spin = dm_alpha - dm_beta
    
    # === Population analysis ===
    ao_labels = mol_min.ao_labels(fmt=None)
    
    pop_alpha = _mulliken_pop_matrix(dm_alpha, S_min)
    pop_beta = _mulliken_pop_matrix(dm_beta, S_min)
    pop_total = _mulliken_pop_matrix(dm_total, S_min)
    pop_spin = _mulliken_pop_matrix(dm_spin, S_min)
    
    # Gross orbital populations (sum over columns)
    gross = np.column_stack([
        np.sum(pop_total, axis=0),
        np.sum(pop_alpha, axis=0),
        np.sum(pop_beta, axis=0),
        np.sum(pop_alpha, axis=0) - np.sum(pop_beta, axis=0)
    ])
    
    # === Atomic properties ===
    condensed = _condense_to_atoms(pop_total, ao_labels)
    spin_atomic = _condense_to_atoms(pop_spin, ao_labels)
    
    nuclear_charges = np.array([mol_min.atom_charge(i) for i in range(mol_min.natm)])
    mulliken_charges = nuclear_charges - np.sum(condensed, axis=1)
    spin_populations = np.sum(spin_atomic, axis=1)
    
    # === Collect results ===
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
    }
    
    if verbose:
        _print_results(results, mol_min, n_doubly, n_singly)
    
    return results


def run_rohf_from_xyz(xyz_file, charge=0, multiplicity=1, basis='6-31+G', verbose=True):
    """
    Run ROHF calculation and MinPop analysis from an XYZ file.
    
    Parameters
    ----------
    xyz_file : str
        Path to XYZ geometry file
    charge : int, optional
        Molecular charge (default: 0)
    multiplicity : int, optional
        Spin multiplicity 2S+1 (default: 1)
    basis : str, optional
        Computational basis set (default: '6-31+G')
    verbose : bool, optional
        Print output (default: True)
    
    Returns
    -------
    dict
        MinPop analysis results (see minpop_rohf)
    
    Notes
    -----
    For exact agreement with Gaussian's orbital-resolved output, use
    Gaussian's "standard orientation" geometry in the XYZ file.
    
    Example
    -------
    >>> results = run_rohf_from_xyz("ch2.xyz", charge=0, multiplicity=3)
    >>> print(f"Carbon charge: {results['mulliken_charges'][0]:.4f}")
    """
    # Parse geometry
    atom_str = _read_xyz(xyz_file)
    spin = multiplicity - 1  # PySCF uses n_alpha - n_beta
    
    # Build molecule
    mol = gto.M(atom=atom_str, basis=basis, charge=charge, spin=spin)
    
    if verbose:
        print(f"Molecule: {xyz_file}")
        print(f"Charge: {charge}, Multiplicity: {multiplicity}")
        print(f"Basis: {basis}")
        print(f"Atoms: {mol.natm}, Electrons: {mol.nelectron}")
        print()
    
    # Run SCF
    mf = scf.ROHF(mol)
    mf.kernel()
    
    if verbose:
        print()
    
    return minpop_rohf(mf, verbose=verbose)


# =============================================================================
# Command Line Interface
# =============================================================================

def main():
    """Command-line entry point."""
    parser = argparse.ArgumentParser(
        description="MinPop population analysis for ROHF wavefunctions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python minpop_rohf.py -xyz molecule.xyz -charge 0 -mult 3
    python minpop_rohf.py -xyz molecule.xyz -charge 0 -mult 3 -basis cc-pVDZ

Note:
    For exact Gaussian agreement, use Gaussian's standard orientation geometry.
        """
    )
    parser.add_argument("-xyz", required=True, dest="xyz_file",
                        help="Path to XYZ geometry file")
    parser.add_argument("-charge", type=int, default=0,
                        help="Molecular charge (default: 0)")
    parser.add_argument("-mult", type=int, default=1,
                        help="Spin multiplicity 2S+1 (default: 1)")
    parser.add_argument("-basis", type=str, default="6-31+G",
                        help="Basis set (default: 6-31+G)")
    parser.add_argument("-q", "--quiet", action="store_true",
                        help="Suppress output")
    
    args = parser.parse_args()
    
    run_rohf_from_xyz(
        args.xyz_file,
        charge=args.charge,
        multiplicity=args.mult,
        basis=args.basis,
        verbose=not args.quiet
    )


if __name__ == "__main__":
    main()
