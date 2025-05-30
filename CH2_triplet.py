import pyscf
import numpy as np
import pyscf.lo
import pyscf.gto
import scipy.linalg

atomspec = '''C   0.122391   -0.000000    0.088352
              H  -0.125474   -0.000000    1.136866
              H   1.039556    0.000000   -0.477002'''
spin = 2

def calc_dm(C, Sbar, Sprime, Sprimeinv, minimal, O):
    """Returns the MBS density matrix for alpha/beta spin state"""
    
    ## MinPop Code
    P = Sbar @ C
    Cprime = P @ scipy.linalg.sqrtm(np.linalg.inv(C.T @ Sbar.T @ Sprimeinv @ P))
    Cprime = Sprimeinv @ Cprime

    ## Pipek-Mezey localization method using Mulliken charges
    pm = pyscf.lo.PM(minimal, Cprime) 
    pm.pop_method = "mulliken"
    loc_orb = pm.kernel()

    dm = (loc_orb * O) @ loc_orb.T 
    return dm

# Original code from: https://github.com/NablaChem/prototype/blob/b92ec8b6f2a7f864056c941adec23c2c31005b9b/minpop/demo.py#L10
def minpop(calculation: pyscf.scf.UHF,spin) -> np.ndarray:
    """Implements the minimum population localization method.

    Follows:
    A complete basis set model chemistry. VII. Use of the minimum population localization method
    J. A. Montgomery, Jr., M. J. Frisch, and J. W. Ochterski, G. A. Petersson
    DOI 10.1063/1.481224, eqn 3
    correction: typo in eq 3: the S'^(-1/2) missing as prefactor

    Parameters
    ----------
    calculation : pyscf.scf.UHF
        A PySCF calculator object after a converged unrestricted SCF.

    Returns
    -------
    np.ndarray
        Populations on atoms. 2D, natoms x natoms.
    """
    minimal = pyscf.gto.M(atom=calculation.mol.atom, basis="STO-3G", spin=spin)

    Sbar = pyscf.gto.intor_cross("int1e_ovlp", minimal, calculation.mol)
    C = [calculation.mo_coeff[i][:, calculation.mo_occ[i] > 0] for i in range(2)]
    O = [calculation.mo_occ[i][calculation.mo_occ[i] > 0] for i in range(2)]
    Sprime = minimal.intor("int1e_ovlp")
    Sprimeinv = np.linalg.inv(Sprime)
    
    dm_alpha = calc_dm(C[0], Sbar, Sprime, Sprimeinv, minimal, O[0]) # Alpha MBS Density Matrix
    print(f'Alpha MBS Density Matrix:\n{dm_alpha}')
    dm_beta = calc_dm(C[1], Sbar, Sprime, Sprimeinv, minimal, O[1]) # Beta MBS Density Matrix
    print(f'Beta MBS Density Matrix:\n{dm_beta}')
    
    dm = dm_alpha + dm_beta
    pop = np.einsum("ij,ji->ij", dm, Sprime).real # Full MBS Mulliken population analysis
    print(f'Full MBS Mulliken population analysis:\n{pop}')
    
    gross_pop = np.sum(pop,axis=0) # MBS Gross orbital populations (Total column only)
    print(f'MBS Gross orbital populations:\n{gross_pop}')

    population = np.zeros((minimal.natm, minimal.natm))
    labels = minimal.ao_labels(fmt=None)
    for i, si in enumerate(labels):
        for j, sj in enumerate(labels):
            population[si[0], sj[0]] += pop[i, j] # MBS Condensed to Atoms (all electrons)
            
    atomic_number = {'H':1,'He':2,'Li':3,'Be':4,'B':5,'C':6,'N':7,'O':8,'F':9,'Ne':10,
                     'Na':11,'Mg':12,'Al':13,'Si':14,'P':15,'S':16,'Cl':17,'Ar':18}
    atoms = np.array([calculation.mol.atom_symbol(i) for i in range(calculation.mol.natm)])
    charges = np.array([atomic_number[atoms[i]] - np.sum(population[:,i]) for i in range(atoms.shape[0])]) # MBS Mulliken charges and spin densities (first column only)
    print(f'MBS Mulliken Charges:\n{charges}')
    return population

calculation = pyscf.scf.UHF(pyscf.gto.M(atom=atomspec, basis="6-31+G", spin = spin))
calculation.kernel()
custom = minpop(calculation, spin)
print(f'MBS condensed to atoms (all electrons):\n{custom}')