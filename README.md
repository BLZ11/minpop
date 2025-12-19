# MinPop: Minimum Population Localization Analysis

**Gaussian 16-compatible population analysis for open-shell Hartree-Fock wavefunctions using PySCF**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PySCF](https://img.shields.io/badge/PySCF-2.0+-green.svg)](https://pyscf.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

MinPop projects Hartree-Fock molecular orbitals onto a chemically intuitive minimal basis set, enabling population analysis that reveals bonding character and charge distribution. This implementation produces output identical to Gaussian 16's `Pop=(Full) IOp(6/27=122,6/12=3)` keyword combination, facilitating direct comparison between codes.

### Features

- **UHF and ROHF support** — Handles both unrestricted and restricted open-shell wavefunctions
- **Gaussian-compatible output** — Matches Gaussian 16's formatting for easy validation
- **Automatic basis selection** — Follows Gaussian's convention (STO-3G, STO-3G*)
- **Spin annihilation** — Removes first spin contaminant from UHF wavefunctions
- **Transition metal support** — Correct handling of 3d/4s orbital ordering

## Theory

### The MinPop Method

The Minimum Population (MinPop) method projects extended basis set MO coefficients onto a minimal basis to obtain chemically meaningful populations. The projection is:

```
C' = S_min⁻¹ · S_cross · C · M^(-1/2)
```

where:
- **S_min** — Overlap matrix in minimal basis
- **S_cross** — Cross-overlap ⟨minimal|extended⟩  
- **M** — Metric matrix ensuring orthonormality

### UHF vs ROHF Implementation

The key difference between UHF and ROHF lies in how alpha and beta orbitals are treated:

| Aspect | UHF | ROHF |
|--------|-----|------|
| **Spatial orbitals** | α and β have different spatial parts | α and β share same spatial functions |
| **Projection** | Project α and β independently | Project once, partition by occupation |
| **Spin contamination** | Present (requires annihilation) | Absent (pure spin state) |
| **Density matrices** | D_α ≠ D_β even for paired electrons | D_α = D_β for doubly occupied |
| **Spin density** | Scaled by S(S+1)/⟨S²⟩ | Exact (no correction needed) |

**UHF**: Alpha and beta electrons see different effective potentials, leading to different spatial wavefunctions. This introduces spin contamination that must be removed via Löwdin annihilation.

**ROHF**: Doubly occupied orbitals have identical alpha and beta spatial functions. Only singly occupied orbitals contribute to spin density, giving a pure spin eigenstate.

### Minimal Basis Selection

Following Gaussian's convention:

| Elements | Basis | d-orbitals |
|----------|-------|------------|
| H–Ne (1st row) | STO-3G | — |
| Na–Ar (2nd row) | STO-3G* | Cartesian (6D) |
| K–Xe (3rd row+) | STO-3G | Spherical (5D) |

### UHF Spin Annihilation

UHF wavefunctions contain spin contamination from higher spin states. The first contaminant (S+1) is removed using Löwdin's projection:

```
⟨S²⟩_after = S(S+1)
```

For singlets, the spin-annihilated spin density is exactly zero.

## Installation

### Requirements

- Python ≥ 3.8
- NumPy
- PySCF ≥ 2.0

### Install

```bash
# Clone repository
git clone https://github.com/BLZ11/minpop.git
cd minpop

# Install dependencies
pip install numpy pyscf
```

## Usage

### Command Line

```bash
# UHF triplet calculation
python minpop_uhf.py -xyz ch2.xyz -charge 0 -mult 3

# ROHF doublet calculation  
python minpop_rohf.py -xyz methyl.xyz -charge 0 -mult 2

# Custom basis set
python minpop_uhf.py -xyz molecule.xyz -mult 3 -basis cc-pVDZ
```

### Python API

```python
from minpop_uhf import minpop_uhf, run_uhf_from_xyz
from minpop_rohf import minpop_rohf, run_rohf_from_xyz

# From XYZ file
results = run_uhf_from_xyz("ch2.xyz", charge=0, multiplicity=3)
print(f"Carbon charge: {results['mulliken_charges'][0]:.4f}")
print(f"Carbon spin: {results['spin_populations'][0]:.4f}")

# From existing PySCF calculation
from pyscf import gto, scf

mol = gto.M(atom='O 0 0 0; O 0 0 1.2', basis='6-31G*', spin=2)
mf = scf.UHF(mol).run()
results = minpop_uhf(mf)
```

### Output

The output format matches Gaussian 16 exactly:

```
UHF orbital structure: 5 alpha, 3 beta
 Annihilation of the first spin contaminant:
 S**2 before annihilation   2.0172,   after   2.0000
============================================================
MinPop Analysis (UHF)
============================================================
     Alpha  MBS Density Matrix:
                           1         2         3         4         5
   1 1   C  1S         1.05126
   2        2S        -0.19228   0.77310
   ...
     MBS Gross orbital populations:
                         Total     Alpha     Beta      Spin
   1 1   C  1S         1.99596   0.99838   0.99758   0.00080
   ...
 MBS Mulliken charges and spin densities:
                  1          2
        1  C   -0.255994   2.203610
        2  H    0.127997  -0.101805
```

## Return Values

Both `minpop_uhf()` and `minpop_rohf()` return a dictionary containing:

| Key | Description |
|-----|-------------|
| `dm_alpha`, `dm_beta` | Alpha/beta density matrices in minimal basis |
| `dm_total`, `dm_spin` | Total and spin density matrices |
| `gross_orbital_pop` | Per-orbital populations [total, α, β, spin] |
| `condensed_to_atoms` | Atom-atom population matrix |
| `mulliken_charges` | Atomic partial charges |
| `spin_populations` | Atomic spin densities |
| `ao_labels` | Minimal basis AO labels |
| `s2_before_annihilation` | ⟨S²⟩ before projection (UHF only) |
| `s2_after_annihilation` | ⟨S²⟩ after projection (UHF only) |

## Validation

Tested against Gaussian 16 for:

| System | Method | Elements | d-orbital type |
|--------|--------|----------|----------------|
| CH (quartet) | ROHF | C, H | — |
| CH₂ (triplet) | UHF, ROHF | C, H | — |
| CHCl₃ | ROHF | C, H, Cl | Cartesian (6D) |
| TiO₂ | UHF, ROHF | Ti, O | Spherical (5D) |

**Agreement**: Mulliken charges and spin populations match to 6 decimal places.

### Reproducing Gaussian Results

For exact agreement with Gaussian's orbital-resolved density matrices, use Gaussian's **standard orientation** geometry:

1. Run Gaussian with `Pop=(Full) IOp(6/27=122,6/12=3)`
2. Extract coordinates from "Standard orientation" section
3. Use these coordinates in your XYZ file

## Comparison with Gaussian

### What Matches Exactly
- ✅ Mulliken charges
- ✅ Spin populations  
- ✅ Gross orbital populations
- ✅ Condensed atom-atom populations
- ✅ S² values (before/after annihilation)

### Known Differences
- ⚠️ Density matrix elements for Cartesian d-orbitals differ by normalization factors (populations still match)

## References

1. Montgomery Jr., J. A.; Frisch, M. J.; Ochterski, J. W.; Petersson, G. A. *J. Chem. Phys.* **1999**, *110*, 2822–2827. [DOI: 10.1063/1.477924](https://doi.org/10.1063/1.477924)

2. Montgomery Jr., J. A.; Frisch, M. J.; Ochterski, J. W.; Petersson, G. A. *J. Chem. Phys.* **2000**, *112*, 6532–6542. [DOI: 10.1063/1.481224](https://doi.org/10.1063/1.481224)

3. Löwdin, P.-O. *Phys. Rev.* **1955**, *97*, 1509–1520. [DOI: 10.1103/PhysRev.97.1509](https://doi.org/10.1103/PhysRev.97.1509)

## License

MIT License — see [LICENSE](LICENSE) for details.

## Contributing

Contributions welcome! Please open an issue or pull request.

## Authors

- **Barbaro Zulueta** — [GitHub](https://github.com/BLZ11)
