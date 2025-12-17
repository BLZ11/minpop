# MinPop: Minimum Population Localization Analysis for PySCF

A Python implementation of Minimum Population (MinPop) localization analysis for open-shell Hartree-Fock wavefunctions. Projects molecular orbitals onto a minimal basis set (STO-3G) for chemically intuitive population analysis with output matching Gaussian 16.

## Overview

MinPop provides an alternative to standard Mulliken population analysis by projecting molecular orbitals from a large extended basis onto a minimal basis set. This yields atomic charges and spin populations that are less basis-set dependent and more chemically meaningful.

The method is closely related to **Pipek-Mezey localization**, which maximizes atomic Mulliken populations to produce localized orbitals. MinPop achieves similar chemical interpretability by performing Mulliken analysis in a minimal basis, where the populations are more robust and less sensitive to extended basis set size.

**Supported methods:**
- ROHF (Restricted Open-Shell Hartree-Fock)
- UHF (Unrestricted Hartree-Fock)

## Theory

The MinPop method projects MO coefficients $\mathbf{C}$ from an extended basis onto a minimal basis using:

$$\mathbf{C}' = \mathbf{S}^{-1} \cdot \bar{\mathbf{S}} \cdot \mathbf{C} \cdot \mathbf{M}^{-1/2}$$

where:
- $\mathbf{S}$ is the minimal basis overlap matrix
- $\bar{\mathbf{S}}$ is the cross-overlap matrix between minimal and extended basis set
- $\mathbf{M} = \mathbf{C}^T \cdot \bar{\mathbf{S}}^T \cdot \bar{\mathbf{S}}^{-1} \cdot \bar{\mathbf{S}} \cdot \mathbf{C}$ ensures Löwdin orthonormality

### Connection to Pipek-Mezey

Pipek-Mezey localization finds unitary transformations that maximize the sum of squared Mulliken atomic populations:

$$\xi_{PM} = \sum_A \sum_i |q_A^i|^2$$

MinPop achieves a similar goal by projecting orbitals to a minimal basis where Mulliken populations are inherently more localized and chemically meaningful. Both methods leverage Mulliken's partitioning scheme but approach the problem differently:
- **Pipek-Mezey**: Transforms orbitals to maximize atomic character
- **MinPop**: Projects to minimal basis where atomic character is naturally enhanced

### ROHF vs UHF

The key algorithmic difference between ROHF and UHF lies in how orbitals are projected:

| Method | Projection Strategy | Reason |
|--------|---------------------|--------|
| ROHF | Project all occupied orbitals **together**, then split into α/β | α and β share spatial functions |
| UHF | Project α and β **separately** | α and β have independent spatial functions |

## Installation

### Requirements
- Python 3.7+
- NumPy
- SciPy
- PySCF

### Install dependencies
```bash
pip install numpy scipy pyscf
```

### Download
```bash
git clone https://github.com/BLZ11/minpop.git
cd minpop
```

## Usage

### Command Line

```bash
# ROHF analysis
python minpop_rohf.py -xyz molecule.xyz -charge 0 -mult 3 -basis 6-31+G

# UHF analysis
python minpop_uhf.py -xyz molecule.xyz -charge 0 -mult 3 -basis 6-31+G
```

**Arguments:**
| Argument | Description | Default |
|----------|-------------|---------|
| `-xyz` | Path to XYZ geometry file | Required |
| `-charge` | Molecular charge | 0 |
| `-mult` | Spin multiplicity (2S+1) | 1 |
| `-basis` | Computational basis set | 6-31+G |
| `-q, --quiet` | Suppress output | False |

### Python API

```python
from minpop_rohf import minpop_rohf, run_rohf_from_xyz
from minpop_uhf import minpop_uhf, run_uhf_from_xyz

# From XYZ file
results = run_rohf_from_xyz("molecule.xyz", charge=0, multiplicity=3)
results = run_uhf_from_xyz("molecule.xyz", charge=0, multiplicity=3)

# From existing PySCF calculation
from pyscf import gto, scf

mol = gto.M(atom='C 0 0 0; H 0 1 0; H 0 -1 0', basis='6-31+G', spin=2)
mf = scf.ROHF(mol).run()
results = minpop_rohf(mf)
```

### Output

The analysis returns a dictionary containing:

| Key | Description |
|-----|-------------|
| `mulliken_charges` | Atomic partial charges |
| `spin_populations` | Atomic spin populations |
| `dm_alpha`, `dm_beta` | Spin density matrices in minimal basis |
| `dm_total`, `dm_spin` | Total and spin density matrices |
| `gross_orbital_pop` | Per-orbital populations [total, α, β, spin] |
| `condensed_to_atoms` | Atom-atom population matrix |
| `spin_atomic` | Atom-atom spin density matrix |
| `ao_labels` | Minimal basis AO labels |

## Example

### Triplet CH₂ (methylene)

**Input:** `ch2.xyz`
```
3

C    0.000000    0.000000    0.103184
H    0.000000    0.995222   -0.309552
H    0.000000   -0.995222   -0.309552
```

**Command:**
```bash
python minpop_rohf.py -xyz ch2.xyz -charge 0 -mult 3
```

**Output (excerpt):**
```
ROHF orbital structure: 3 doubly occupied, 2 singly occupied
============================================================
MinPop Analysis (ROHF)
============================================================
...
MBS Mulliken charges and spin densities:
                   1          2
     1  C   -0.260502   1.964131
     2  H    0.130251   0.017935
     3  H    0.130251   0.017935
Sum of MBS Mulliken charges =   0.00000   2.00000
```

## Gaussian 16 Compatibility

The output format is designed to match Gaussian 16's MinPop analysis (`Pop=(Full) IOp(6/27=122,6/12=3)`). For exact numerical agreement with Gaussian's orbital-resolved output (density matrices, gross orbital populations), use Gaussian's **standard orientation** geometry in your XYZ file.

You can extract the standard orientation from Gaussian output:
```
Standard orientation:
---------------------------------------------------------------------
Center     Atomic      Atomic             Coordinates (Angstroms)
Number     Number       Type             X           Y           Z
---------------------------------------------------------------------
     1          6           0        0.000000    0.000000    0.103184
     2          1           0        0.000000    0.995222   -0.309552
     3          1           0        0.000000   -0.995222   -0.309552
---------------------------------------------------------------------
```

**Note:** Atomic charges and spin populations are rotationally invariant and will match regardless of molecular orientation.

## References

### MinPop Method
1. Montgomery, J. A., Jr.; Frisch, M. J.; Ochterski, J. W.; Petersson, G. A. *J. Chem. Phys.* **1999**, *110*, 2822-2827. [DOI: 10.1063/1.477924](https://doi.org/10.1063/1.477924)

2. Montgomery, J. A., Jr.; Frisch, M. J.; Ochterski, J. W.; Petersson, G. A. *J. Chem. Phys.* **2000**, *112*, 6532-6542. [DOI: 10.1063/1.481224](https://doi.org/10.1063/1.481224)

### Related: Pipek-Mezey Localization
3. Pipek, J.; Mezey, P. G. *J. Chem. Phys.* **1989**, *90*, 4916-4926. [DOI: 10.1063/1.456588](https://doi.org/10.1063/1.456588)

## License

MIT License

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.

## Citation

If you use this code in your research, please cite the original MinPop references above and this repository.
