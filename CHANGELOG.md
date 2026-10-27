# Changelog

## 2.0.0 (2026-10-27)

Added
- `-minimal-basis ano-rcc-mb` (H to Cm, all-electron) and `sto-3g-ano` (the sto-3g rule, STO-3G / STO-3G* on Na–Ar / STO-3G, for H to Xe and ANO-RCC-MB beyond; identical output to `sto-3g` up to Xe).
- `minpop/basis_sets/anorcc_basis_pyscf.py`: ANO-RCC-MB/VDZP/VTZP/VQZP sliced from PySCF's ANO-RCC; verified against Basis Set Exchange.
- `-x2c` scalar-relativistic Hamiltonian in every driver.
- `-basis-light` / `-light-max-z`: per-element computational basis split with per-side ECP detection.
- `-symmetry` / `-symmetry-shells` (alias `-sigma-pi`): exact σ/π/δ/φ resolution of bond populations (`minpop/bond_symmetry.py`), printed as MBS matrices condensed to atoms and shells for the total and spin density.
- `minpop_gvbpp.py`: MinPop on GVB-PP natural orbitals (keithgroup/gvbpp) with occupation-weighted projection, representability rule for correlating NOs, valence-restricted pair search, multi-start with seed control, per-pair bond table.
- "Projection completeness" diagnostic (captured electrons, leak, least-captured MO).
- `-label-style gaussian|physical`; Gaussian's labels remain the default for `sto-3g` and `sto-3g-ano`, physical labels are the default for `ano-rcc-mb`.
- Package layout with console commands (`minpop-rohf`, `minpop-uhf`, `minpop-rks`, `minpop-uks`, `minpop-gvbpp`, `minpop-export`), tests, examples.

Unchanged
- The STO-3G (Gaussian MBS) path is byte-identical to 1.x except for the added completeness line.

Known
- The legacy Gaussian shell-counter labels mark the 4d block of 5th-row atoms as "5D" where Gaussian prints "6D"; numbers and ordering are unaffected. Use `-label-style physical`.
