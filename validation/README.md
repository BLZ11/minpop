# Validation data

`gaussian_outputs/` holds the Gaussian 16 reference calculations
(`Pop=(Full) IOp(6/27=122,6/12=3)`) and `pyscf_outputs/` the matching MinPop
reports for the systems used to validate the STO-3G (Gaussian MBS) path:
CH (quartet), CH2 (triplet, UHF and ROHF), CHCl3, TiO2 (UHF and ROHF), SnH4.
Mulliken charges, spin populations, gross orbital populations and the
condensed matrices agree to six decimals. The geometries in these files are
Gaussian's standard orientation; reuse them for any new comparison, since the
orbital-resolved matrices depend on the frame.

These are the `gaussian_outputs/` and `pyscf_outputs/` folders of the 1.x
repository, moved here unchanged. Version 2 keeps the STO-3G path
byte-identical to 1.x apart from the added "Projection completeness" line,
so the PySCF reports remain current.

For the new features (ANO-RCC-MB, X2C, σ/π/δ/φ, GVB-PP) there is no Gaussian
counterpart; their worked outputs are in `../examples/outputs/` and the
internal checks are listed in the top-level README.
