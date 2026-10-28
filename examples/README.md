# Examples

`xyz/` holds the diatomics used in the paper-style validation at their experimental bond lengths
(Pt2 2.333, PtO 1.728, PtH 1.53, Cr2 1.679, CrO 1.615, CrH 1.655 Å) plus H2, H2O, CH2, CO.

`run_pt_pbe.sh` reproduces `outputs/pt*_*.out` (UKS/PBE, ANO-RCC-VDZP, X2C, ANO-RCC-MB, σ/π/δ matrices).
`run_rhco3.sh` runs Rh(CO)3 (triplet, UKS/PBE) through five basis / minimal-basis
combinations; `xyz/rhco3.xyz` is a C3v guess to be replaced by an optimized geometry.
`run_cr_gvbpp.sh` reproduces `outputs/cr*_gvb.out` (GVB-PP followed by MinPop); `cr2_rhf.out` and
`cr2_rks_pbe.out` are the single-reference comparisons discussed in the README.

Run from this directory. Each Pt or Cr job takes minutes to tens of minutes on one core;
the GVB-PP Cr2 jobs are the longest.
