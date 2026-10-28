#!/bin/bash
# Rh(CO)3, neutral doublet, UKS/PBE: eleven computational-basis / Hamiltonian /
# minimal-basis combinations. Each run prints populations, the total bond
# populations ("Condensed to atoms") and their σ/π/δ/φ resolution (condensed
# to atoms and to shells).
#
# Legend: AE = all-electron; X2C = scalar-relativistic (atomic = atom-by-atom
# decoupling, same populations as full X2C); ECP = def2-ECP, 28-electron core
# on Rh, which carries the relativistic effect itself (no X2C).
# ANO-RCC-MB-OCC = ANO-RCC-MB without the empty outermost p on atoms beyond
# Ar (Rh 5s3p2d), the same valence shell structure as MINAO.
# Wall times are from the cluster/local computer, one node.
#
#  case  Rh basis       ligand basis   Hamiltonian   minimal basis Rh / C,O        time      purpose
#   1    def2-TZVPP     def2-TZVPP     ECP           MINAO / STO-3G                 6-8 min   ECP route, STO-3G ligand convention
#   2    def2-TZVPP     def2-TZVPP     ECP           MINAO / MINAO                  6-8 min   ECP route, free-atom ligand convention
#   3    def2-TZVPP     def2-TZVPP     ECP           ANO-RCC-MB / ANO-RCC-MB        6-8 min   INVALID: all-electron minimal basis on an ECP atom;
#                                                                                              1.3 e spills into Rh 3s3p3d. Kept as the negative control.
#   4    ANO-RCC-VTZP   def2-TZVPP     full X2C      ANO-RCC-MB / ANO-RCC-MB        2 h       AE reference, Roos's full minimal basis (Rh 5p included)
#   5    ANO-RCC-VTZP   def2-TZVPP     atomic X2C    ANO-RCC-MB / ANO-RCC-MB        20-30 min same as 4 with atomic X2C: tests the decoupling (agrees to 2e-6)
#   6    ANO-RCC-VTZP   def2-TZVPP     atomic X2C    ANO-RCC-MB-OCC / ANO-RCC-MB    20-30 min 5 without the empty Rh 5p: the 5p convention (0.6 e on Rh)
#   7    ANO-RCC-VTZP   def2-TZVPP     atomic X2C    ANO-RCC-MB-OCC / STO-3G        20-30 min 6 with STO-3G ligands: ligand convention; AE partner of case 1
#   8    ANO-RCC-VTZP   def2-TZVPP     atomic X2C    ANO-RCC-MB-OCC / MINAO         20-30 min 6 with MINAO ligands: MINAO vs ANO on ligands; AE partner of case 2
#   9    x2c-TZVPPall   def2-TZVPP     atomic X2C    ANO-RCC-MB / ANO-RCC-MB        8-15 min  5 with a segmented Rh basis: metal computational basis
#  10    ANO-RCC-VTZP   6-31G*         atomic X2C    ANO-RCC-MB / ANO-RCC-MB        10-15 min 5 with 6-31G* ligands: ligand computational basis
#  11    x2c-TZVPPall   6-31G*         atomic X2C    ANO-RCC-MB / ANO-RCC-MB        8-10 min  cheapest AE combination; 4, 9, 10, 11 form the 2x2 basis grid
#
# Pairs that isolate one variable:
#   4 vs 5           full vs atomic X2C             (no difference)
#   5 vs 6           empty Rh 5p in or out          (largest effect: 0.6 e on Rh, sign of Rh-C σ)
#   6 vs 7/8         ligand minimal basis           (STO-3G vs free-atom functions: 0.15 e on O)
#   6 vs 8           MINAO vs ANO-RCC-MB on ligands (interchangeable, 0.01 e)
#   7 vs 1, 8 vs 2   AE vs ECP on Rh, matched shells (0.25 e on Rh, 0.05 on σ)
#   4/5 vs 9, 10, 11 computational basis           (converged: < 0.02 e everywhere)
#
# Recommended production settings:
# x2c-tzvppall + 6-31G*, -x2c-atomic, -minimal-basis sto-3g-ano-occ (for small organometallic/inorganic clusters with one or more metals)
# def2-TZVPP (with SD ECP), -minimal-basis sto-3g+minao (for larger organometallic/inorganic clusters with one or more metals)

set -u
export PYTHONIOENCODING=utf-8
XYZ=xyz/rhco3.xyz   # change if the geometry lives elsewhere
COMMON="-xyz $XYZ -mult 2 -xc pbe -stable -symmetry-shells"

# --- ECP family (def2-ECP on Rh, no X2C) -------------------------------------
minpop-uks $COMMON -basis def2-tzvpp                                           -minimal-basis sto-3g+minao          > rhco3_case1.out  2>&1
minpop-uks $COMMON -basis def2-tzvpp                                           -minimal-basis minao                 > rhco3_case2.out  2>&1
minpop-uks $COMMON -basis def2-tzvpp                                           -minimal-basis ano-rcc-mb            > rhco3_case3.out  2>&1   # negative control

# --- All-electron family, ANO-RCC-VTZP on Rh ---------------------------------
minpop-uks $COMMON -basis ano-rcc-vtzp -basis-light def2-tzvpp  -x2c           -minimal-basis ano-rcc-mb            > rhco3_case4.out  2>&1
minpop-uks $COMMON -basis ano-rcc-vtzp -basis-light def2-tzvpp  -x2c-atomic    -minimal-basis ano-rcc-mb            > rhco3_case5.out  2>&1
minpop-uks $COMMON -basis ano-rcc-vtzp -basis-light def2-tzvpp  -x2c-atomic    -minimal-basis ano-rcc-mb-occ        > rhco3_case6.out  2>&1
minpop-uks $COMMON -basis ano-rcc-vtzp -basis-light def2-tzvpp  -x2c-atomic    -minimal-basis sto-3g-ano-occ        > rhco3_case7.out  2>&1
minpop-uks $COMMON -basis ano-rcc-vtzp -basis-light def2-tzvpp  -x2c-atomic    -minimal-basis minao+ano-rcc-mb-occ  > rhco3_case8.out  2>&1

# --- Computational-basis grid (fixed minimal basis ano-rcc-mb) ---------------
minpop-uks $COMMON -basis x2c-tzvppall -basis-light def2-tzvpp  -x2c-atomic    -minimal-basis ano-rcc-mb            > rhco3_case9.out  2>&1
minpop-uks $COMMON -basis ano-rcc-vtzp -basis-light "6-31G*"    -x2c-atomic    -minimal-basis ano-rcc-mb            > rhco3_case10.out 2>&1
minpop-uks $COMMON -basis x2c-tzvppall -basis-light "6-31G*"    -x2c-atomic    -minimal-basis ano-rcc-mb            > rhco3_case11.out 2>&1

# --- Summary -------------------------------------------------------------------
for n in 1 2 3 4 5 6 7 8 9 10 11; do
  f=rhco3_case$n.out
  [ -f "$f" ] || continue
  echo "== case $n"
  grep -m1 "Basis:" $f; grep -m1 "Hamiltonian" $f; grep -m1 "Minimal-basis projection" $f
  grep -m1 "WARNING" $f
  grep "converged SCF energy" $f | tail -1
  grep "Projection completeness" $f | cut -c1-110
  grep -A4 "MBS Mulliken charges and spin" $f | tail -4
  for c in σ π δ; do grep -A4 "MBS $c Condensed to atoms (all electrons)" $f | grep "^ *[234]  C " | head -1 | sed "s/^/   $c Rh-C row: /"; done
done
