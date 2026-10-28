#!/bin/bash
# Rh(CO)3 triplet, UKS/PBE: five computational-basis / minimal-basis combinations.
# Populations, total bond populations ("Condensed to atoms") and their σ/π/δ/φ
# resolution (condensed to atoms and to shells) for each.
#
#  case  computational basis                     Hamiltonian  minimal basis
#   1    ANO-RCC-VTZP on all atoms               X2C          ANO-RCC-MB on all atoms
#   2    ANO-RCC-VTZP on all atoms               X2C          ANO-RCC-MB on Rh, STO-3G on C,O
#   3    ANO-RCC-VTZP on Rh, def2-TZVPP on C,O   X2C          ANO-RCC-MB on Rh, STO-3G on C,O
#   4    ANO-RCC-VTZP on Rh, 6-31G* on C,O       X2C          ANO-RCC-MB on Rh, STO-3G on C,O
#   5    def2-TZVPP on all, def2-ECP (28 e) on Rh  non-rel.   MINAO on Rh (4s4p4d5s), STO-3G on C,O
#
# 1 vs 2: effect of the light-atom minimal basis.  2 vs 3 vs 4: effect of the
# light-atom computational basis.  5: ECP + valence minimal basis (MINAO's ten
# functions are exactly the 4s 5s 4p 4d the ECP leaves; the completeness line
# shows how well they hold the 17 valence electrons).
set -u
export PYTHONIOENCODING=utf-8
XYZ=rhco3.xyz
COMMON="-xyz $XYZ -mult 2 -xc pbe -stable -symmetry-shells"

minpop-uks $COMMON -basis def2-tzvpp                                 -minimal-basis sto-3g+minao        > rhco3_case1.out 2>&1
minpop-uks $COMMON -basis def2-tzvpp                                 -minimal-basis minao               > rhco3_case2.out 2>&1
minpop-uks $COMMON -basis def2-tzvpp                                 -minimal-basis ano-rcc-mb          > rhco3_case3.out 2>&1
minpop-uks $COMMON -basis ano-rcc-vtzp -basis-light def2-tzvpp  -x2c -minimal-basis ano-rcc-mb         > rhco3_case4.out 2>&1
minpop-uks $COMMON -basis ano-rcc-vtzp -basis-light def2-tzvpp  -x2c-atomic -minimal-basis ano-rcc-mb  > rhco3_case5.out 2>&1
minpop-uks $COMMON -basis ano-rcc-vtzp -basis-light def2-tzvpp  -x2c-atomic -minimal-basis ano-rcc-mb-occ  > rhco3_case6.out 2>&1
minpop-uks $COMMON -basis ano-rcc-vtzp -basis-light def2-tzvpp  -x2c-atomic -minimal-basis sto-3g-ano-occ  > rhco3_case7.out 2>&1
minpop-uks $COMMON -basis ano-rcc-vtzp -basis-light def2-tzvpp  -x2c-atomic -minimal-basis minao+ano-rcc-mb-occ  > rhco3_case8.out 2>&1
minpop-uks $COMMON -basis x2c-tzvppall -basis-light def2-tzvpp  -x2c-atomic -minimal-basis ano-rcc-mb  > rhco3_case9.out 2>&1
minpop-uks $COMMON -basis ano-rcc-vtzp -basis-light "6-31G*"  -x2c-atomic -minimal-basis ano-rcc-mb  > rhco3_case10.out 2>&1
minpop-uks $COMMON -basis x2c-tzvppall -basis-light "6-31G*"  -x2c-atomic -minimal-basis ano-rcc-mb  > rhco3_case11.out 2>&1

for n in 1 2 3 4 5 6 7 8 9 10 11; do
  f=rhco3_case$n.out
  echo "== case $n"
  grep -m1 "Basis:" $f; grep -m1 "Minimal-basis projection" $f
  grep "converged SCF energy" $f | tail -1
  grep "Projection completeness" $f | cut -c1-110
  grep -A4 "MBS Mulliken charges and spin" $f | tail -4
  for c in σ π δ; do grep -A4 "MBS $c Condensed to atoms (all electrons)" $f | grep "^ *[234]  C " | head -1 | sed "s/^/   $c Rh-C row: /"; done
done
