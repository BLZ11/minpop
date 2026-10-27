#!/bin/bash
# Pt2, PtO, PtH at experimental bond lengths: UKS/PBE, ANO-RCC-VDZP, X2C,
# ANO-RCC-MB projection, sigma/pi/delta/phi matrices.
set -u
export PYTHONIOENCODING=utf-8
OPTS="-xc pbe -basis ano-rcc-vdzp -minimal-basis ano-rcc-mb -x2c -symmetry-shells"
minpop-uks -xyz xyz/pt2.xyz -mult 3 $OPTS > pt2_uks_pbe_t.out 2>&1
minpop-uks -xyz xyz/pto.xyz -mult 3 $OPTS > pto_uks_pbe_t.out 2>&1
minpop-uks -xyz xyz/pth.xyz -mult 2 $OPTS > pth_uks_pbe_d.out 2>&1
grep -H "converged SCF energy" pt2_uks_pbe_t.out pto_uks_pbe_t.out pth_uks_pbe_d.out
