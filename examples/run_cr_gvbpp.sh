#!/bin/bash
# CrH, CrO, Cr2 at experimental bond lengths: GVB-PP (keithgroup/gvbpp) followed
# by MinPop. ANO-RCC-VDZP, X2C, ANO-RCC-MB projection.
# Pair counts: CrH 1 (4s-1s sigma, five 3d open), CrO 4 (sigma, pi, two O lone
# pairs, four open), Cr2 6 (the sextuple bond). The pair search is restricted to
# the valence orbitals automatically; semicore 3s/3p never enter it.
# Cr2 has several perfect-pairing minima: run two seeds and keep the lowest.
set -u
export PYTHONIOENCODING=utf-8
OPTS="-basis ano-rcc-vdzp -minimal-basis ano-rcc-mb -x2c -symmetry-shells -soscf"
minpop-gvbpp -xyz xyz/crh.xyz -mult 6 -npairs 1 $OPTS                         > crh_gvb.out 2>&1
minpop-gvbpp -xyz xyz/cro.xyz -mult 5 -npairs 4 $OPTS                         > cro_gvb.out 2>&1
minpop-gvbpp -xyz xyz/cr2.xyz -mult 1 -npairs 6 $OPTS -gvb-trials 8 -gvb-seed 0 > cr2_gvb_s0.out 2>&1
minpop-gvbpp -xyz xyz/cr2.xyz -mult 1 -npairs 6 $OPTS -gvb-trials 8 -gvb-seed 1 > cr2_gvb_s1.out 2>&1
grep -H "converged GVB-PP energy\|WARNING" crh_gvb.out cro_gvb.out cr2_gvb_s?.out
