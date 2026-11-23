#!/usr/bin/env bash
# Review round 4, queue B: lag-restricted bivariate tests, mkm4 cross-window test, desync and PCA baselines.
ROOT=${FPTM_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}
cd "$ROOT"; export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
heavy() { flock /tmp/heavy.lock "$@"; }
python3 -I code/biv_delta.py ascad 500,707,1000,1414,2000,2828,4000,5657,8000 0:10 > results/r4_biv_delta_ascad.log 2>&1
for s in 0 10 25 50; do python3 -I code/desync.py base ascad $s > results/r4_desync_base_ascad_s$s.log 2>&1; done
python3 -I code/pca_detect.py raw ascad > results/r4_lda_raw_ascad.log 2>&1
for nc in 20 100; do python3 -I code/pca_detect.py base ascad $nc > results/r4_pca_base_ascad_$nc.log 2>&1; done
heavy python3 -I code/mkm4_xwin.py mkm4 > results/r4_mkm4_xwin.log 2>&1
heavy python3 -I code/mkm4_xwin.py mkm4big >> results/r4_mkm4_xwin.log 2>&1
for s in 5 10 20; do heavy python3 -I code/desync.py base pqm4 $s > results/r4_desync_base_pqm4_s$s.log 2>&1; done
for nc in 20 100; do heavy python3 -I code/pca_detect.py base pqm4 $nc > results/r4_pca_base_pqm4_$nc.log 2>&1; done
python3 -I code/biv_delta.py ascad_noov 4000,5657,8000,11314,16000,22627,32000 0:10 > results/r4_biv_delta_noov.log 2>&1
echo DONE > results/r4_b.done
