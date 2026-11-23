#!/usr/bin/env bash
# Review round 4, runs started outside run_r4_a2.sh and run_r4_b.sh:
# small-lag pair test at large N, the no-overlap lag test, ASCAD desync baselines, permuted-label controls for ref and pqm4,
# single-thread reproducibility check. Run after run_r4_b.sh (the ASCAD desync baselines need the patched desync.py).
ROOT=${FPTM_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}
cd "$ROOT"; J=${JULIA:-julia}; S=${TSETLIN_JL:-$ROOT/src/Tsetlin.jl}
fp() { $J -O3 -t 1 code/fptm_detect.jl $S $ROOT/work/$1 $2 200 100 100 16 8 15 $3 $4 2>&1 | grep -E "RESULT|literals"; }
BIV_DELTAS=1,2,5,10,20,50,100,200,300 BIV_TAG=_small python3 -I code/biv_delta.py ascad 8000,11314,16000,22627 0:10 > results/r4_biv_delta_small.log 2>&1
BIV_TAG=_d python3 -I code/biv_delta.py ascad_noov 2828,4000,5657,8000,11314,16000,22627 0:10 > results/r4_biv_delta_noov_d.log 2>&1
for s in 0 10 25 50; do python3 -I code/desync.py base ascad $s > results/r4_desync_base_ascad_s$s.log 2>&1; done
python3 -I code/null_perm.py ref 5; python3 -I code/null_perm.py pqm4 3
for r in 1 2 3 4 5; do echo "perm $r"; fp ref_K3 ynull$r $r n$r; done > results/r4_nullB_ref.log
for r in 1 2 3; do echo "perm $r"; fp pqm4_K3 ynull$r $r n$r; done > results/r4_nullB_pqm4.log
for t in repA repB; do fp ascad_K3 y 1 $t; done > results/r4_repro.log
cmp work/ascad_K3/S_repA.bin work/ascad_K3/S_repB.bin && echo IDENTICAL > results/r4_repro_check.txt
python3 -I code/summ_r4.py; python3 -I code/table_r4.py; python3 -I code/table_lag.py
