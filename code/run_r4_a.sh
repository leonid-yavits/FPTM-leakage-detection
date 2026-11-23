#!/usr/bin/env bash
# Review round 4, queue A: held-aside pqm4, desync FPTM runs, PCA FPTM runs, permuted-label controls.
ROOT=${FPTM_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}
cd "$ROOT"; J=$HOME/.juliaup/bin/julia; S=$ROOT/src/Tsetlin.jl
fp() { $J -O3 -t 1 code/fptm_detect.jl $S $ROOT/work/$1 $2 200 100 100 16 8 15 $3 $4 2>&1 | grep -E "RESULT|literals"; }
heavy() { flock /tmp/heavy.lock "$@"; }
heavy python3 -I code/build_heldout.py > results/r4_pqm4h.log 2>&1
heavy python3 -I code/prep_fptm.py pqm4h 3 >> results/r4_pqm4h.log 2>&1
for s in 1 2 3; do fp pqm4h_K3 y $s s$s; done >> results/r4_pqm4h.log
python3 -I code/loc_eval.py pqm4h $ROOT/work/pqm4h_K3 3 1,2,3 >> results/r4_pqm4h.log 2>&1
for s in 0 10 25 50; do python3 -I code/desync.py prep ascad $s > results/r4_desync_fptm_ascad_s$s.log 2>&1
  for k in 1 2 3; do fp ascad_ds$s y $k s$k; done >> results/r4_desync_fptm_ascad_s$s.log; done
for s in 5 10 20; do heavy python3 -I code/desync.py prep pqm4 $s > results/r4_desync_fptm_pqm4_s$s.log 2>&1
  for k in 1 2 3; do fp pqm4_ds$s y $k s$k; done >> results/r4_desync_fptm_pqm4_s$s.log; done
for ds in ascad pqm4; do for nc in 20 100; do heavy python3 -I code/pca_detect.py prep $ds $nc > results/r4_pca_fptm_${ds}_$nc.log 2>&1
  for k in 1 2 3; do fp ${ds}_pca$nc y $k s$k; done >> results/r4_pca_fptm_${ds}_$nc.log; done; done
python3 -I code/null_perm.py ascad 20; python3 -I code/null_perm.py ref 20; python3 -I code/null_perm.py pqm4 10
for ds in ascad ref pqm4; do n=20; [ $ds = pqm4 ] && n=10
  for r in $(seq 1 $n); do echo "perm $r"; fp ${ds}_K3 ynull$r $r n$r; done > results/r4_null_$ds.log; done
echo DONE > results/r4_a.done
