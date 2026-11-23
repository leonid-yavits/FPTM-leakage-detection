#!/usr/bin/env bash
# Review round 4, queue A (reduced): desync FPTM runs, PCA FPTM runs, permuted-label controls.
ROOT=${FPTM_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}
cd "$ROOT"; J=$HOME/.juliaup/bin/julia; S=$ROOT/src/Tsetlin.jl
fp() { $J -O3 -t 1 code/fptm_detect.jl $S $ROOT/work/$1 $2 200 100 100 16 8 15 $3 $4 2>&1 | grep -E "RESULT|literals"; }
heavy() { flock /tmp/heavy.lock "$@"; }
for s in 0 10 25 50; do python3 -I code/desync.py prep ascad $s > results/r4_desync_fptm_ascad_s$s.log 2>&1
  for k in 1 2 3; do fp ascad_ds$s y $k s$k; done >> results/r4_desync_fptm_ascad_s$s.log; done
for ds in ascad pqm4; do for nc in 20 100; do heavy python3 -I code/pca_detect.py prep $ds $nc > results/r4_pca_fptm_${ds}_$nc.log 2>&1
  for k in 1 2 3; do fp ${ds}_pca$nc y $k s$k; done >> results/r4_pca_fptm_${ds}_$nc.log; done; done
python3 -I code/null_perm.py ascad 20; python3 -I code/null_perm.py ref 10; python3 -I code/null_perm.py pqm4 5
for r in $(seq 1 20); do echo "perm $r"; fp ascad_K3 ynull$r $r n$r; done > results/r4_null_ascad.log
for s in 5 10 20; do heavy python3 -I code/desync.py prep pqm4 $s > results/r4_desync_fptm_pqm4_s$s.log 2>&1
  for k in 1 2; do fp pqm4_ds$s y $k s$k; done >> results/r4_desync_fptm_pqm4_s$s.log; done
for r in $(seq 1 10); do echo "perm $r"; fp ref_K3 ynull$r $r n$r; done > results/r4_null_ref.log
for r in $(seq 1 5); do echo "perm $r"; fp pqm4_K3 ynull$r $r n$r; done > results/r4_null_pqm4.log
echo DONE > results/r4_a.done
