#!/usr/bin/env bash
# FPTM_ROOT: folder holding code/, results/, data/, dl/, work/ and src/Tsetlin.jl (default: repository root)
ROOT=${FPTM_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}
JULIA=${JULIA:-julia}
TSETLIN_JL=${TSETLIN_JL:-$ROOT/src/Tsetlin.jl}
cd "$ROOT"
J=$JULIA
python3 -I code/loc_vs_n.py > results/locn.log 2>&1
for N in 8000 16000 32000; do for r in 0 1 2; do
  $J -O3 -t 2 $ROOT/code/fptm_loc_sweepfile.jl $ROOT/work/locn_ascad $N $r 10 $((r+1)) $ROOT/work/locn_ascad/S_${N}_${r}.bin >> results/locn.log 2>&1
done; done
for K in 1 5; do
  python3 -I code/prep_fptm.py ascad $K >> results/ablation.log 2>&1
  for sd in 1 2 3; do $J -O3 -t 2 $ROOT/code/fptm_detect.jl $TSETLIN_JL $ROOT/work/ascad_K$K y 200 100 100 16 8 15 $sd s$sd 2>&1 | grep RESULT | sed "s/^/ascad K$K /" >> results/ablation.log; done
  python3 -I code/prep_fptm.py pqm4 $K >> results/ablation.log 2>&1
  for sd in 1 2 3; do $J -O3 -t 2 $ROOT/code/fptm_detect.jl $TSETLIN_JL $ROOT/work/pqm4_K$K y 200 100 100 16 8 15 $sd s$sd 2>&1 | grep RESULT | sed "s/^/pqm4 K$K /" >> results/ablation.log; done
done
echo R4DONE > results/round4.done
