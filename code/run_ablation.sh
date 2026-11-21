#!/usr/bin/env bash
ROOT=${FPTM_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}
cd "$ROOT"; J=$HOME/.juliaup/bin/julia
for v in no_overlap mask_no_overlap mv shares_no_overlap; do
  python3 -I code/ascad_ablation.py prep $v > results/ascad_abl_$v.log 2>&1
  for s in 1 2 3; do $J -O3 -t 2 code/fptm_detect.jl $ROOT/src/Tsetlin.jl $ROOT/work/ascad_abl_$v y 200 100 100 16 8 15 $s s$s 2>&1 | grep -E "RESULT|literals"; done > work/ascad_abl_$v/runs.log
  python3 -I code/ascad_ablation.py eval $v >> results/ascad_abl_$v.log 2>&1
done
echo DONE > results/ablation.done
