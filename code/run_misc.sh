#!/usr/bin/env bash
ROOT=${FPTM_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}
cd "$ROOT"; J=$HOME/.juliaup/bin/julia
while [ ! -f results/locb.done ]; do sleep 30; done
for s in 2 3 4 5; do $J -O3 -t 1 code/dump_model.jl $ROOT/src/Tsetlin.jl $ROOT/work/ref_K3 15 $s $ROOT/work/models/ref_s$s > results/dump_ref_s$s.log 2>&1; done
python3 -I code/null_classic.py > results/null_classic.log 2>&1
echo DONE > results/misc.done
