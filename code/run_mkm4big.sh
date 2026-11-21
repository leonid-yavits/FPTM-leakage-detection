#!/usr/bin/env bash
# FPTM_ROOT: folder holding code/, results/, data/, dl/, work/ and src/Tsetlin.jl (default: repository root)
ROOT=${FPTM_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}
JULIA=${JULIA:-julia}
TSETLIN_JL=${TSETLIN_JL:-$ROOT/src/Tsetlin.jl}
cd "$ROOT"
while [ ! -f results/mkm4.done ]; do sleep 20; done
python3 -I code/mkm4_big.py build > results/mkm4big_build.log 2>&1
python3 -I code/mkm4_big.py sweep 24000,40000 > results/mkm4big_sweep.log 2>&1
$JULIA -O3 -t 2 $ROOT/code/fptm_sweep2.jl $TSETLIN_JL $ROOT/work/sweep_mkm4big 200 100 100 16 8 10 $ROOT/results/mkm4big_fptm.jsonl > results/mkm4big_fptm.log 2>&1
echo DONE > results/mkm4big.done
