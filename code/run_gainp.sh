#!/usr/bin/env bash
# Traces-to-detection sweeps on the simulated GainP trace sets, then FPTM on the same subsets.
# The grid stops at 16,000 traces for the two lowest noise levels, where every method detects far earlier.
ROOT=${FPTM_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}
JULIA=${JULIA:-julia}
TSETLIN_JL=${TSETLIN_JL:-$ROOT/src/Tsetlin.jl}
cd "$ROOT"
declare -A GRID=([250]=250,500,1000,2000,4000,8000,16000 [500]=250,500,1000,2000,4000,8000,16000
                 [1000]=250,500,1000,2000,4000,8000,16000,32000 [2000]=250,500,1000,2000,4000,8000,16000,32000,48000)
for s in 250 500 1000 2000; do
  if [ ! -f work/sweep_gainp_s$s/meta.json ]; then
    python3 -I code/sweep.py gainp_s$s ${GRID[$s]} 3 welch,chi2,mvdcov,mlp,fptm 30 > results/gainp_s${s}_sweep.log 2>&1
  fi
  $JULIA -O3 -t 2 code/fptm_sweep2.jl $TSETLIN_JL $ROOT/work/sweep_gainp_s$s 200 100 100 16 8 10 $ROOT/results/gainp_s${s}_fptm.jsonl > results/gainp_s${s}_fptm.log 2>&1
done
echo DONE > results/gainp_sweeps.done
