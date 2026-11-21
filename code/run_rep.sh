#!/usr/bin/env bash
# 10 repeats per trace count on a finer grid around the detection thresholds (round 1 review, F12).
ROOT=${FPTM_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}
cd "$ROOT"
J=$HOME/.juliaup/bin/julia
python3 -I code/sweep_rep.py ref 500,707,1000,1414,2000,2828,4000 0:10 welch,chi2,mvdcov,mlp,fptm 30 > results/ref_rep.log 2>&1
$J -O3 -t 2 code/fptm_sweep2.jl $ROOT/src/Tsetlin.jl $ROOT/work/sweep_ref_rep 200 100 100 16 8 10 $ROOT/results/ref_rep_fptm.jsonl > results/ref_rep_fptm.log 2>&1
python3 -I code/sweep_rep.py pqm4 1000,1414,2000,2828,4000,5657,8000 0:10 welch,chi2,mvdcov,mlp,fptm 30 > results/pqm4_rep.log 2>&1
$J -O3 -t 2 code/fptm_sweep2.jl $ROOT/src/Tsetlin.jl $ROOT/work/sweep_pqm4_rep 200 100 100 16 8 10 $ROOT/results/pqm4_rep_fptm.jsonl > results/pqm4_rep_fptm.log 2>&1
python3 -I code/sweep_rep.py ascad 8000,11314,16000,22627,32000,40000 0:10 chi2,welch2,mvdcov,mlp,fptm 50 > results/ascad_rep.log 2>&1
$J -O3 -t 2 code/fptm_sweep2.jl $ROOT/src/Tsetlin.jl $ROOT/work/sweep_ascad_rep 200 100 100 16 8 10 $ROOT/results/ascad_rep_fptm.jsonl > results/ascad_rep_fptm.log 2>&1
echo DONE > results/rep.done
