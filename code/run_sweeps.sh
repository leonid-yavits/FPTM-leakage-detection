#!/usr/bin/env bash
# FPTM_ROOT: folder holding code/, results/, data/, dl/, work/ and src/Tsetlin.jl (default: repository root)
ROOT=${FPTM_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}
JULIA=${JULIA:-julia}
TSETLIN_JL=${TSETLIN_JL:-$ROOT/src/Tsetlin.jl}
cd "$ROOT"
python3 -I code/sweep.py ascad 1000,2000,4000,8000,16000,32000,40000 3 welch,chi2,mvdcov,welch2,biv,mlp,fptm 50 > results/ascad_sweep.log 2>&1
python3 -I code/sweep.py pqm4 250,500,1000,2000,4000,8000,12000 3 welch,chi2,mvdcov,mlp,fptm 30 > results/pqm4_sweep.log 2>&1
python3 -I code/sweep.py ref 250,500,1000,2000,4000,8000,12000 3 welch,chi2,mvdcov,mlp,fptm 30 > results/ref_sweep.log 2>&1
python3 -I code/sweep.py mkm4 12000 3 welch,chi2,mvdcov,welch2,mlp,fptm 50 > results/mkm4_sweep.log 2>&1
echo ALLDONE > results/sweeps.done
