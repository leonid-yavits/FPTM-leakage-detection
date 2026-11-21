#!/usr/bin/env bash
# FPTM_ROOT: folder holding code/, results/, data/, dl/, work/ and src/Tsetlin.jl (default: repository root)
ROOT=${FPTM_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}
JULIA=${JULIA:-julia}
TSETLIN_JL=${TSETLIN_JL:-$ROOT/src/Tsetlin.jl}
J=$JULIA; S=$ROOT/code/fptm_sweep2.jl; T=$TSETLIN_JL
$J -O3 -t 2 $S $T $ROOT/work/sweep_ascad 200 100 100 16 8 10 $ROOT/results/ascad_fptm.jsonl
$J -O3 -t 2 $S $T $ROOT/work/sweep_pqm4 200 100 100 16 8 10 $ROOT/results/pqm4_fptm.jsonl
echo FPTMSWEEPDONE
