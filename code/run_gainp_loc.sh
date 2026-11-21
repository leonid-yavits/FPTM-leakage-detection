#!/usr/bin/env bash
# Localization runs, DL-LA sensitivity analysis and FPTM-on-GainP inference for one simulated GainP set.
ROOT=${FPTM_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}
JULIA=${JULIA:-julia}
TSETLIN_JL=${TSETLIN_JL:-$ROOT/src/Tsetlin.jl}
ds=${1:-gainp_s1000}
cd "$ROOT"
for sd in 1 2 3; do $JULIA -O3 -t 2 code/fptm_detect.jl $TSETLIN_JL $ROOT/work/${ds}_K3 y 200 100 100 16 8 15 $sd s$sd 2>&1 | grep -E "RESULT|literals"; done > results/loc_runs_$ds.log
$JULIA -O3 -t 2 code/fptm_detect.jl $TSETLIN_JL $ROOT/work/${ds}_K3 ynull 200 100 100 16 8 15 1 null 2>&1 | grep RESULT >> results/loc_runs_$ds.log
python3 -I code/dlla_sa.py $ds 30 > results/sa_$ds.log 2>&1
python3 -I code/loc_eval_gainp.py $ds > results/loc_eval_$ds.log 2>&1
mkdir -p work/models
$JULIA -O3 -t 2 code/dump_model.jl $TSETLIN_JL $ROOT/work/${ds}_K3 15 1 $ROOT/work/models/$ds > results/dump_$ds.log 2>&1
python3 -I code/gainp_fptm.py $ROOT/work/${ds}_K3 $ROOT/work/models/$ds $ROOT/results/gainp_fptm_$ds.json >> results/dump_$ds.log 2>&1
echo DONE > results/loc_$ds.done
