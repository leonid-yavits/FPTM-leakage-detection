#!/usr/bin/env bash
ROOT=${FPTM_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}
cd "$ROOT"
python3 -I code/rf_sweep.py ref 500,707,1000,1414,2000,2828,4000 0:10 > results/ref_rep_rf.log 2>&1
python3 -I code/rf_sweep.py pqm4 1000,1414,2000,2828,4000,5657,8000 0:10 > results/pqm4_rep_rf.log 2>&1
python3 -I code/rf_sweep.py ascad 8000,11314,16000,22627,32000,40000 0:10 --mlp-es 50 > results/ascad_rep_rf.log 2>&1
echo DONE > results/rf.done
