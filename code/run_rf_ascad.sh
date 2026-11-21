#!/usr/bin/env bash
ROOT=${FPTM_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}
cd "$ROOT"
while pgrep -f "rf_sweep.py pqm4" >/dev/null; do sleep 10; done
python3 -I code/rf_sweep.py ascad 8000,11314,16000,22627,32000,40000 0:5 --mlp-es 50 > results/ascad_rep_rf_a.log 2>&1 &
python3 -I code/rf_sweep.py ascad 8000,11314,16000,22627,32000,40000 5:10 --mlp-es 50 > results/ascad_rep_rf_b.log 2>&1 &
wait
echo DONE > results/rf.done
