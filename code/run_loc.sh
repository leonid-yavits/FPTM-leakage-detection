#!/usr/bin/env bash
# FPTM_ROOT: folder holding code/, results/, data/, dl/, work/ and src/Tsetlin.jl (default: repository root)
ROOT=${FPTM_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}
JULIA=${JULIA:-julia}
TSETLIN_JL=${TSETLIN_JL:-$ROOT/src/Tsetlin.jl}
cd $ROOT/work
for ds in pqm4 ref; do
  for sd in 1 2 3; do $JULIA -O3 -t 2 $ROOT/code/fptm_detect.jl $TSETLIN_JL $ROOT/work/${ds}_K3 y 200 100 100 16 8 15 $sd s$sd 2>&1 | grep -E "RESULT|literals" | sed "s/^/$ds /"; done
  $JULIA -O3 -t 2 $ROOT/code/fptm_detect.jl $TSETLIN_JL $ROOT/work/${ds}_K3 ynull 200 100 100 16 8 15 1 null 2>&1 | grep RESULT | sed "s/^/$ds /"
done
echo LOCDONE
