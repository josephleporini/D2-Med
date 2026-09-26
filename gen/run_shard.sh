#!/bin/bash
# render worker k of N for one scene folder. usage: run_shard.sh <mode full|train> <dir> <k> <N>
cd "$(dirname "$0")"; mode=$1; d=$2; k=$3; N=$4; L=../out/pod_render_${k}.log; i=0
for p in $(ls ../out/$d/C*_params.json | sort); do
  if [ $((i % N)) -eq $k ]; then sid=$(basename $p _params.json)
    if [ "$mode" = full ]; then [ -f ../out/$d/${sid}_sidecar.json ] || timeout 900 python3 scene3.py full $p ../out/$d 2>&1 | grep -E "^DONE|Traceback|Error" >> $L
    else [ -f ../out/$d/${sid}_views.json ] || timeout 1500 python3 scene3.py train $p ../out/$d 3 2>&1 | grep -E "VIEWSDONE|Traceback|Error" >> $L; fi
  fi; i=$((i+1)); done
