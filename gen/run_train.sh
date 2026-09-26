#!/bin/bash
cd /home/claude/probeB/gen
for p in $(ls ../out/train/C*_params.json | sort); do
  sid=$(basename $p _params.json)
  [ -f ../out/train/${sid}_sidecar.json ] && continue
  timeout 900 python3 scene.py "$p" ../out/train 2>&1 | grep -E "DONE|Traceback|Error:" | grep -v EGL >> ../out/train/run.log
done
echo ALLDONE >> ../out/train/run.log
