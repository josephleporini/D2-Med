#!/bin/bash
cd /home/claude/probeB/gen
for p in ../out/batch100/C*_params.json; do
  timeout 900 nice -n 5 python3 scene.py "$p" ../out/batch100 2>&1 | grep -E "DONE|Traceback|Error:" | grep -v EGL >> ../out/batch100/run.log
done
echo ALLDONE >> ../out/batch100/run.log
