#!/bin/bash
cd /home/claude/probeB/gen
for p in ../out/check20/C*_params.json; do
  timeout 900 python3 scene.py "$p" ../out/check20 2>&1 | grep -E "DONE|Error|Traceback" >> ../out/check20/run.log
done
echo ALLDONE >> ../out/check20/run.log
