#!/bin/bash
# multi-view training renders (scene2.py train), v2 part labels; skips scenes already rendered by either mode
cd /home/claude/probeB/gen
while pgrep -f "scene.py ../out/train" >/dev/null; do sleep 5; done
for p in $(ls ../out/train/C*_params.json | sort); do
  sid=$(basename $p _params.json)
  [ -f ../out/train/${sid}_sidecar.json ] && continue
  [ -f ../out/train/${sid}_views.json ] && continue
  timeout 1200 python3 scene2.py train "$p" ../out/train 3 2>&1 | grep -E "VIEWSDONE|Traceback|Error:" | grep -v EGL >> ../out/train/run2.log
done
echo ALLDONE >> ../out/train/run2.log
