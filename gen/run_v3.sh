#!/bin/bash
# v3 renders: test3 and dev3 with full 4-class labels, then train3 (3 views per body); resumable
cd /home/claude/probeB/gen; L=../out/run_v3.log
for d in test3 dev3; do
  for p in $(ls ../out/$d/C*_params.json | sort); do sid=$(basename $p _params.json)
    [ -f ../out/$d/${sid}_sidecar.json ] && continue
    timeout 900 python3 scene3.py full $p ../out/$d 2>&1 | grep -E "^DONE|Traceback" >> $L; done
  echo ${d}DONE >> $L
done
for p in $(ls ../out/train3/C*_params.json | sort); do sid=$(basename $p _params.json)
  [ -f ../out/train3/${sid}_views.json ] && continue
  timeout 1500 python3 scene3.py train $p ../out/train3 3 2>&1 | grep -E "VIEWSDONE|Traceback" >> $L; done
echo train3DONE >> $L
