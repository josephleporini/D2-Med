#!/bin/bash
cd /home/claude/probeB/gen; source ../venv_pose/bin/activate
python seg_features.py ../out/check20 det >> ../out/train/feat.log 2>&1
python seg_features.py ../out/batch100 det >> ../out/train/feat.log 2>&1
while true; do
  python seg_features.py ../out/train gt 2>&1 | grep FEAT >> ../out/train/feat.log
  grep -q ALLDONE ../out/train/run.log && { python seg_features.py ../out/train gt 2>&1 | grep FEAT >> ../out/train/feat.log; echo FEATDONE >> ../out/train/feat.log; break; }
  sleep 900
done
