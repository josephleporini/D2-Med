#!/bin/bash
# v2 feature cache: test sets (det) first, then training scenes (gt) in a loop while renders arrive
cd /home/claude/probeB/gen; source ../venv_pose/bin/activate
until grep -q LABEL2DONE ../out/label2.log 2>/dev/null; do python seg_features2.py ../out/train gt 20 2>&1 | grep FEAT2 >> ../out/train/feat2.log; sleep 20; done
python seg_features2.py ../out/check20 det 2>&1 | grep FEAT2 >> ../out/train/feat2.log
python seg_features2.py ../out/batch100 det 2>&1 | grep FEAT2 >> ../out/train/feat2.log
while true; do
  python seg_features2.py ../out/train gt 2>&1 | grep FEAT2 >> ../out/train/feat2.log
  grep -q ALLDONE ../out/train/run2.log 2>/dev/null && break
  sleep 300
done
