#!/bin/bash
# Test F increment: (A) test set re-rendered at 16 samples, (B) dev set with full labels, (C) 300 new training bodies
cd /home/claude/probeB/gen; O=../out; L=$O/run_next.log
PY=../venv_pose/bin/python
for d in check20_s16 batch100_s16; do
  for p in $(ls $O/$d/C*_params.json | sort); do sid=$(basename $p _params.json)
    [ -f $O/$d/$sid.jpg ] && continue
    RGB_ONLY=1 timeout 900 python3 scene.py $p $O/$d 2>&1 | grep -E "RGBDONE|Traceback" >> $L; done
  $PY rtmw_cache.py $O/$d >> $L 2>&1; $PY seg_features2.py $O/$d det 2>&1 | grep FEAT2 >> $L
done
echo TESTS16DONE >> $L
for p in $(ls $O/dev/C*_params.json | sort); do sid=$(basename $p _params.json)
  [ -f $O/dev/${sid}_sidecar.json ] && continue
  timeout 900 python3 scene.py $p $O/dev 2>&1 | grep -E "^DONE|Traceback" >> $L
  timeout 300 python3 scene2.py label $p $O/dev 2>&1 | grep -E "PART2DONE|Traceback" >> $L; done
python3 gt_frame.py $O/dev > /dev/null 2>&1; $PY rtmw_cache.py $O/dev >> $L 2>&1; $PY seg_features2.py $O/dev det 2>&1 | grep FEAT2 >> $L
echo DEVDONE >> $L
for p in $(ls $O/train2/C*_params.json | sort); do sid=$(basename $p _params.json)
  [ -f $O/train2/${sid}_views.json ] && continue
  timeout 1500 python3 scene2.py train $p $O/train2 3 2>&1 | grep -E "VIEWSDONE|Traceback" >> $L; done
echo TRAIN2DONE >> $L
