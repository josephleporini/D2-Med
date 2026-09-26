#!/bin/bash
# stop training renders after body C3150, then cache limb-end windows for train2 (serial = faster on 2 cores)
cd /home/claude/probeB/gen
until [ -f ../out/train2/C3150_views.json ]; do sleep 30; done
kill $(pgrep -f "bash ./run_next.sh") 2>/dev/null
sleep 5; kill $(pgrep -f "scene2.py train") 2>/dev/null
source ../venv_pose/bin/activate
python lend_cache.py ../out/train2 2>&1 | grep LEND >> ../out/run_next.log
python lend_relabel.py ../out/train2 2>&1 | tail -1 >> ../out/run_next.log
echo CKPT150READY >> ../out/run_next.log
