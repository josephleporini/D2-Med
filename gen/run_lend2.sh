#!/bin/bash
cd /home/claude/probeB/gen; source ../venv_pose/bin/activate
until grep -q DEVDONE ../out/run_next.log 2>/dev/null; do sleep 120; done
while true; do python lend_cache.py ../out/train2 2>&1 | grep LEND >> ../out/run_next.log
  grep -q TRAIN2DONE ../out/run_next.log && { python lend_cache.py ../out/train2 2>&1 | grep LEND >> ../out/run_next.log; echo LEND2DONE >> ../out/run_next.log; break; }
  sleep 600; done
