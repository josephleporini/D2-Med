#!/usr/bin/env bash
# destination: pull every top-level dir from the relay, verify counts, bytes and split manifests
U=$1; cd /workspace
until curl -sf $U/ready >/dev/null; do echo "waiting relay $(date -u +%H:%M:%S)"; sleep 30; done
for d in $(curl -sf $U/list | python3 -c 'import sys,json;print(" ".join(json.load(sys.stdin)))'); do
  for try in 1 2 3; do
    t0=$(date +%s); curl -sf "$U/tar?d=$d" | tar xf - -C /workspace && break; echo "RETRY $d $try"; done
  src=$(curl -sf "$U/stat?d=$d"); dst=$(python3 -c "import json,subprocess;print(json.dumps(subprocess.run(\"find '/workspace/$d' -type f | wc -l; du -sb '/workspace/$d' | cut -f1\",shell=True,capture_output=True,text=True).stdout.split()))")
  echo "COPY $d src=$src dst=$dst s=$(( $(date +%s)-t0 ))"
done
for sp in dev3 test4; do python3 /workspace/probeB/repo/score/manifest.py verify /workspace/probeB/out/$sp /workspace/probeB/manifests/$sp.sha256; done
echo MOVE_DONE; df -h /workspace | tail -1
