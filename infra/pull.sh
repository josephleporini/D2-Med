#!/usr/bin/env bash
# destination: pull every top-level entry from the relay (stall-safe, logged), then verify
U=$1; cd /workspace
until curl -sf --max-time 20 $U/ready >/dev/null; do echo "waiting relay $(date -u +%H:%M:%S)"; sleep 30; done
L=$(curl -sf --max-time 60 $U/list); echo "LIST $L"
for d in $(echo "$L" | python3 -c "import sys,json;l=json.load(sys.stdin);l=[x for x in l if x!=\".cache\"]+([\".cache\"] if \".cache\" in l else []);print(\" \".join(l))"); do
  echo "START $d $(date -u +%H:%M:%S)"; t0=$(date +%s)
  for try in 1 2 3; do
    curl -sf --speed-time 120 --speed-limit 10000 "$U/tar?d=$d" | tar xf - -C /workspace --totals 2>&1 | tail -1 && [ ${PIPESTATUS[0]} -eq 0 ] && break
    echo "RETRY $d $try"; done
  src=$(curl -sf --max-time 600 "$U/stat?d=$d"); dst=$(printf '["%s", "%s"]' "$(find "/workspace/$d" -type f | wc -l)" "$(du -sb "/workspace/$d" | cut -f1)")
  echo "COPY $d src=$src dst=$dst s=$(( $(date +%s)-t0 ))"
done
for sp in dev3 test4; do python3 /workspace/probeB/repo/score/manifest.py verify /workspace/probeB/out/$sp /workspace/probeB/manifests/$sp.sha256; done
echo MOVE_DONE; df -h /workspace | tail -1
