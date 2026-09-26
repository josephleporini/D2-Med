#!/usr/bin/env bash
# Test-time mirror experiment (dev3, adopted distilled model) + publish L1 and TTA results to DDData.
set -u
P=/workspace/probeB; R=$P/repo_git; X=$P/out/s15; F=$OUT
for k in 0 1 2; do FLIP=1 SIDE_MODE=distill CAP_THREADS=4 $PY $R/jobs/sidehead2.py extract $P/models/seg3_side_distill.pt $P/out/dev3 dev3 $k 3 $F/dev3_f$k > $F/log_f$k.txt 2>&1 & done
( while sleep 120; do echo PROG $(date -u +%T) $(cat $F/dev3_f*_sidec.jsonl 2>/dev/null | wc -l); done ) & PP=$!
wait %1 %2 %3 2>/dev/null; wait; kill $PP 2>/dev/null
grep -h -E 'Traceback|Error|EXT_DONE' $F/log_f*.txt | grep -v onnxruntime | head -6
$PY $R/jobs/tta_score.py $P/gen "$X/dev3_t*_sidec.jsonl" "$F/dev3_f*_sidec.jsonl" "$X/dev3_t*_ceil.jsonl" $F/tta_dev3.json
# publish
cd /tmp && git clone -q --depth 1 https://x-access-token:${GH_TOKEN}@github.com/josephleporini/DDData.git dd && cd dd
mkdir -p results/l1 results/tta_flip_dev3
cp -r $P/out/l1/orig $P/out/l1/adopted results/l1/ 2>/dev/null; cp $F/tta_dev3.json $F/log_f0.txt results/tta_flip_dev3/
git add -A && git -c user.email=jslepo@gmail.com -c user.name="Joseph Leporini (pod)" commit -qm "L1 ledgers and sheets; test-time mirror on dev3" && git push -q origin main && echo PUBLISHED
