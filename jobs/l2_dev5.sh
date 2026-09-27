#!/usr/bin/env bash
# L2: failure ledger on dev5 (generator vNext), both arms, same method as L1 (jobs/l1/run_l1.sh). Publishes to DDData.
#   1. assemble dev5 from the DDData clone (batch 0 was rendered locally, 1-11 on the dev5 pod)
#   2. checks (original segmenter, true map, rendered side) and adopted-model extraction, 3 shards each, concurrent
#   3. manifest, ledgers for the original and adopted arms, publish
set -u
P=/workspace/probeB; R=${R:-$P/repo_git2}; DD=${DD:-$P/dddata_pod2}; D=$P/out/dev5
export SCENE_PREFIX=D
git -C $DD remote set-url origin https://x-access-token:${GH_TOKEN}@github.com/josephleporini/DDData.git; git -C $DD pull -q --rebase origin main
mkdir -p $D $OUT/maps; cp $DD/dev5/batch_*/D* $D/
N=$(ls $D/*_sidecar.json | wc -l); echo "SCENES dev5 $N"; [ $N -ge 470 ] || { echo "dev5 incomplete"; exit 4; }
cp $R/jobs/checks_job7.py $P/gen/checks_job7_v5.py; cp $R/jobs/sidehead2.py $P/gen/sidehead2_v5.py; cd $P/gen
for k in 0 1 2; do
  CAP_THREADS=2 $PY checks_job7_v5.py $D dev5 $k 3 $OUT/chk_dev5_$k.jsonl > $OUT/log_c$k.txt 2>&1 &
  SIDE_MODE=distill CAP_THREADS=2 SAVE_MAPS=$OUT/maps $PY sidehead2_v5.py extract $P/models/seg3_side_distill.pt $D dev5 $k 3 $OUT/dev5_t$k > $OUT/log_t$k.txt 2>&1 &
done
( while sleep 120; do echo PROG $(date -u +%T) chk $(cat $OUT/chk_dev5_*.jsonl 2>/dev/null | wc -l) ext $(cat $OUT/dev5_t*_sidec.jsonl 2>/dev/null | wc -l); done ) & PP=$!
for j in $(jobs -p); do [ $j != $PP ] && wait $j; done; kill $PP 2>/dev/null
grep -h -E 'Traceback|Error|EXT_DONE|CHK_DONE' $OUT/log_*.txt | grep -v onnxruntime | head -12
$PY $R/score/manifest.py write $D $P/manifests/dev5.sha256 --seeds
$PY - <<'PYEOF'
import json, glob, os
o = open(os.environ['OUT'] + '/orig_dev5.jsonl', 'w'); n = 0
for f in sorted(glob.glob(os.environ['OUT'] + '/chk_dev5_*.jsonl')):
    for l in open(f):
        r = json.loads(l)
        if r.get('pred_sites_ceil'):
            car = json.load(open(f"/workspace/probeB/out/dev5/{r['scene']}_sidecar.json"))
            o.write(json.dumps(dict(scene=r['scene'], split='dev5', labels=car['labels_by_threshold'], sites=r['pred_sites_ceil'], kp=None)) + '\n'); n += 1
print('ORIG_ROWS', n)
PYEOF
set -f
SC="$PY $R/score/score.py --gen $R/gen --sidecars $D --truth $OUT/chk_dev5_*.jsonl --manifest $P/manifests/dev5.sha256 --images $D --maps $OUT/maps"
$SC --phase L2_dev5_orig_seg_trueside --pred "$OUT/orig_dev5.jsonl" --out $OUT/orig
$SC --phase L2_dev5_adopted_gated --pred "$OUT/dev5_t*_sidec.jsonl" --side-truth "$OUT/dev5_t*_ceil.jsonl" --prev $OUT/orig/ledger.jsonl --out $OUT/adopted
set +f
for a in orig adopted; do echo "==== METRICS $a"; cat $OUT/$a/metrics.json; echo; echo "==== RETRO $a"; head -80 $OUT/$a/retro.md; done
cd $DD && mkdir -p results/l2_dev5 && cp -r $OUT/orig $OUT/adopted results/l2_dev5/ && cp $OUT/log_*.txt $P/manifests/dev5.sha256 results/l2_dev5/ \
 && git add -A results/l2_dev5 && git -c user.email=jslepo@gmail.com -c user.name="Joseph Leporini (pod)" commit -qm "L2 failure ledger on dev5 (generator vNext)" \
 && for t in 1 2 3; do git pull -q --rebase origin main && git push -q origin main && break; sleep 20; done && echo PUBLISHED
