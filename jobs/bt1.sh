#!/usr/bin/env bash
# BT-1 (IPR T2 + T4, R3 flip-rate measurement, T3 per-rule refit). Run under infra/runjob.sh with REPO set to this clone.
#   S  smoke: limb-model shape and mirror checks; adopted-model reproducibility on 12 dev5 scenes; timed training burst
#   T  train LimbModel on train5 (mirror-consistency target), init from the adopted seg3_side_distill.pt
#   E  dev5 extraction, BT-1 model (original and mirrored) and adopted model (mirrored), sharded on CPU plus GPU
#   C  scoring on dev5 (grouped 5-fold OOF): BT-1 without and with limb features; T3 rules; M3-02 flip measure; events
#   P  publish results to DDData results/bt1 (no test5 anywhere in this job)
# Env: STEPS (3600), BATCH (6), SHARDS (6), MAX_H (5, whole-job wall clock), GH_TOKEN (pod secret)
set -u
P=/workspace/probeB; R=${REPO:?}; export PYTHONPATH=$R/gen:$R/jobs SCENE_PREFIX=D
STEPS=${STEPS:-3600}; BATCH=${BATCH:-6}; SH=${SHARDS:-6}; MAX_H=${MAX_H:-5}
TAG=${TAG:-bt1}; export SEED=${SEED:-0}   # TAG=bt1_seed1 SEED=1: second seed, published to results/$TAG, adopted-model and T3 arms skipped
MODEL=$P/models/seg4_limb_$TAG.pt
ADOPT=$P/models/seg3_side_distill.pt
T0=$(date +%s); stage() { echo "=== STAGE $1 $(date -u +%T) elapsed $(( $(date +%s) - T0 ))s"; }
( sleep $((MAX_H * 3600)); echo "BT1_WALLCLOCK_LIMIT ${MAX_H}h reached: stopping"; pkill -P $$; kill $$ ) & WD=$!
fail() {  # publish the log tail before exiting so a self-terminated pod leaves a diagnosis
  echo "BT1_FAIL $*"; kill $WD 2>/dev/null
  if [ -d "${DD:-}/.git" ]; then mkdir -p $DD/results/${TAG:-bt1}_failed && tail -150 /workspace/probeB/logs/${JOBNAME:-bt1}.log > $DD/results/${TAG:-bt1}_failed/FAILED_log.txt
    (cd $DD && git add --sparse -A results/${TAG:-bt1}_failed && git commit -qm "BT-1 ${TAG:-bt1}: failure log ($*)" && git pull -q --rebase origin main && git push -q origin main && echo FAIL_LOG_PUBLISHED); fi
  exit 5; }
[ $TAG = bt1 ] || [ ! -f $MODEL ] || fail "$MODEL exists; refusing to overwrite"
[ -e $R/models ] || ln -sfn $P/models $R/models     # jobs/*.py read ../models (the BT-1 run had this link on its clone)
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || fail "no GPU"
GPUTEST="import torch, torch.nn.functional as F; x = torch.randn(64, 64, device='cuda'); (x @ x).sum().item(); F.conv2d(torch.randn(1, 3, 32, 32, device='cuda'), torch.randn(4, 3, 3, 3, device='cuda')).sum().item()"
if ! $PY -c "$GPUTEST" 2>/dev/null; then
  # the volume venv's torch has no kernels for this GPU (e.g. Blackwell sm_120): use the image's own torch, add the rest
  echo "VENV_TORCH_UNUSABLE $($PY -c 'import torch; print(torch.__version__, torch.cuda.get_arch_list())' 2>&1 | tail -1); switching to system python"
  python3 -m pip install -q rtmlib onnxruntime scikit-learn scipy opencv-python-headless hydra-core iopath omegaconf jsonschema 2>&1 | tail -2
  SAM2_BUILD_CUDA=0 python3 -m pip install -q --no-deps git+https://github.com/facebookresearch/sam2.git 2>&1 | tail -2
  PY=python3; export PY
  $PY -c "$GPUTEST" || fail "no usable torch for this GPU"
  $PY -c "import sam2, rtmlib, sklearn, cv2, onnxruntime; print('SYSTEM_ENV_OK')" || fail "system env"
fi
$PY -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, torch.cuda.get_device_name(0))" || fail "torch"

# data: sparse clone of DDData (train5, dev5, results only; test5 and challenge5 are not checked out)
DD=$P/dddata_bt1; rm -rf $DD
git clone -q --filter=blob:none --sparse https://x-access-token:${GH_TOKEN}@github.com/josephleporini/DDData.git $DD || fail "clone"
git -C $DD config user.email jslepo@gmail.com; git -C $DD config user.name "Joseph Leporini (pod)"
git -C $DD sparse-checkout set train5 dev5 results || fail "sparse"
T5=$OUT/train5; D5=$OUT/dev5; mkdir -p $T5 $D5
cp $DD/train5/batch_*/T* $T5/; cp $DD/dev5/batch_*/D* $D5/
echo "DATA train5 $(ls $T5/*_sidecar.json | wc -l) dev5 $(ls $D5/*_sidecar.json | wc -l)"
$PY $R/score/manifest.py write $D5 $OUT/dev5.sha256 --seeds >/dev/null && cmp <(sort -k2 $OUT/dev5.sha256 | awk '{print $1}') \
   <(sort -k2 $DD/results/l2_dev5/dev5.sha256 | awk '{print $1}') && echo "DEV5_MANIFEST matches L2" || echo "DEV5_MANIFEST differs from L2 (see logs)"
L2=$(dirname $(ls -t $P/out/*/chk_dev5_0.jsonl | head -1)); echo "L2_DIR $L2"
[ -f $L2/dev5_t0_sidec.jsonl ] || fail "L2 extraction rows not found"

stage S
$PY $R/jobs/bt1_limb.py smoke $T5 2>&1 | tail -4 | tee $OUT/smoke.txt; grep -q SMOKE_OK $OUT/smoke.txt || fail "limb smoke"
# reproducibility: adopted model, repo code, 12 dev5 scenes vs the L2 rows (same model, volume code)
mkdir -p $OUT/repro/d; for s in $(ls $D5/*_sidecar.json | head -12); do b=${s%_sidecar.json}; cp $b* $OUT/repro/d/; done
SIDE_MODE=distill CAP_THREADS=8 $PY $R/jobs/sidehead2.py extract $ADOPT $OUT/repro/d dev5 0 1 $OUT/repro/r > $OUT/repro/log.txt 2>&1 || { tail -20 $OUT/repro/log.txt; fail "repro extract"; }
$PY - "$OUT/repro/r_sidec.jsonl" "$L2" <<'EOF' | tee $OUT/repro/result.txt
import json, glob, sys
new = {json.loads(l)['scene']: json.loads(l) for l in open(sys.argv[1])}
old = {}
for f in glob.glob(sys.argv[2] + '/dev5_t*_sidec.jsonl'):
    for l in open(f):
        r = json.loads(l)
        if r['scene'] in new: old[r['scene']] = r
keys = ['vis_px', 'ext_px', 'stump_px', 'wound_px', 'tq_px']; n = bad = 0; worst = 0.0
for s in new:
    for site, a in new[s]['sites'].items():
        b = old[s]['sites'][site]; n += 1
        d = max([abs(a[k] - b[k]) for k in keys] + [abs(x - y) for x, y in zip(a['p_end'] or [], b['p_end'] or [])])
        worst = max(worst, d); bad += d > 2
print('REPRO sites', n, 'differing(>2px or p)', bad, 'worst', round(worst, 4))
print('REPRO_OK' if bad <= n * 0.05 else 'REPRO_DIFF')
EOF
RERUN_BASE=0; grep -q REPRO_OK $OUT/repro/result.txt || { RERUN_BASE=1; echo "adopted baseline will be re-extracted with repo code"; }
# timed burst: 100 steps, to size the run (the step log prints every 50 steps)
INIT=$ADOPT WORKERS=8 timeout 1500 $PY $R/jobs/bt1_limb.py train $T5 100 $OUT/burst.pt $BATCH 2>&1 | grep -E 'BT1_|step|Error|error|memory' | tail -8 | tee $OUT/burst.txt
grep -q '"step": 100' $OUT/burst.txt || fail "training burst"
S50=$(grep '"step": 50,' $OUT/burst.txt | sed 's/.*"s": \([0-9]*\).*/\1/'); S100=$(grep '"step": 100,' $OUT/burst.txt | sed 's/.*"s": \([0-9]*\).*/\1/')
SPS=$(( S100 - S50 )); echo "BURST steps 50-100 in ${SPS}s"
LEFT=$(( MAX_H * 3600 - ($(date +%s) - T0) - 5400 ))        # keep 90 min for extraction, scoring, publishing
FIT=$(( LEFT * 50 / (SPS > 0 ? SPS : 1) )); [ $FIT -lt $STEPS ] && { echo "STEPS reduced $STEPS -> $FIT to fit MAX_H"; STEPS=$FIT; }
[ $STEPS -ge 1200 ] || fail "not enough time for a useful run ($STEPS steps)"
# limb extraction path on 3 scenes with the burst checkpoint, original and mirrored, before committing GPU time
mkdir -p $OUT/lx/d; for s in $(ls $OUT/repro/d/*_sidecar.json | head -3); do cp ${s%_sidecar.json}* $OUT/lx/d/; done
for f in 0 1; do FLIP=$f LIMB=1 SIDE_MODE=distill CAP_THREADS=8 $PY $R/jobs/sidehead2.py extract $OUT/burst.pt $OUT/lx/d dev5 0 1 $OUT/lx/f$f \
  > $OUT/lx/log$f.txt 2>&1 || { tail -20 $OUT/lx/log$f.txt; fail "limb extraction flip=$f"; }; done
$PY - $OUT/lx <<'EOF' || fail "limb rows"
import json, sys, glob
for f in (0, 1):
    R = [json.loads(l) for l in open(f'{sys.argv[1]}/f{f}_sidec.jsonl')]
    assert len(R) == 3 and all('limb' in r['sites'][s] and len(r['sites'][s]['limb']['p_cause']) == 5 for r in R for s in r['sites'])
    assert all(r.get('frame', {}).get('facing') in ('front', 'back', 'unknown') for r in R)
print('LIMB_EXTRACT_OK', R[0]['sites']['LUE']['limb'], R[0]['frame'])
EOF

stage T
INIT=$ADOPT WORKERS=10 $PY $R/jobs/bt1_limb.py train $T5 $STEPS $MODEL $BATCH > $OUT/train.log 2>&1 || { tail -30 $OUT/train.log; fail "train"; }
grep -E 'BT1_' $OUT/train.log; tail -2 $OUT/train.log | head -1
md5sum $MODEL | tee $OUT/model.md5

stage E
ext() {  # ext <tag> <ckpt> <flip 0|1> <limb 0|1>
  for k in $(seq 0 $((SH - 1))); do
    FLIP=$3 LIMB=$4 SIDE_MODE=distill CAP_THREADS=2 $PY $R/jobs/sidehead2.py extract $2 $D5 dev5 $k $SH $OUT/$1_t$k > $OUT/log_$1_$k.txt 2>&1 &
  done
}
ext bt1 $MODEL 0 1; ext bt1f $MODEL 1 1; [ $TAG = bt1 ] && ext adf $ADOPT 1 0
[ $RERUN_BASE = 1 ] && ext ad $ADOPT 0 0
( while sleep 180; do echo PROG $(date -u +%T) $(for t in bt1 bt1f adf ad; do echo $t $(cat $OUT/${t}_t*_sidec.jsonl 2>/dev/null | wc -l); done); done ) & PP=$!
for j in $(jobs -p); do [ $j != $PP ] && [ $j != $WD ] && wait $j; done; kill $PP 2>/dev/null
grep -h -E 'Traceback|Error' $OUT/log_*.txt | grep -v onnxruntime | head -6
for t in bt1 bt1f $([ $TAG = bt1 ] && echo adf); do n=$(cat $OUT/${t}_t*_sidec.jsonl | wc -l); echo "EXT $t $n"; [ $n -ge 470 ] || fail "extraction $t incomplete"; done
AD="$L2/dev5_t*_sidec.jsonl"; ADC="$L2/dev5_t*_ceil.jsonl"; [ $RERUN_BASE = 1 ] && { AD="$OUT/ad_t*_sidec.jsonl"; ADC="$OUT/ad_t*_ceil.jsonl"; }

stage C
set -f
SC="$PY $R/score/score.py --gen $R/gen --sidecars $D5 --truth $L2/chk_dev5_*.jsonl --manifest $OUT/dev5.sha256 --dev dev5"
$SC --phase BT1_dev5_base --pred "$OUT/bt1_t*_sidec.jsonl" --side-truth "$OUT/bt1_t*_ceil.jsonl" --prev $DD/results/l2_dev5/adopted/ledger.jsonl --out $OUT/base 2>&1 | tail -1
$SC --phase BT1_dev5_limb --limb --pred "$OUT/bt1_t*_sidec.jsonl" --side-truth "$OUT/bt1_t*_ceil.jsonl" --prev $DD/results/l2_dev5/adopted/ledger.jsonl --out $OUT/limb 2>&1 | tail -1
[ $RERUN_BASE = 1 ] && $SC --phase BT1_dev5_adopted_repo --pred "$AD" --side-truth "$ADC" --out $OUT/adopted_repo 2>&1 | tail -1
[ $TAG = bt1 ] && for rule in v3_compat guide_primary guide_wound_present guide_amp_any; do       # T3: same predictions, truth under each rule
  $SC --phase T3_adopted_$rule --label-rule $rule --pred "$AD" --side-truth "$ADC" --out $OUT/t3/adopted_$rule 2>&1 | tail -1
  $SC --phase T3_bt1limb_$rule --label-rule $rule --limb --pred "$OUT/bt1_t*_sidec.jsonl" --side-truth "$OUT/bt1_t*_ceil.jsonl" --out $OUT/t3/bt1limb_$rule 2>&1 | tail -1
done
LIMB=0 $PY $R/jobs/tta_score.py $R/gen "$OUT/bt1_t*_sidec.jsonl" "$OUT/bt1f_t*_sidec.jsonl" "$OUT/bt1_t*_ceil.jsonl" $OUT/flip_bt1_base.json | tail -1
LIMB=1 $PY $R/jobs/tta_score.py $R/gen "$OUT/bt1_t*_sidec.jsonl" "$OUT/bt1f_t*_sidec.jsonl" "$OUT/bt1_t*_ceil.jsonl" $OUT/flip_bt1_limb.json | tail -1
[ $TAG = bt1 ] && LIMB=0 $PY $R/jobs/tta_score.py $R/gen "$AD" "$OUT/adf_t*_sidec.jsonl" "$ADC" $OUT/flip_adopted.json | tail -1
set +f

stage P
mkdir -p $DD/results/$TAG && cd $DD/results/$TAG && cp -r $OUT/base $OUT/limb . && { [ -d $OUT/t3 ] && cp -r $OUT/t3 . ; true; } && cp $OUT/flip_*.json $OUT/smoke.txt $OUT/burst.txt \
  $OUT/repro/result.txt $OUT/model.md5 $OUT/dev5.sha256 . && grep -E 'BT1_|"step": [0-9]*00,' $OUT/train.log > train_log.txt
[ -d $OUT/adopted_repo ] && cp -r $OUT/adopted_repo .
mkdir -p ext && cp $OUT/bt1_t*_sidec.jsonl ext/ 2>/dev/null; du -sh .      # extraction rows: events are built from these offline
cd $DD && git add --sparse -A results/$TAG && git commit -qm "BT-1 ($TAG, seed $SEED): limb head and side retrain on train5, dev5 ledgers, T3 rules, M3-02 flip measure (commit $(cat $R/COMMIT))" \
  && for t in 1 2 3; do git pull -q --rebase origin main && git push -q origin main && break; sleep 20; done && echo PUBLISHED
kill $WD 2>/dev/null
echo "BT1_DONE elapsed $(( $(date +%s) - T0 ))s"
