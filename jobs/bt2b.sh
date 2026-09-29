#!/usr/bin/env bash
# BT-2b: training toward real photographs (Real Fidelity Gap v1.0; BT-2 Results v1.0 section 1 item 5).
#   arms   aug : BT-2 fine-tuned on train5 + train6 with photometric and background augmentation (AUG_PHOTO=1)
#          v3  : BT-2 fine-tuned on train5 + train6 + train7 (generator v3) with the same augmentation
#          both start from seg4_limb_bt2.pt, STEPS each (default 4000), batch 6, seed 0
#   E      bt2 (largest-box pick, as shipped), bt2 lying pick (bt2L), aug and v3 (lying pick) through the qualification
#          container on dev5, dev6, dev7 (generator v3, bystanders) and the close-up set DK
#   C      tools/bt2_eval.py (dev5 out of fold; others with the layer fitted on dev5), paired fixed/broken
#   R      real-photo proxy on the 200 T5 RESERVE photos (never the primary 81): tools/fidelity_real.py measure per model,
#          tools/real_proxy.py against the screening class, domain gap vs dev5 and dev7. Aggregates are printed and kept
#          on the volume; nothing real-derived is published (G3)
#   P      publish synthetic results to DDData results/${TAG:-bt2b} (publish guard on every input set)
# Development splits only; test5 is not checked out. Env: GH_TOKEN, STEPS, MAX_H (default 5), TAG, SEED, HOLD_S
set -u
P=/workspace/probeB; R=${REPO:?}; export PYTHONPATH=$R/gen:$R/jobs
STEPS=${STEPS:-4000}; MAX_H=${MAX_H:-5}; TAG=${TAG:-bt2b}; export SEED=${SEED:-0}; export CAP_THREADS=8
MA=$P/models/seg4_limb_${TAG}_aug.pt; MV=$P/models/seg4_limb_${TAG}_v3.pt; BT2=$P/models/seg4_limb_bt2.pt
T0=$(date +%s); stage() { echo "=== STAGE $1 $(date -u +%T) elapsed $(( $(date +%s) - T0 ))s"; }
( sleep $(python3 -c "print(int($MAX_H*3600))"); echo "BT2B_WALLCLOCK_LIMIT"; pkill -P $$; kill $$ ) & WD=$!
fail() {
  echo "BT2B_FAIL $*"; kill $WD 2>/dev/null
  if [ -d "${DD:-}/.git" ]; then mkdir -p $DD/results/${TAG}_failed && grep -v -E "FIDELITY_SUMMARY|MEASURE_DONE|FETCH_|REAL_" $P/logs/$TAG.log | tail -150 > $DD/results/${TAG}_failed/FAILED_log.txt
    (cd $DD && git add --sparse -A results/${TAG}_failed && git commit -qm "BT-2b $TAG: failure log ($*)" && git pull -q --rebase origin main && git push -q origin main); fi
  exit 5; }
for f in $MA $MV; do [ ! -f $f ] || fail "$f exists; refusing to overwrite"; done; [ -f $BT2 ] || fail "no BT-2 model"
[ -e $R/models ] || ln -sfn $P/models $R/models
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || fail "no GPU"
NP=$(nproc); CPUS=0-$(( NP < 8 ? NP - 1 : 7 ))
export PATH=$HOME/.local/bin:$PATH; command -v uv >/dev/null || pip install -q uv
$PY -c "import jsonschema" 2>/dev/null || uv pip install -q --python $PY jsonschema 2>&1 | tail -1
command -v /usr/bin/time >/dev/null || (apt-get install -y -qq time >/dev/null 2>&1 || true)
XD=$P/ortgpu12; [ -d $XD/onnxruntime ] || uv pip install -q --python $PY --no-deps --target $XD "onnxruntime-gpu==1.22.0" 2>&1 | tail -2

DD=$P/dddata_$TAG; rm -rf $DD
git clone -q --filter=blob:none --sparse https://x-access-token:${GH_TOKEN}@github.com/josephleporini/DDData.git $DD || fail clone
git -C $DD config user.email jslepo@gmail.com; git -C $DD config user.name "Joseph Leporini (pod)"
git -C $DD sparse-checkout set train5 train6 train7 dev5 dev6 dev7 dev5_closeup results/bt1 || fail sparse
T5=$OUT/train5; T6=$OUT/train6; T7=$OUT/train7; mkdir -p $T5 $T6 $T7 $OUT/sets/dev5 $OUT/sets/dev6 $OUT/sets/dev7 $OUT/sets/dk
cp $DD/train5/batch_*/T* $T5/; cp $DD/train6/batch_*/U* $T6/; cp $DD/train7/batch_*/W* $T7/
cp $DD/dev5/batch_*/D* $OUT/sets/dev5/; cp $DD/dev6/batch_*/V* $OUT/sets/dev6/; cp $DD/dev7/batch_*/Y* $OUT/sets/dev7/; cp $DD/dev5_closeup/DK* $OUT/sets/dk/
for s in train5 train6 train7; do echo "DATA $s $(ls $OUT/$s/*_sidecar.json | wc -l)"; done
for s in dev5 dev6 dev7 dk; do echo "DATA $s $(ls $OUT/sets/$s/*_sidecar.json | wc -l)"; done
n=$(grep -l '"garment_label_fix"' $T5/*_sidecar.json | wc -l); echo "TRAIN5_RELABELED_SCENES $n"; [ $n -gt 0 ] || fail "train5 not relabeled"
[ $(ls $T6/*_sidecar.json | wc -l) -ge 590 ] || fail "train6 incomplete"
[ $(ls $T7/*_sidecar.json | wc -l) -ge 760 ] || fail "train7 incomplete"; [ $(ls $OUT/sets/dev7/*_sidecar.json | wc -l) -ge 230 ] || fail "dev7 incomplete"

stage T
AUG_PHOTO=1 INIT=$BT2 WORKERS=10 $PY $R/jobs/bt1_limb.py train "$T5,$T6" $STEPS $MA 6 > $OUT/train_aug.log 2>&1 || { tail -30 $OUT/train_aug.log; fail train_aug; }
grep -E 'BT1_' $OUT/train_aug.log | sed 's/^/aug /'
AUG_PHOTO=1 INIT=$BT2 WORKERS=10 $PY $R/jobs/bt1_limb.py train "$T5,$T6,$T7" $STEPS $MV 6 > $OUT/train_v3.log 2>&1 || { tail -30 $OUT/train_v3.log; fail train_v3; }
grep -E 'BT1_' $OUT/train_v3.log | sed 's/^/v3 /'
md5sum $MA $MV | tee $OUT/model.md5

stage E
mdir() {  # mdir <name> <ckpt>
  local M=$OUT/model_$1; mkdir -p $M
  for f in lend3d.pt lwound2.pt sam2_1_hiera_tiny.pt end2end.onnx 20230928; do ln -sfn $P/models/$f $M/$f; done
  ln -sfn $2 $M/weights.pt; cp $DD/results/bt1/decision_layer_bt1.json $M/; ln -sfn $R $M/d2med
  cat > $M/model_config.json <<EOF
{"engine": "structured", "input_size": 1280, "fallback_class": "no_injury",
 "components": ["sam2.1_hiera_tiny", "rtmw_wholebody", "yolox_m_humanart", "lend3d", "lwound2", "bt1_limb"],
 "structured": {"code_dir": "d2med", "weights": "weights.pt", "decision": "decision_layer_bt1.json"}}
EOF
}
mdir bt2 $BT2; mdir bt2L $BT2; mdir aug $MA; mdir v3 $MV
run() {  # run <model> <set>; bt2 keeps the shipped largest-box pick, every other arm picks the lying person
  local o=$OUT/run_$1_$2 pick=lying; rm -rf $o $OUT/rows_$1_$2.jsonl; mkdir -p $o; [ $1 = bt2 ] && pick=largest
  env BT1_PICK=$pick D2_TTA_FLIP=1 BT1_FAST=1 BT1_AMP=1 ORT_GPU=1 D2_FEATURES_OUT=$OUT/rows_$1_$2.jsonl D2_INPUT=$OUT/in_$2 D2_OUTPUT=$o \
    D2_MODEL_DIR=$OUT/model_$1 D2_SCHEMA=$R/container/schema/qual-predictions.schema.json D2_TEAM_NAME=d2dev \
    D2_TEAM_EMAIL=jslepo@gmail.com D2_GPU_CAP_GB=14 D2_BUDGET_S=3000 PYTHONPATH=$XD:$R/container \
    taskset -c $CPUS $PY -m d2qual > $OUT/stdout_$1_$2.txt 2> $OUT/stderr_$1_$2.txt
  echo "RUN $1 $2 rc=$? $(tail -1 $OUT/stdout_$1_$2.txt | cut -c1-200)"
  grep -E 'structured_image_failed|Traceback' $OUT/stderr_$1_$2.txt | head -2 | cut -c1-300
}
for s in dev5 dev6 dev7 dk; do mkdir -p $OUT/in_$s; cp $OUT/sets/$s/*.jpg $OUT/in_$s/; done
for m in bt2 bt2L aug v3; do for s in dev7 dev5 dk dev6; do run $m $s; done; done

stage C
ARGS=""; for m in bt2 bt2L aug v3; do for s in dev5 dev6 dev7 dk; do ARGS="$ARGS $m=$s:$OUT/rows_${m}_$s.jsonl:$OUT/sets/$s"; done; done
$PY $R/tools/bt2_eval.py $OUT/bt2_eval.json $ARGS 2>&1 | grep -E 'BT2EVAL|BT2PAIR|Error' | cut -c1-600

stage R
RD=$P/real_reserve; RO=$P/out/${TAG}_real; mkdir -p $RO $RO/syn7          # volume only (G3)
if [ $(ls $RD/R*.jpg 2>/dev/null | wc -l) -ge 50 ]; then
  DEC=$DD/results/bt1/decision_layer_bt1.json; export PYTHONPATH=$XD:$PYTHONPATH ORT_GPU=1 BT1_AMP=1
  cp $(ls $OUT/sets/dev7/Y*.jpg | head -200) $RO/syn7/
  for m in bt2:largest:$BT2 bt2L:lying:$BT2 aug:lying:$MA v3:lying:$MV; do IFS=: read n pk ck <<< "$m"
    BT1_PICK=$pk $PY $R/tools/fidelity_real.py measure $P/models $ck $DEC real $RD $RO/real_$n.jsonl 2>&1 | grep -E 'MEASURE|Traceback|Error' | tail -2; done
  BT1_PICK=lying $PY $R/tools/fidelity_real.py measure $P/models $MV $DEC dev7 $RO/syn7 $RO/dev7.jsonl 2>&1 | grep -E 'MEASURE|Traceback|Error' | tail -2
  $PY $R/tools/real_proxy.py $RO/real_proxy.json bt2=$RO/real_bt2.jsonl bt2L=$RO/real_bt2L.jsonl aug=$RO/real_aug.jsonl v3=$RO/real_v3.jsonl 2>&1 | tail -6
  [ -f $P/out/fidelity_real/dev5.jsonl ] && $PY $R/tools/fidelity_real.py compare $RO $RO/real_bt2.jsonl $P/out/fidelity_real/dev5.jsonl $RO/dev7.jsonl 2>&1 \
    | tail -1 | python3 -c "import sys,json; t=sys.stdin.read(); d=json.loads(t.split(' ',1)[1]) if t.startswith('SUMMARY') else {}; print('REAL_GAP', json.dumps({k: d.get(k) for k in ('domain_auc','nn_cosine')}), json.dumps({k: v for k, v in d.get('stats', {}).items() if k in ('contrast','sat','bright','red_share','n_person')}))"
else echo "REAL_PROXY_SKIPPED no reserve photos on the volume"; fi

stage P
PUB=$DD/results/$TAG; mkdir -p $PUB
for s in dev5 dev6 dev7 dk; do $PY $R/infra/guard_public.py $OUT/in_$s $OUT/sets/$s || fail "publish guard $s"; done
cp $OUT/bt2_eval.json $OUT/model.md5 $OUT/stdout_*.txt $PUB/; for a in aug v3; do grep -E 'BT1_|"step": [0-9]*00,' $OUT/train_$a.log > $PUB/train_log_$a.txt; done
for f in $OUT/rows_*.jsonl; do gzip -c $f > $PUB/$(basename $f).gz; done
cd $DD && git add --sparse -A results/$TAG && git commit -qm "BT-2b ($TAG): photometric augmentation and generator v3 arms vs BT-2 on dev5, dev6, dev7, close-ups (commit $(cat $R/COMMIT)); synthetic only" \
  && for t in 1 2 3; do git pull -q --rebase origin main && git push -q origin main && break; sleep 20; done && echo PUBLISHED
kill $WD 2>/dev/null
echo "BT2B_DONE elapsed $(( $(date +%s) - T0 ))s"
[ -f $P/out/${TAG}_real/real_proxy.json ] && echo "REAL_PROXY_REPEAT $(tr -d '\n ' < $P/out/${TAG}_real/real_proxy.json)"
sleep ${HOLD_S:-300}     # keep the pod log readable for a few minutes (it vanishes on termination)
