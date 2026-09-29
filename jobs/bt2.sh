#!/usr/bin/env bash
# BT-2: retrain the BT-1 network on corrected and new renders, then compare with BT-1 on the same images.
#   data   train5 (relabelled 29 Sep, garment label fix) + train6 (BT-2 mix: v2 wounds, 1/3 close-ups, blood smear)
#   T      same recipe as BT-1 (jobs/bt1_limb.py train, INIT seg3_side_distill, batch 6), STEPS (default 5400) for 2x data
#   E      both models through the qualification container (fast path: exact CPU fixes, bf16 encoders, det/pose on GPU),
#          rows logged, on dev5 (relabelled), dev6 (new, BT-2 mix) and the close-up set DK (relabelled)
#   C      tools/bt2_eval.py: dev5 out of fold, dev6 and DK with the layer fitted on dev5, paired fixed/broken
#   P      publish to DDData results/${TAG:-bt2} (publish guard on every input set); weights stay on the volume
# Development splits only; test5 is not checked out. Env: GH_TOKEN, STEPS, MAX_H (default 3.5), TAG, SEED
set -u
P=/workspace/probeB; R=${REPO:?}; export PYTHONPATH=$R/gen:$R/jobs
STEPS=${STEPS:-5400}; MAX_H=${MAX_H:-3.5}; TAG=${TAG:-bt2}; export SEED=${SEED:-0}; export CAP_THREADS=8
MODEL=$P/models/seg4_limb_$TAG.pt; BT1=$P/models/seg4_limb_bt1.pt
T0=$(date +%s); stage() { echo "=== STAGE $1 $(date -u +%T) elapsed $(( $(date +%s) - T0 ))s"; }
( sleep $(python3 -c "print(int($MAX_H*3600))"); echo "BT2_WALLCLOCK_LIMIT"; pkill -P $$; kill $$ ) & WD=$!
fail() {
  echo "BT2_FAIL $*"; kill $WD 2>/dev/null
  if [ -d "${DD:-}/.git" ]; then mkdir -p $DD/results/${TAG}_failed && tail -150 $P/logs/$TAG.log > $DD/results/${TAG}_failed/FAILED_log.txt
    (cd $DD && git add --sparse -A results/${TAG}_failed && git commit -qm "BT-2 $TAG: failure log ($*)" && git pull -q --rebase origin main && git push -q origin main); fi
  exit 5; }
[ ! -f $MODEL ] || fail "$MODEL exists; refusing to overwrite"
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
git -C $DD sparse-checkout set train5 train6 dev5 dev6 dev5_closeup results/bt1 || fail sparse
T5=$OUT/train5; T6=$OUT/train6; mkdir -p $T5 $T6 $OUT/sets/dev5 $OUT/sets/dev6 $OUT/sets/dk
cp $DD/train5/batch_*/T* $T5/; cp $DD/train6/batch_*/U* $T6/
cp $DD/dev5/batch_*/D* $OUT/sets/dev5/; cp $DD/dev6/batch_*/V* $OUT/sets/dev6/; cp $DD/dev5_closeup/DK* $OUT/sets/dk/
for s in train5 train6; do echo "DATA $s $(ls $OUT/$s/*_sidecar.json | wc -l)"; done
for s in dev5 dev6 dk; do echo "DATA $s $(ls $OUT/sets/$s/*_sidecar.json | wc -l)"; done
n=$(grep -l '"garment_label_fix"' $T5/*_sidecar.json | wc -l); echo "TRAIN5_RELABELLED_SCENES $n"; [ $n -gt 0 ] || fail "train5 not relabelled"
[ $(ls $T6/*_sidecar.json | wc -l) -ge 590 ] || fail "train6 incomplete"

stage T
INIT=$P/models/seg3_side_distill.pt WORKERS=10 $PY $R/jobs/bt1_limb.py train "$T5,$T6" $STEPS $MODEL 6 > $OUT/train.log 2>&1 || { tail -30 $OUT/train.log; fail train; }
grep -E 'BT1_' $OUT/train.log; md5sum $MODEL | tee $OUT/model.md5

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
mdir bt1 $BT1; mdir $TAG $MODEL
run() {  # run <model> <set>
  local o=$OUT/run_$1_$2; rm -rf $o $OUT/rows_$1_$2.jsonl; mkdir -p $o
  env D2_TTA_FLIP=1 BT1_FAST=1 BT1_AMP=1 ORT_GPU=1 D2_FEATURES_OUT=$OUT/rows_$1_$2.jsonl D2_INPUT=$OUT/in_$2 D2_OUTPUT=$o \
    D2_MODEL_DIR=$OUT/model_$1 D2_SCHEMA=$R/container/schema/qual-predictions.schema.json D2_TEAM_NAME=d2dev \
    D2_TEAM_EMAIL=jslepo@gmail.com D2_GPU_CAP_GB=14 D2_BUDGET_S=3000 PYTHONPATH=$XD:$R/container \
    taskset -c $CPUS $PY -m d2qual > $OUT/stdout_$1_$2.txt 2> $OUT/stderr_$1_$2.txt
  echo "RUN $1 $2 rc=$? $(tail -1 $OUT/stdout_$1_$2.txt | cut -c1-200)"
  grep -E 'structured_image_failed|Traceback' $OUT/stderr_$1_$2.txt | head -2 | cut -c1-300
}
for s in dev5 dev6 dk; do mkdir -p $OUT/in_$s; cp $OUT/sets/$s/*.jpg $OUT/in_$s/; done
for m in bt1 $TAG; do for s in dev5 dev6 dk; do run $m $s; done; done

stage C
ARGS=""; for m in bt1 $TAG; do for s in dev5 dev6 dk; do ARGS="$ARGS $m=$s:$OUT/rows_${m}_$s.jsonl:$OUT/sets/$s"; done; done
$PY $R/tools/bt2_eval.py $OUT/bt2_eval.json $ARGS 2>&1 | grep -E 'BT2EVAL|BT2PAIR|Error' | cut -c1-600

stage P
PUB=$DD/results/$TAG; mkdir -p $PUB
for s in dev5 dev6 dk; do $PY $R/infra/guard_public.py $OUT/in_$s $OUT/sets/$s || fail "publish guard $s"; done
cp $OUT/bt2_eval.json $OUT/model.md5 $OUT/stdout_*.txt $PUB/; grep -E 'BT1_|"step": [0-9]*00,' $OUT/train.log > $PUB/train_log.txt
for f in $OUT/rows_*.jsonl; do gzip -c $f > $PUB/$(basename $f).gz; done
cd $DD && git add --sparse -A results/$TAG && git commit -qm "BT-2 ($TAG): retrain on train5 (relabelled) + train6, compared with BT-1 on dev5, dev6, close-ups (commit $(cat $R/COMMIT))" \
  && for t in 1 2 3; do git pull -q --rebase origin main && git push -q origin main && break; sleep 20; done && echo PUBLISHED
kill $WD 2>/dev/null
echo "BT2_DONE elapsed $(( $(date +%s) - T0 ))s"
