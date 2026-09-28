#!/usr/bin/env bash
# BT-3 early: the BT-1 engine inside the qualification container code (container/d2qual, structured engine), timed on
# an evaluator-class GPU under the APL limits emulated in-process. NOT the Docker image: no build, no --network none,
# no ClamAV (a RunPod pod cannot run Docker). What it does establish: the engine runs end to end through M1/M12/M13,
# per-image time and GPU memory on this GPU, and that the packaged path reproduces the BT-1 extraction rows.
#   R  parity: engine (no mirror) on 24 dev5 scenes vs DDData results/bt1/ext rows
#   X  full dev5 (480 images) through python -m d2qual, mirror on, 8 CPUs (taskset), 14 GB GPU cap, rows logged
#   Y  timing variants on 120 images: mirror off; ORT on GPU if onnxruntime-gpu is usable
#   K  conformance suite, native mode (edge cases, no-model fallback, watchdog, determinism)
#   C  scoring: container classes (in-sample layer) and out-of-fold from the container's own rows
#   P  publish to DDData results/bt3_a40
# dev5 is a development split (scored many times already); no test split is touched. Env: GH_TOKEN, MAX_H (default 1.5)
set -u
P=/workspace/probeB; R=${REPO:?}; MAX_H=${MAX_H:-1.5}; export CAP_THREADS=8
T0=$(date +%s); stage() { echo "=== STAGE $1 $(date -u +%T) elapsed $(( $(date +%s) - T0 ))s"; }
( sleep $(python3 -c "print(int($MAX_H*3600))"); echo "BT3_WALLCLOCK_LIMIT reached"; pkill -P $$; kill $$ ) & WD=$!
fail() { echo "BT3_FAIL $*"; kill $WD 2>/dev/null; exit 5; }
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || fail "no GPU"
NP=$(nproc); CPUS=0-$(( NP < 8 ? NP - 1 : 7 )); echo "HOST nproc $NP mem $(free -g | awk '/Mem/{print $2}')G cpus $CPUS"
$PY -c "import torch; x=torch.randn(64,64,device='cuda'); print('torch', torch.__version__, torch.cuda.get_device_name(0), float((x@x).sum())>-1e9)" || fail "torch"
export PATH=$HOME/.local/bin:$PATH; command -v uv >/dev/null || pip install -q uv; $PY -c "import jsonschema" 2>/dev/null || uv pip install -q --python $PY jsonschema 2>&1 | tail -1; $PY -c "import jsonschema, sam2, rtmlib, onnxruntime as o; print('ORT', o.__version__, o.get_available_providers())" || fail "env"
command -v /usr/bin/time >/dev/null || (apt-get install -y -qq time >/dev/null 2>&1 || true)

DD=$P/dddata_bt3; rm -rf $DD
git clone -q --filter=blob:none --sparse https://x-access-token:${GH_TOKEN}@github.com/josephleporini/DDData.git $DD || fail clone
git -C $DD config user.email jslepo@gmail.com; git -C $DD config user.name "Joseph Leporini (pod)"
git -C $DD sparse-checkout set dev5 results/bt1 || fail sparse
D5=$OUT/dev5; IN=$OUT/in; mkdir -p $D5 $IN; cp $DD/dev5/batch_*/D*.jpg $DD/dev5/batch_*/D*_sidecar.json $D5/; cp $D5/*.jpg $IN/
echo "DEV5 images $(ls $IN | wc -l)"

MD=$OUT/model; mkdir -p $MD
for f in lend3d.pt lwound2.pt sam2_1_hiera_tiny.pt seg4_limb_bt1.pt end2end.onnx 20230928; do
  [ -e $P/models/$f ] || fail "missing model file $f"; ln -sfn $P/models/$f $MD/$f; done
md5sum $P/models/seg4_limb_bt1.pt | tee $OUT/model.md5 | grep -q 68557a9b0990bdc429023116b22d4bb8 || fail "BT-1 weights md5 differs from the BT-1 report"
cp $DD/results/bt1/decision_layer_bt1.json $MD/ || fail "decision layer"
ln -sfn $R $MD/d2med
cat > $MD/model_config.json <<'EOF'
{"engine": "structured", "input_size": 1280, "fallback_class": "no_injury",
 "components": ["sam2.1_hiera_tiny", "rtmw_wholebody", "yolox_m_humanart", "lend3d", "lwound2", "bt1_limb"],
 "structured": {"code_dir": "d2med", "weights": "seg4_limb_bt1.pt", "decision": "decision_layer_bt1.json"}}
EOF
du -shL $MD | tee $OUT/model_size.txt

stage R
taskset -c $CPUS $PY $R/tools/bt3_check.py parity $MD $MD/seg4_limb_bt1.pt $MD/decision_layer_bt1.json $D5 \
  "$DD/results/bt1/ext/bt1_t*_sidec.jsonl" 24 $OUT/parity.json 2>&1 | grep -E 'PARITY|Traceback|Error' | tee $OUT/parity.txt
grep -q 'PARITY_' $OUT/parity.txt || fail "parity run"
grep -q PARITY_OK $OUT/parity.txt || echo "PARITY_DIFF: continuing for timing; accuracy figures below come from the container's own rows"

# gpu memory sampler (whole device, includes anything outside torch)
( while sleep 2; do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits; done ) > $OUT/gpu_mem.txt & GS=$!
run() {  # run <tag> <input dir> [env...]
  local tag=$1 in=$2; shift 2; local o=$OUT/run_$tag; rm -rf $o; mkdir -p $o
  env "$@" D2_INPUT=$in D2_OUTPUT=$o D2_MODEL_DIR=$MD D2_SCHEMA=$R/container/schema/qual-predictions.schema.json \
    D2_TEAM_NAME=d2dev D2_TEAM_EMAIL=jslepo@gmail.com D2_GPU_CAP_GB=14 D2_BUDGET_S=3000 PYTHONPATH=$R/container \
    /usr/bin/time -v taskset -c $CPUS $PY -m d2qual > $OUT/stdout_$tag.txt 2> $OUT/stderr_$tag.txt
  echo "RUN $tag rc=$? $(tail -1 $OUT/stdout_$tag.txt)"; grep -E 'Elapsed|Maximum resident' $OUT/stderr_$tag.txt
  grep -E 'model_loaded|batch_failed|structured_image_failed|Traceback' $OUT/stderr_$tag.txt | head -3 | cut -c1-400
}
stage X
run full $IN D2_TTA_FLIP=1 D2_FEATURES_OUT=$OUT/rows_full.jsonl
[ -s $OUT/run_full/predictions.json ] || fail "no predictions.json"
stage Y
SUB=$OUT/in120; mkdir -p $SUB; ls $IN | awk 'NR % 4 == 1' | head -120 | while read f; do ln -s $IN/$f $SUB/$f; done
run notta $SUB D2_TTA_FLIP=0
if $PY -c "import onnxruntime as o; import sys; sys.exit(0 if 'CUDAExecutionProvider' in o.get_available_providers() else 1)"; then
  run ortgpu $SUB D2_TTA_FLIP=1 ORT_GPU=1
else echo "ORT_GPU skipped: onnxruntime in this venv has no CUDA provider"; fi
kill $GS 2>/dev/null; echo "GPU_MEM_MAX_MIB $(sort -n $OUT/gpu_mem.txt | tail -1)"

stage K
( cd $R/container && CAP_THREADS=8 taskset -c $CPUS $PY harness/conformance.py --model-dir $MD --bulk 32 ) > $OUT/conformance.txt 2>&1
grep -E '^(PASS|FAIL)' $OUT/conformance.txt

stage C
$PY $R/tools/bt3_check.py score $OUT/run_full/predictions.json $OUT/rows_full.jsonl $D5 $MD/decision_layer_bt1.json $OUT/score.json | cut -c1-1500

stage P
PUB=$DD/results/bt3_a40; mkdir -p $PUB
cp $OUT/parity.json $OUT/score.json $OUT/conformance.txt $OUT/model.md5 $OUT/model_size.txt $OUT/gpu_mem.txt $OUT/stdout_*.txt $PUB/
for f in $OUT/stderr_*.txt; do grep -v '"msg": "aux"' $f | tail -60 > $PUB/$(basename $f); done
cp $OUT/run_full/predictions.json $PUB/predictions_dev5.json; gzip -c $OUT/rows_full.jsonl > $PUB/rows_full.jsonl.gz
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader > $PUB/gpu.txt; echo "nproc $(nproc)" >> $PUB/gpu.txt
cd $DD && git add -A results/bt3_a40 && git commit -qm "BT-3 early: BT-1 engine in the qualification container code on $(head -1 $PUB/gpu.txt | cut -d, -f1) (commit $(cat $R/COMMIT))" \
  && for t in 1 2 3; do git pull -q --rebase origin main && git push -q origin main && break; sleep 20; done && echo PUBLISHED
kill $WD 2>/dev/null
echo "BT3_DONE elapsed $(( $(date +%s) - T0 ))s"
