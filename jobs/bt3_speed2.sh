#!/usr/bin/env bash
# BT-3 speed, round 2 (M3-08). Stage timings from the 28 Sep runs: per pass, detection and pose on the CPU 0.20 s,
# per-limb stage 0.24 s (SAM encoders plus CPU mask analysis), segmenter 0.05 s, side assignment 0.02 s; two passes with
# mirroring. Variants on the same 120 dev5 images, each compared with the reference run (class agreement, largest
# probability difference, accuracy against truth, time and breakdown):
#   ref          as measured 28 Sep (batched encoder)
#   fast         + exact CPU speed-ups (gen/fastops.py, bit-identical by tests/test_fastops.py)
#   amp          + bfloat16 SAM encoders
#   reuse        + mirrored pass reuses detection and pose (mirrored)
#   amp_reuse    both
#   ortgpu       fast + amp + detection and pose on the GPU (onnxruntime-gpu for CUDA 12 in a side directory)
#   ortgpu_exact fast + ORT on GPU, fp32 encoders
# Then the full 480 images with the fastest variant at >= 0.98 class agreement, scored (in-sample and out of fold), and
# the conformance suite with that variant. dev5 is a development split; no test split is touched.
# Env: GH_TOKEN, MAX_H (default 1.5), ORT_VER (1.22.0), RESTAG (bt3_speed2)
set -u
P=/workspace/probeB; R=${REPO:?}; MAX_H=${MAX_H:-1.5}; export CAP_THREADS=8
T0=$(date +%s); stage() { echo "=== STAGE $1 $(date -u +%T) elapsed $(( $(date +%s) - T0 ))s"; }
( sleep $(python3 -c "print(int($MAX_H*3600))"); echo "SPEED2_WALLCLOCK_LIMIT reached"; pkill -P $$; kill $$ ) & WD=$!
fail() {  # publish the log tail before exiting, so a self-terminated pod still leaves a diagnosis
  echo "SPEED2_FAIL $*"; kill $WD 2>/dev/null
  if [ -d "${DD:-}/.git" ] && { [ ! -d "${IN:-/nonexistent}" ] || $PY $R/infra/guard_public.py $IN ${D5:-}; }; then  # G3: no inputs yet, or synthetic inputs mkdir -p $DD/results/${RESTAG:-bt3_speed2} && tail -200 /workspace/probeB/logs/bt3_container.log > $DD/results/${RESTAG:-bt3_speed2}/FAILED_log.txt; cp $OUT/parity_full.txt $DD/results/${RESTAG:-bt3_speed2}/ 2>/dev/null; for f in $OUT/stderr_*.txt; do grep -v '"msg": "aux"' $f | tail -80 > $DD/results/${RESTAG:-bt3_speed2}/FAILED_$(basename $f); done 2>/dev/null
    ls -la $P/models > $DD/results/${RESTAG:-bt3_speed2}/FAILED_models_ls.txt 2>&1
    (cd $DD && git add --sparse -A results/${RESTAG:-bt3_speed2} && git commit -qm "BT-3 speed 2: failure log ($*)" && git pull -q --rebase origin main && git push -q origin main && echo FAIL_LOG_PUBLISHED); fi
  exit 5; }
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || fail "no GPU"
NP=$(nproc); CPUS=0-$(( NP < 8 ? NP - 1 : 7 )); echo "HOST nproc $NP mem $(free -g | awk '/Mem/{print $2}')G cpus $CPUS"
$PY -c "import torch; x=torch.randn(64,64,device='cuda'); print('torch', torch.__version__, torch.cuda.get_device_name(0), float((x@x).sum())>-1e9)" || fail "torch"
export PATH=$HOME/.local/bin:$PATH; command -v uv >/dev/null || pip install -q uv; $PY -c "import jsonschema" 2>/dev/null || uv pip install -q --python $PY jsonschema 2>&1 | tail -1; $PY -c "import jsonschema, sam2, rtmlib, onnxruntime as o; print('ORT', o.__version__, o.get_available_providers())" || fail "env"
command -v /usr/bin/time >/dev/null || (apt-get install -y -qq time >/dev/null 2>&1 || true)

DD=$P/dddata_bt3; rm -rf $DD
git clone -q --filter=blob:none --sparse https://x-access-token:${GH_TOKEN}@github.com/josephleporini/DDData.git $DD || fail clone
git -C $DD config user.email jslepo@gmail.com; git -C $DD config user.name "Joseph Leporini (pod)"
git -C $DD sparse-checkout set dev5 results/bt1 results/${RESTAG:-bt3_speed2} || fail sparse
D5=$OUT/dev5; IN=$OUT/in; mkdir -p $D5 $IN; cp $DD/dev5/batch_*/D*.jpg $DD/dev5/batch_*/D*_sidecar.json $D5/; cp $D5/*.jpg $IN/
echo "DEV5 images $(ls $IN | wc -l)"; ls $P/models | head -40

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


# gpu memory sampler
( while sleep 2; do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits; done ) > $OUT/gpu_mem.txt & GS=$!
run() {  # run <tag> <input dir> [env...]; rows to $OUT/rows_<tag>.jsonl
  local tag=$1 in=$2; shift 2; local o=$OUT/run_$tag; rm -rf $o $OUT/rows_$tag.jsonl; mkdir -p $o
  env D2_TTA_FLIP=1 D2_FEATURES_OUT=$OUT/rows_$tag.jsonl "$@" D2_INPUT=$in D2_OUTPUT=$o D2_MODEL_DIR=$MD \
    D2_SCHEMA=$R/container/schema/qual-predictions.schema.json D2_TEAM_NAME=d2dev D2_TEAM_EMAIL=jslepo@gmail.com \
    D2_GPU_CAP_GB=14 D2_BUDGET_S=3000 PYTHONPATH=${XPP:+$XPP:}$R/container \
    /usr/bin/time -v taskset -c $CPUS $PY -m d2qual > $OUT/stdout_$tag.txt 2> $OUT/stderr_$tag.txt
  echo "RUN $tag rc=$? $(tail -1 $OUT/stdout_$tag.txt | cut -c1-220)"
  grep -E 'batch_failed|structured_image_failed|Traceback' $OUT/stderr_$tag.txt | head -3 | cut -c1-400
}
cmp_() { $PY $R/tools/bt3_check.py compare $OUT/rows_ref.jsonl $OUT/rows_$1.jsonl $D5 $OUT/cmp_$1.json | cut -c1-900; }
SUB=$OUT/in120; rm -rf $SUB; mkdir -p $SUB; ls $IN | awk 'NR % 4 == 1' | head -120 | while read f; do ln -s $IN/$f $SUB/$f; done

stage V1   # reference = the configuration measured on 28 Sep (batched encoder, no other switch)
run ref $SUB BT1_FAST=0 BT1_AMP=0 BT1_POSE_REUSE=0
[ -s $OUT/rows_ref.jsonl ] || fail "reference run"
stage V2   # exact CPU speed-ups
run fast $SUB BT1_FAST=1; cmp_ fast
stage V3   # bfloat16 encoders
run amp $SUB BT1_FAST=1 BT1_AMP=1; cmp_ amp
stage V4   # mirrored pass reuses detection and pose
run reuse $SUB BT1_FAST=1 BT1_POSE_REUSE=1; cmp_ reuse
run amp_reuse $SUB BT1_FAST=1 BT1_AMP=1 BT1_POSE_REUSE=1; cmp_ amp_reuse
stage V5   # detection and pose on the GPU: onnxruntime-gpu built for CUDA 12 in a side directory (the venv is not changed)
XD=$P/ortgpu12; if [ ! -d $XD/onnxruntime ]; then uv pip install -q --python $PY --no-deps --target $XD "onnxruntime-gpu==${ORT_VER:-1.22.0}" 2>&1 | tail -2; fi
XPP=$XD $PY -c "import torch, sys; sys.path.insert(0, '$XD'); import onnxruntime as o; o.preload_dlls(); print('ORTGPU', o.__version__, o.get_available_providers())" 2>&1 | tail -1
XPP=$XD run ortgpu $SUB BT1_FAST=1 BT1_AMP=1 ORT_GPU=1; cmp_ ortgpu
grep -c -i "CUDAExecutionProvider\|Failed to create CUDA" $OUT/stderr_ortgpu.txt | sed 's/^/ORT_CUDA_MESSAGES /'
XPP=$XD run ortgpu_exact $SUB BT1_FAST=1 BT1_AMP=0 ORT_GPU=1; cmp_ ortgpu_exact
kill $GS 2>/dev/null; echo "GPU_MEM_MAX_MIB $(sort -n $OUT/gpu_mem.txt | tail -1)"

stage F    # full dev5 with the fastest variant whose class agreement with the reference is at least 0.98
BEST=$($PY - <<PYEOF
import json, glob, os
c = []
for f in glob.glob('$OUT/cmp_*.json'):
    d = json.load(open(f)); c.append((d['s_var'], os.path.basename(f)[4:-5], d['class_agreement']))
ok = sorted(x for x in c if x[2] >= 0.98)
print(ok[0][1] if ok else 'fast')
PYEOF
)
echo "BEST $BEST"
case $BEST in
  fast) ENV="BT1_FAST=1";; amp) ENV="BT1_FAST=1 BT1_AMP=1";; reuse) ENV="BT1_FAST=1 BT1_POSE_REUSE=1";;
  amp_reuse) ENV="BT1_FAST=1 BT1_AMP=1 BT1_POSE_REUSE=1";; ortgpu) ENV="BT1_FAST=1 BT1_AMP=1 ORT_GPU=1";; ortgpu_exact) ENV="BT1_FAST=1 ORT_GPU=1";;
esac
case $BEST in ortgpu*) export XPP=$XD;; esac
run full $IN $ENV
$PY $R/tools/bt3_check.py score $OUT/run_full/predictions.json $OUT/rows_full.jsonl $D5 $MD/decision_layer_bt1.json $OUT/score.json | cut -c1-1500
stage K
( cd $R/container && CAP_THREADS=8 PYTHONPATH=${XPP:+$XPP:}$R/container env $ENV taskset -c $CPUS $PY harness/conformance.py --model-dir $MD --bulk 32 ) > $OUT/conformance.txt 2>&1
grep -E '^(PASS|FAIL)' $OUT/conformance.txt

stage P
PUB=$DD/results/${RESTAG:-bt3_speed2}; mkdir -p $PUB
$PY $R/infra/guard_public.py $IN $D5 || fail "publish guard"
cp $OUT/cmp_*.json $OUT/score.json $OUT/conformance.txt $OUT/model.md5 $OUT/gpu_mem.txt $OUT/stdout_*.txt $PUB/ 2>/dev/null
for f in $OUT/stderr_*.txt; do grep -v '"msg": "aux"' $f | tail -40 > $PUB/$(basename $f); done
echo "$BEST $ENV" > $PUB/best.txt
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader > $PUB/gpu.txt; echo "nproc $(nproc); $(lscpu | grep 'Model name')" >> $PUB/gpu.txt
cd $DD && git add --sparse -A results/${RESTAG:-bt3_speed2} && git commit -qm "BT-3 speed 2 (${RESTAG:-bt3_speed2}): speed variants on $(head -1 $PUB/gpu.txt | cut -d, -f1) (commit $(cat $R/COMMIT))" \
  && for t in 1 2 3; do git pull -q --rebase origin main && git push -q origin main && break; sleep 20; done && echo PUBLISHED
kill $WD 2>/dev/null
echo "SPEED2_DONE elapsed $(( $(date +%s) - T0 ))s"
