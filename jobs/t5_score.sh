#!/usr/bin/env bash
# T5 single scoring run (T5 Labeling Protocol v1.0 section 4; Joseph 29 Sep: one run, BT-1, BT-2 and both BT-2b arms).
#   1  refuses if T5 was already scored (marker on the volume)
#   2  labels: private D2_Dev blockt/t5 (read-only token), sha256 checked against the freeze file and T5_LABELS_SHA
#   3  photos: the usable T5 primary photos fetched from their public pages to the volume ($P/real_t5), never to a repo
#   4  freeze manifest on the volume: labels, photos, model weights, dev5 rows used to fit each decision layer, commit
#   5  models through the qualification container: bt1 and bt2 (largest-box pick, as shipped), bt2L, aug, v3 (lying pick)
#   6  tools/t5_score.py: aggregates only, printed; per-image rows stay on the volume. NOTHING is published (G3)
# Needs the BT-2b outputs on the volume ($P/out/bt2b). Env: GH_TOKEN, T5_LABELS_SHA, MAX_H (default 3), HOLD_S
set -u
P=/workspace/probeB; R=${REPO:?}; export PYTHONPATH=$R/gen:$R/jobs CAP_THREADS=8
MAX_H=${MAX_H:-3}; O=$P/out/t5_score; B=$P/out/bt2b
( sleep $(python3 -c "print(int($MAX_H*3600))"); echo "T5_WALLCLOCK_LIMIT"; pkill -P $$; kill $$ ) & WD=$!
fail() { echo "T5_FAIL $*"; kill $WD 2>/dev/null; exit 5; }
[ -f $O/_SCORED ] && fail "T5 already scored ($(cat $O/_SCORED)); single use as a test"
mkdir -p $O
for f in $B/model_bt2/decision_layer_bt1.json $P/models/seg4_limb_bt1.pt $P/models/seg4_limb_bt2.pt $P/models/seg4_limb_bt2b_aug.pt $P/models/seg4_limb_bt2b_v3.pt \
         $P/out/bt2/rows_bt1_dev5.jsonl $B/rows_bt2_dev5.jsonl $B/rows_bt2L_dev5.jsonl $B/rows_aug_dev5.jsonl $B/rows_v3_dev5.jsonl; do
  [ -s $f ] || fail "missing $f"; done
[ -e $R/models ] || ln -sfn $P/models $R/models
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || fail "no GPU"
NP=$(nproc); CPUS=0-$(( NP < 8 ? NP - 1 : 7 ))
export PATH=$HOME/.local/bin:$PATH; command -v uv >/dev/null || pip install -q uv
$PY -c "import jsonschema" 2>/dev/null || uv pip install -q --python $PY jsonschema 2>&1 | tail -1
XD=$P/ortgpu12; [ -d $XD/onnxruntime ] || uv pip install -q --python $PY --no-deps --target $XD "onnxruntime-gpu==1.22.0" 2>&1 | tail -2

# 2 labels (private repository, read-only)
DV=$O/d2dev; rm -rf $DV
git clone -q --filter=blob:none --sparse https://x-access-token:${GH_TOKEN}@github.com/josephleporini/D2_Dev.git $DV || fail "clone D2_Dev"
git -C $DV sparse-checkout set blockt/t5 || fail sparse
LAB=$DV/blockt/t5/t5_labels_v1.2.csv
(cd $DV/blockt/t5 && sha256sum -c FREEZE_T5_labels.sha256) || fail "labels do not match the freeze file"
[ "$(sha256sum $LAB | cut -d' ' -f1)" = "${T5_LABELS_SHA:?}" ] || fail "labels sha256 differs from T5_LABELS_SHA"
echo "T5_LABELS_OK $(git -C $DV rev-parse --short HEAD) $(grep -c ',yes,' $LAB) usable"

# 3 photos (usable only) to the volume
RD=$P/real_t5; mkdir -p $RD
python3 - "$LAB" "$O/fetch.csv" <<'EOF'
import csv, sys
w = csv.writer(open(sys.argv[2], 'w', newline='')); w.writerow(['n', 'page_url', 'screening_class', 'pose', 'camera_view'])
for r in csv.DictReader(open(sys.argv[1])):
    if r['usable'] == 'yes':
        w.writerow([r['n'], r['page_url'], '', '', ''])
EOF
$PY $R/tools/fidelity_real.py fetch $O/fetch.csv $RD 2>&1 | grep -E 'FETCH_' | tail -12
mkdir -p $O/in_t5; rm -f $O/in_t5/*; for f in $RD/R*.jpg; do ln -sfn $f $O/in_t5/; done
NF=$(ls $O/in_t5/*.jpg | wc -l); echo "T5_PHOTOS $NF"; [ $NF -ge 60 ] || fail "too few photos ($NF)"

# 4 freeze manifest (volume)
( cd / && sha256sum $LAB $RD/R*.jpg $P/models/seg4_limb_bt1.pt $P/models/seg4_limb_bt2.pt $P/models/seg4_limb_bt2b_aug.pt \
    $P/models/seg4_limb_bt2b_v3.pt $P/out/bt2/rows_bt1_dev5.jsonl $B/rows_*_dev5.jsonl ) > $O/FREEZE_T5.sha256
echo "T5_FREEZE commit $(cat $R/COMMIT) manifest_sha256 $(sha256sum $O/FREEZE_T5.sha256 | cut -c1-16) files $(wc -l < $O/FREEZE_T5.sha256)"
grep -E 'seg4_limb' $O/FREEZE_T5.sha256 | sed 's|/workspace/probeB/models/||' | cut -c1-16,65-

# 5 models through the container
mdir() {  # mdir <name> <ckpt>
  local M=$O/model_$1; mkdir -p $M
  for f in lend3d.pt lwound2.pt sam2_1_hiera_tiny.pt end2end.onnx 20230928; do ln -sfn $P/models/$f $M/$f; done
  ln -sfn $2 $M/weights.pt; cp $B/model_bt2/decision_layer_bt1.json $M/;   # container needs a layer; the score refits one per model on dev5
  ln -sfn $R $M/d2med
  cat > $M/model_config.json <<EOF
{"engine": "structured", "input_size": 1280, "fallback_class": "no_injury",
 "components": ["sam2.1_hiera_tiny", "rtmw_wholebody", "yolox_m_humanart", "lend3d", "lwound2", "bt1_limb"],
 "structured": {"code_dir": "d2med", "weights": "weights.pt", "decision": "decision_layer_bt1.json"}}
EOF
}
mdir bt1 $P/models/seg4_limb_bt1.pt; mdir bt2 $P/models/seg4_limb_bt2.pt; mdir bt2L $P/models/seg4_limb_bt2.pt
mdir aug $P/models/seg4_limb_bt2b_aug.pt; mdir v3 $P/models/seg4_limb_bt2b_v3.pt
run() {  # run <model>; bt1 and bt2 keep the shipped largest-box pick, the others pick the lying person
  local o=$O/run_$1 pick=lying; rm -rf $o $O/rows_$1.jsonl; mkdir -p $o; case $1 in bt1|bt2) pick=largest;; esac
  env BT1_PICK=$pick D2_TTA_FLIP=1 BT1_FAST=1 BT1_AMP=1 ORT_GPU=1 D2_FEATURES_OUT=$O/rows_$1.jsonl D2_INPUT=$O/in_t5 D2_OUTPUT=$o \
    D2_MODEL_DIR=$O/model_$1 D2_SCHEMA=$R/container/schema/qual-predictions.schema.json D2_TEAM_NAME=d2dev \
    D2_TEAM_EMAIL=jslepo@gmail.com D2_GPU_CAP_GB=14 D2_BUDGET_S=3000 PYTHONPATH=$XD:$R/container \
    taskset -c $CPUS $PY -m d2qual > $O/stdout_$1.txt 2> $O/stderr_$1.txt
  echo "RUN $1 rc=$? rows $(wc -l < $O/rows_$1.jsonl) $(tail -1 $O/stdout_$1.txt | cut -c1-160)"
  grep -cE 'structured_image_failed' $O/stderr_$1.txt | sed 's/^/  image failures: /'
}
for m in bt1 bt2 bt2L aug v3; do run $m; done

# 6 score once, aggregates only
S5=$B/sets/dev5
echo "T5 scored $(date -u +%FT%TZ) commit $(cat $R/COMMIT)" > $O/_SCORED
$PY $R/tools/t5_score.py $O/t5_score.json $LAB bt1=$O/rows_bt1.jsonl:$P/out/bt2/rows_bt1_dev5.jsonl:$S5 \
  bt2=$O/rows_bt2.jsonl:$B/rows_bt2_dev5.jsonl:$S5 bt2L=$O/rows_bt2L.jsonl:$B/rows_bt2L_dev5.jsonl:$S5 \
  aug=$O/rows_aug.jsonl:$B/rows_aug_dev5.jsonl:$S5 v3=$O/rows_v3.jsonl:$B/rows_v3_dev5.jsonl:$S5 2>&1 | grep -E 'T5SCORE|T5PAIRED|Error|Traceback'
kill $WD 2>/dev/null
echo "T5_DONE"; echo "T5_JSON $(tr -d '\n ' < $O/t5_score.json | cut -c1-6000)"
sleep ${HOLD_S:-300}     # keep the pod log readable for a few minutes (it vanishes on termination)
