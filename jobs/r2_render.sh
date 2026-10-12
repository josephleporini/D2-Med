#!/usr/bin/env bash
# R2 (Perception thread, WC-1, KP-1): segmentation overlays on the T5 photos and the tagging sheet. GPU, about 15 min.
#   1  labels: private D2_Dev blockt/t5 (read-only token), sha256 checked against the freeze file and T5_LABELS_SHA
#   2  tools/r2_overlays.py errors: R1's out-of-fold v3 predictions recomputed, parity with R1 checked, errors listed
#   3  tools/r2_overlays.py render: v3 engine (lying pick, as scored 30 Sep) on each photo, overlays to the volume
#   4  tools/r2_overlays.py sheet: tagging notebook R2_tag.ipynb beside the overlays
# The KP-1 decision rule is fixed in the header of tools/r2_overlays.py and committed before this job runs.
# COUNTS ONLY to the pod log. Photos, overlays, rows and tags stay on the volume. NOTHING is published (G3, C-14, C-15).
# Env: GH_TOKEN, T5_LABELS_SHA, MAX_H (default 1), HOLD_S
set -u
P=/workspace/probeB; R=${REPO:?}; export PYTHONPATH=$R/gen:$R/jobs:$R/score:$R/adapt:$R/tools CAP_THREADS=8
MAX_H=${MAX_H:-1}; T=$P/out/t5_score; B=$P/out/bt2b; O=${OUT:?}
( sleep $(python3 -c "print(int($MAX_H*3600))"); echo "R2_WALLCLOCK_LIMIT"; pkill -P $$; kill $$ ) & WD=$!
fail() { echo "R2_FAIL $*"; kill $WD 2>/dev/null; exit 5; }
for f in $T/rows_v3.jsonl $B/rows_v3_dev5.jsonl $B/model_bt2/decision_layer_bt1.json $P/models/seg4_limb_bt2b_v3.pt \
         $P/models/lend3d.pt $P/models/lwound2.pt $P/models/sam2_1_hiera_tiny.pt; do [ -s $f ] || fail "missing $f"; done
[ -d $B/sets/dev5 ] || fail "missing $B/sets/dev5"
NPH=$(ls $P/real_t5/R*.jpg 2>/dev/null | wc -l); echo "R2_PHOTOS_ON_VOLUME $NPH"; [ $NPH -ge 60 ] || fail "too few photos"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || fail "no GPU"
export PATH=$HOME/.local/bin:$PATH; command -v uv >/dev/null || pip install -q uv
XD=$P/ortgpu12; [ -d $XD/onnxruntime ] || uv pip install -q --python $PY --no-deps --target $XD "onnxruntime-gpu==1.22.0" 2>&1 | tail -2
export PYTHONPATH=$XD:$PYTHONPATH

DV=$O/d2dev; rm -rf $DV
git clone -q --filter=blob:none --sparse https://x-access-token:${GH_TOKEN}@github.com/josephleporini/D2_Dev.git $DV || fail "clone D2_Dev"
git -C $DV sparse-checkout set blockt/t5 || fail sparse
LAB=$DV/blockt/t5/t5_labels_v1.2.csv
(cd $DV/blockt/t5 && sha256sum -c FREEZE_T5_labels.sha256) || fail "labels do not match the freeze file"
[ "$(sha256sum $LAB | cut -d' ' -f1)" = "${T5_LABELS_SHA:?}" ] || fail "labels sha256 differs from T5_LABELS_SHA"
echo "R2_LABELS_OK $(git -C $DV rev-parse --short HEAD)"
( cd / && sha256sum $LAB $T/rows_v3.jsonl $B/rows_v3_dev5.jsonl $P/models/seg4_limb_bt2b_v3.pt $R/tools/r2_overlays.py ) > $O/R2_INPUTS.sha256
echo "R2_MANIFEST commit $(cat $R/COMMIT) sha256 $(sha256sum $O/R2_INPUTS.sha256 | cut -c1-16)"
echo "R2_RULE_SHA $(sed -n '/KP-1 DECISION RULE/,/itself is Joseph/p' $R/tools/r2_overlays.py | sha256sum | cut -c1-16)"

$PY $R/tools/r2_overlays.py errors $O $LAB $P/real_t5 $T/rows_v3.jsonl $B/rows_v3_dev5.jsonl $B/sets/dev5 2>&1 \
  | grep -E '^R2|Error|Traceback|File "' ; [ -s $O/errors.csv ] || fail "errors stage"
BT1_PICK=lying ORT_GPU=1 BT1_FAST=1 BT1_AMP=1 $PY $R/tools/r2_overlays.py render $O $P/real_t5 $P/models \
  $P/models/seg4_limb_bt2b_v3.pt $B/model_bt2/decision_layer_bt1.json 2>&1 | grep -E '^R2|Error|Traceback|File "'
$PY $R/tools/r2_overlays.py sheet $O 2>&1 | grep -E '^R2|Error|Traceback' || fail "sheet stage"
rm -rf $DV; kill $WD 2>/dev/null
echo "R2_COMPLETE"
sleep ${HOLD_S:-240}
