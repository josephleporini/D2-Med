#!/usr/bin/env bash
# R1 (Perception thread, WC-1): decision-layer refit on T5 as a development diagnostic. CPU only.
#   1  labels: private D2_Dev blockt/t5 (read-only token), sha256 checked against the freeze file and T5_LABELS_SHA
#   2  inputs already on the volume from jobs/t5_score.sh: T5 engine rows per model, T5 photos (difference hash only),
#      each model's dev5 rows; checked present, nothing re-extracted
#   3  tools/r1_refit.py: arms A, B, C, D and prior-only, grouped CV, nested selection; decision rule fixed in its header
#   4  AGGREGATES ONLY to the pod log and $OUT; per-photo rows and predictions stay on the volume. NOTHING is published (G3)
# T5 is a development set; this does not touch the T5 single-use marker. Env: GH_TOKEN, T5_LABELS_SHA, MAX_H (default 1)
set -u
P=/workspace/probeB; R=${REPO:?}; export PYTHONPATH=$R/gen:$R/jobs:$R/score:$R/adapt:$R/tools
MAX_H=${MAX_H:-1}; T=$P/out/t5_score; B=$P/out/bt2b; O=${OUT:?}
( sleep $(python3 -c "print(int($MAX_H*3600))"); echo "R1_WALLCLOCK_LIMIT"; pkill -P $$; kill $$ ) & WD=$!
fail() { echo "R1_FAIL $*"; kill $WD 2>/dev/null; exit 5; }
for f in $T/rows_v3.jsonl $T/rows_bt2.jsonl $T/rows_bt1.jsonl $B/rows_v3_dev5.jsonl $B/rows_bt2_dev5.jsonl $P/out/bt2/rows_bt1_dev5.jsonl; do
  [ -s $f ] || fail "missing $f"; done
[ -d $B/sets/dev5 ] || fail "missing $B/sets/dev5"
NPH=$(ls $P/real_t5/R*.jpg 2>/dev/null | wc -l); echo "R1_PHOTOS_ON_VOLUME $NPH"

DV=$O/d2dev; rm -rf $DV
git clone -q --filter=blob:none --sparse https://x-access-token:${GH_TOKEN}@github.com/josephleporini/D2_Dev.git $DV || fail "clone D2_Dev"
git -C $DV sparse-checkout set blockt/t5 || fail sparse
LAB=$DV/blockt/t5/t5_labels_v1.2.csv
(cd $DV/blockt/t5 && sha256sum -c FREEZE_T5_labels.sha256) || fail "labels do not match the freeze file"
[ "$(sha256sum $LAB | cut -d' ' -f1)" = "${T5_LABELS_SHA:?}" ] || fail "labels sha256 differs from T5_LABELS_SHA"
echo "R1_LABELS_OK $(git -C $DV rev-parse --short HEAD)"

( cd / && sha256sum $LAB $T/rows_v3.jsonl $T/rows_bt2.jsonl $T/rows_bt1.jsonl $B/rows_v3_dev5.jsonl $B/rows_bt2_dev5.jsonl \
    $P/out/bt2/rows_bt1_dev5.jsonl $R/tools/r1_refit.py ) > $O/R1_INPUTS.sha256
echo "R1_MANIFEST commit $(cat $R/COMMIT) sha256 $(sha256sum $O/R1_INPUTS.sha256 | cut -c1-16) files $(wc -l < $O/R1_INPUTS.sha256)"

S5=$B/sets/dev5
$PY $R/tools/r1_refit.py $O/r1_refit.json $LAB $P/real_t5 \
  v3=$T/rows_v3.jsonl:$B/rows_v3_dev5.jsonl:$S5 bt2=$T/rows_bt2.jsonl:$B/rows_bt2_dev5.jsonl:$S5 \
  bt1=$T/rows_bt1.jsonl:$P/out/bt2/rows_bt1_dev5.jsonl:$S5 2>&1 | grep -E '^R1|Error|Traceback|File "' || fail "refit tool"
rm -rf $DV; kill $WD 2>/dev/null
echo "R1_COMPLETE"
sleep ${HOLD_S:-240}
