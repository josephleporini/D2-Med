#!/usr/bin/env bash
# Real-vs-synthetic fidelity gap on the T5 RESERVE photos (never the primary 81). tools/fidelity_real.py.
# Real photos and per-image rows stay on the volume ($P/real_reserve, $OUT); only the aggregate summary is printed.
# Nothing is pushed to any repository from this job (G3: no real-derived data in public repositories).
# Env: MAX_H (default 1), N_SYN (synthetic images per set, default 200)
set -u
P=/workspace/probeB; R=${REPO:?}; export PYTHONPATH=$R/gen:$R/jobs CAP_THREADS=8 BT1_AMP=1 ORT_GPU=1
MAX_H=${MAX_H:-1}; N=${N_SYN:-200}
( sleep $(python3 -c "print(int($MAX_H*3600))"); echo "FIDELITY_WALLCLOCK_LIMIT"; pkill -P $$; kill $$ ) & WD=$!
[ -e $R/models ] || ln -sfn $P/models $R/models
export PATH=$HOME/.local/bin:$PATH; command -v uv >/dev/null || pip install -q uv
XD=$P/ortgpu12; [ -d $XD/onnxruntime ] || uv pip install -q --python $PY --no-deps --target $XD "onnxruntime-gpu==1.22.0" 2>&1 | tail -2
export PYTHONPATH=$XD:$PYTHONPATH
RD=$P/real_reserve; $PY $R/tools/fidelity_real.py fetch $R/real/reserve_urls.csv $RD 2>&1 | grep -E 'FETCH_' | tail -25
echo "REAL_IMAGES $(ls $RD/R*.jpg 2>/dev/null | wc -l)"; [ $(ls $RD/R*.jpg | wc -l) -ge 50 ] || { echo "FIDELITY_FAIL too few downloads"; exit 5; }
DD=$P/dddata_fid; rm -rf $DD
git clone -q --filter=blob:none --sparse https://x-access-token:${GH_TOKEN}@github.com/josephleporini/DDData.git $DD || exit 5
git -C $DD sparse-checkout set dev5 dev6 dev5_closeup results/bt1
mkdir -p $OUT/syn/dev5 $OUT/syn/dev6 $OUT/syn/dk
cp $(ls $DD/dev5/batch_*/D*.jpg | awk 'NR % 2 == 1' | head -$N) $OUT/syn/dev5/; cp $(ls $DD/dev6/batch_*/V*.jpg | head -$N) $OUT/syn/dev6/; cp $DD/dev5_closeup/DK*.jpg $OUT/syn/dk/
M=$P/models; DEC=$DD/results/bt1/decision_layer_bt1.json
$PY $R/tools/fidelity_real.py measure $M $M/seg4_limb_bt1.pt $DEC real $RD $OUT/real.jsonl 2>&1 | grep -E 'MEASURE|Traceback|Error' | tail -3
for s in dev5 dev6 dk; do $PY $R/tools/fidelity_real.py measure $M $M/seg4_limb_bt1.pt $DEC $s $OUT/syn/$s $OUT/$s.jsonl 2>&1 | grep -E 'MEASURE|Traceback|Error' | tail -3; done
$PY $R/tools/fidelity_real.py compare $OUT $OUT/real.jsonl $OUT/dev5.jsonl $OUT/dev6.jsonl $OUT/dk.jsonl 2>&1 | tail -3
kill $WD 2>/dev/null; echo "FIDELITY_DONE"
