#!/usr/bin/env bash
# Model maturity item 1 (28 Sep): close-up framing robustness. Re-renders 120 dev5 casualties with the camera on one limb
# (tools/make_closeup_params.py, generator framing 'limb_closeup'), extracts with the BT-1 model (original and
# mirrored), and scores with the shipped decision layer. Publishes to DDData results/closeup_v1 (renders to dev5_closeup/).
# Env: N (120), JOBS (4 parallel renders), MAX_H (1.5), GH_TOKEN
set -u
P=/workspace/probeB; R=${REPO:?}; BV=$P/venv_bpy; N=${N:-120}; J=${JOBS:-4}; MAX_H=${MAX_H:-1.5}; SH=4
export PYTHONPATH=$R/gen:$R/jobs SCENE_PREFIX=DK
( sleep $(python3 -c "print(int($MAX_H*3600))"); echo "CLOSEUP_WALLCLOCK_LIMIT"; pkill -P $$; kill $$ ) & WD=$!
fail() { echo "CLOSEUP_FAIL $*"; kill $WD 2>/dev/null
  if [ -d "${DD:-}/.git" ]; then mkdir -p $DD/results/closeup_v1 && tail -150 $P/logs/closeup.log > $DD/results/closeup_v1/FAILED_log.txt
    (cd $DD && git add --sparse -A results/closeup_v1 && git commit -qm "closeup: failure log ($*)" && git pull -q --rebase origin main && git push -q origin main); fi
  exit 5; }
[ -e $R/models ] || ln -sfn $P/models $R/models; ln -sfn $P/assets $R/assets
DEBIAN_FRONTEND=noninteractive apt-get install -y -qq libx11-6 libxrender1 libxxf86vm1 libxfixes3 libxi6 libxkbcommon0 libsm6 libice6 libegl1 >/dev/null 2>&1
[ -x $BV/bin/python ] || fail "bpy venv missing"
nvidia-smi --query-gpu=name --format=csv,noheader
DD=$P/dddata_closeup; rm -rf $DD
git clone -q --filter=blob:none --sparse https://x-access-token:${GH_TOKEN}@github.com/josephleporini/DDData.git $DD || fail clone
git -C $DD config user.email jslepo@gmail.com; git -C $DD config user.name "Joseph Leporini (pod)"
git -C $DD sparse-checkout set dev5 results/bt1 results/closeup_v1 dev5_closeup || fail sparse
C=$OUT/closeup; rm -rf $C; mkdir -p $C/params $C/r
$PY $R/tools/make_closeup_params.py $DD/dev5 $C/params $N || fail params
echo "=== RENDER $(date -u +%T)"
export BV R C; one() { cd $R/gen && CYCLES_DEVICE=OPTIX timeout 900 $BV/bin/python scene4.py full $1 $C/r 2>&1 | grep -E '^DONE|Traceback' | head -2; }
export -f one
ls $C/params/*_params.json | xargs -P $J -I{} bash -c 'one {}' > $C/render.log
n=$(ls $C/r/*_sidecar.json | wc -l); echo "RENDERED $n of $N"; [ $n -ge $((N - 4)) ] || { tail -5 $C/render.log; fail "renders $n"; }
rm -f $C/r/_alone_*
echo "=== EXTRACT $(date -u +%T)"
for f in 0 1; do for k in $(seq 0 $((SH - 1))); do
  FLIP=$f LIMB=1 SIDE_MODE=distill CAP_THREADS=2 $PY $R/jobs/sidehead2.py extract $P/models/seg4_limb_bt1.pt $C/r closeup $k $SH $C/x${f}_t$k > $C/xlog${f}_$k.txt 2>&1 &
done; done; wait
for f in 0 1; do echo "EXT flip$f $(cat $C/x${f}_t*_sidec.jsonl | wc -l)"; done
grep -h -E 'Traceback|Error' $C/xlog*.txt | grep -v onnxruntime | head -4
echo "=== SCORE $(date -u +%T)"
$PY $R/tools/closeup_score.py $C/r "$C/x0_t*_sidec.jsonl" "$C/x1_t*_sidec.jsonl" $DD/results/bt1/decision_layer_bt1.json \
  $DD/results/bt1/limb/ledger.jsonl $C/score.json || fail score
PUB=$DD/results/closeup_v1; mkdir -p $PUB $DD/dev5_closeup
$PY $R/infra/guard_public.py $C/r || fail "publish guard"
cp $C/score.json $C/render.log $PUB/; cat $C/x0_t*_sidec.jsonl > $PUB/rows_orig.jsonl; cat $C/x1_t*_sidec.jsonl > $PUB/rows_flip.jsonl
cp $C/r/DK* $DD/dev5_closeup/
cd $DD && git add --sparse -A results/closeup_v1 dev5_closeup && git commit -qm "Close-up framing robustness set: 120 dev5 casualties, BT-1 scored (commit $(cat $R/COMMIT))" \
  && for t in 1 2 3; do git pull -q --rebase origin main && git push -q origin main && break; sleep 20; done && echo PUBLISHED
kill $WD 2>/dev/null; echo CLOSEUP_DONE
