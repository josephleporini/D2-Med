#!/usr/bin/env bash
# Blanket relief oracle, stage 1: rebuild the 72 dev5 blanket scenes and ray-cast depth, normal, hit class and height
# (gen/relief_pass.py), check alignment against the rendered occ.png, publish the npz files to DDData
# results/relief_oracle/. CPU only; analysis runs offline (tools/relief_oracle.py). Run under infra/runjob.sh.
# Env: JOBS (parallel scenes, default nproc), GH_TOKEN
set -u
P=/workspace/probeB; R=${REPO:?}; BV=$P/venv_bpy; J=${JOBS:-$(nproc)}
fail() { echo "RELIEF_FAIL $*"; exit 5; }
DEBIAN_FRONTEND=noninteractive apt-get install -y -qq libx11-6 libxrender1 libxxf86vm1 libxfixes3 libxi6 libxkbcommon0 libsm6 libice6 libegl1 >/dev/null 2>&1
export PATH=$HOME/.local/bin:$PATH; command -v uv >/dev/null || pip install -q uv
[ -x $BV/bin/python ] || fail "bpy venv missing at $BV"
$BV/bin/python -c "import bpy, PIL" 2>/dev/null || uv pip install -q --python $BV/bin/python pillow || fail "pillow"
$BV/bin/python -c "import bpy; print('bpy', bpy.app.version_string)"
ln -sfn $P/assets $R/assets; [ -f $R/assets/mh/makehuman/data/3dobjs/base.obj ] || fail "assets missing"
DD=$P/dddata_relief; rm -rf $DD
git clone -q --filter=blob:none --sparse https://x-access-token:${GH_TOKEN}@github.com/josephleporini/DDData.git $DD || fail clone
git -C $DD config user.email jslepo@gmail.com; git -C $DD config user.name "Joseph Leporini (pod)"
git -C $DD sparse-checkout set dev5 || fail sparse
IN=$OUT/in; mkdir -p $IN $OUT/npz
for f in $DD/dev5/batch_*/D*_params.json; do
  grep -q '"occluder": "blanket"' $f && { b=$(dirname $f); id=$(basename $f _params.json); cp $f $b/${id}_occ.png $IN/; }
done
N=$(ls $IN/*_params.json | wc -l); echo "BLANKET_SCENES $N"; [ $N -ge 60 ] || fail "expected about 72 blanket scenes"
export BV R OUT
one() { id=$(basename $1 _params.json); cd $R/gen && timeout 900 $BV/bin/python relief_pass.py $1 $OUT/in $OUT/npz 2>&1 | grep -E '^RELIEF|Traceback|Error' | head -3; }
export -f one
ls $IN/*_params.json | xargs -P $J -I{} bash -c 'one {}' | tee $OUT/relief.log
n=$(ls $OUT/npz/*_relief.npz | wc -l); echo "NPZ $n"; [ $n -ge $((N - 2)) ] || fail "relief pass incomplete ($n of $N)"
$BV/bin/python - $OUT/relief.log <<'EOF'
import json, sys
R = [json.loads(l.split('RELIEF ', 1)[1]) for l in open(sys.argv[1]) if l.startswith('RELIEF ')]
io = sorted(r['iou_occluder'] for r in R if r.get('iou_occluder') is not None)
fl = sorted(r['iou_floor'] for r in R if r.get('iou_floor') is not None)
print('ALIGN occluder IoU min / median', io[0], io[len(io) // 2], '| floor IoU min / median', fl[0], fl[len(fl) // 2], '| s/scene median', sorted(r['s'] for r in R)[len(R) // 2])
print('ALIGN_OK' if io[0] > 0.95 else 'ALIGN_WEAK')
EOF
mkdir -p $DD/results/relief_oracle && cp $OUT/npz/*_relief.npz $OUT/relief.log $DD/results/relief_oracle/ && cd $DD \
  && git add -A results/relief_oracle && git commit -qm "Relief oracle stage 1: ray-cast depth, normal, height for $n dev5 blanket scenes (D2-Med $(cat $R/COMMIT))" \
  && for t in 1 2 3; do git pull -q --rebase origin main && git push -q origin main && break; sleep 20; done && echo PUBLISHED
echo RELIEF_DONE
