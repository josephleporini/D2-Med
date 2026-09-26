#!/usr/bin/env bash
# GPU render timing for generator vNext: 10 pilot scenes on OptiX and 2 on CPU; results published to DDData.
set -u
R=/workspace/probeB/repo_git; A=/workspace/probeB/assets/mh; BV=/workspace/probeB/venv_bpy
DEBIAN_FRONTEND=noninteractive apt-get install -y -qq libx11-6 libxrender1 libxxf86vm1 libxfixes3 libxi6 libxkbcommon0 libsm6 libice6 >/dev/null 2>&1
export PATH=$HOME/.local/bin:$PATH; command -v uv >/dev/null || pip install -q uv
[ -x $BV/bin/python ] || { uv venv -q --python 3.11 $BV && uv pip install -q --python $BV/bin/python bpy numpy; }
if [ ! -f $A/makehuman/data/3dobjs/base.obj ]; then
  mkdir -p $A && cd $A && git clone -q --depth 1 --filter=blob:none --sparse https://github.com/makehumancommunity/makehuman.git tmp && cd tmp \
   && git sparse-checkout set makehuman/data/3dobjs makehuman/data/rigs && mv makehuman $A/ && cd $A && rm -rf tmp
fi
ln -sfn /workspace/probeB/assets $R/assets; nvidia-smi --query-gpu=name --format=csv,noheader; nproc
cd $R/gen; export GEN_COMMIT=$(git -C $R rev-parse --short HEAD)
$BV/bin/python sample_batch4.py pilot $OUT/gpu >/dev/null; mkdir -p $OUT/cpu; cp $OUT/gpu/P0[12]a_params.json $OUT/cpu/
for f in $(ls $OUT/gpu/P*_params.json | head -10); do CYCLES_DEVICE=OPTIX timeout 900 $BV/bin/python scene4.py full $f $OUT/gpu 2>&1 | grep -E '^DONE|Traceback|Error' -A3; done
for f in $OUT/cpu/P*_params.json; do timeout 900 $BV/bin/python scene4.py full $f $OUT/cpu 2>&1 | grep -E '^DONE|Traceback' -A3; done
python3 - <<PY
import json,glob,os,statistics as st
o=os.environ['OUT']
for d in ('gpu','cpu'):
    t=[json.load(open(f))['provenance']['time_s'] for f in glob.glob(f'{o}/{d}/*_sidecar.json')]
    if t: print('TIMING',d,len(t),'rgb median',st.median(x['rgb'] for x in t),'total median',st.median(sum(x.values()) for x in t))
PY
cd /tmp && rm -rf dd && git clone -q --depth 1 https://x-access-token:${GH_TOKEN}@github.com/josephleporini/DDData.git dd && cd dd && mkdir -p results/gpu_timing \
 && cp $OUT/gpu/*_sidecar.json results/gpu_timing/ && git add -A && git -c user.email=jslepo@gmail.com -c user.name="Joseph Leporini (pod)" commit -qm "GPU render timing sidecars" && git push -q origin main && echo PUBLISHED
