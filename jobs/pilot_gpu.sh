#!/usr/bin/env bash
# vNext pilot on a GPU pod: 20 paired scenes on GPU (OptiX), 4 on CPU, pilot checks, timing and GPU-vs-CPU image diff
set -u
R=/workspace/probeB/repo; A=/workspace/probeB/assets/mh; BV=/workspace/probeB/venv_bpy
DEBIAN_FRONTEND=noninteractive apt-get install -y -qq libx11-6 libxrender1 libxxf86vm1 libxfixes3 libxi6 libxkbcommon0 libsm6 libice6 >/dev/null 2>&1
export PATH=$HOME/.local/bin:$PATH
[ -x $BV/bin/python ] || { uv venv -q --python 3.11 $BV && uv pip install -q --python $BV/bin/python bpy numpy; }
$BV/bin/python -c "import bpy;print('BPY',bpy.app.version_string)"
if [ ! -f $A/makehuman/data/3dobjs/base.obj ]; then
  mkdir -p $A && cd $A && git clone -q --depth 1 --filter=blob:none --sparse https://github.com/makehumancommunity/makehuman.git tmp && cd tmp \
   && git sparse-checkout set makehuman/data/3dobjs makehuman/data/rigs && mv makehuman $A/ && cd $A && rm -rf tmp
fi
ln -sfn /workspace/probeB/assets $R/assets; md5sum $A/makehuman/data/3dobjs/base.obj $A/makehuman/data/rigs/default.mhskel
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader; nproc
cd $R/gen; export GEN_COMMIT=$(cat $R/COMMIT)
$BV/bin/python sample_batch4.py pilot $OUT/gpu >/dev/null; mkdir -p $OUT/cpu; cp $OUT/gpu/P01a_params.json $OUT/gpu/P04a_params.json $OUT/gpu/P06b_params.json $OUT/gpu/P10b_params.json $OUT/cpu/
t0=$(date +%s)
for f in $OUT/gpu/P*_params.json; do CYCLES_DEVICE=OPTIX timeout 900 $BV/bin/python scene4.py full $f $OUT/gpu 2>&1 | grep -E '^DONE|Traceback|Error' -A3; done
echo GPU_WALL $(( $(date +%s)-t0 ))
t0=$(date +%s)
for f in $OUT/cpu/P*_params.json; do timeout 900 $BV/bin/python scene4.py full $f $OUT/cpu 2>&1 | grep -E '^DONE|Traceback|Error' -A3; done
echo CPU_WALL $(( $(date +%s)-t0 ))
cd $R; $PY score/pilot_check.py $OUT/gpu "not run on pod (local regression passed)"
$PY - <<PYEOF
import cv2, json, os
o=os.environ['OUT']
for s in ['P01a','P04a','P06b','P10b']:
    a=cv2.imread(f'{o}/gpu/{s}.jpg').astype(int); b=cv2.imread(f'{o}/cpu/{s}.jpg').astype(int)
    ia=cv2.imread(f'{o}/gpu/{s}_id.png'); ib=cv2.imread(f'{o}/cpu/{s}_id.png')
    ta=json.load(open(f'{o}/gpu/{s}_sidecar.json'))['provenance']['time_s']; tb=json.load(open(f'{o}/cpu/{s}_sidecar.json'))['provenance']['time_s']
    print('GPUvCPU', s, 'rgb mean abs diff', round(abs(a-b).mean(),2), 'id px differ', int((ia!=ib).any(axis=2).sum()), 'gpu', ta, 'cpu', tb)
PYEOF
du -sh $OUT/gpu
