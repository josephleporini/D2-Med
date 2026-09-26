#!/usr/bin/env bash
# GPU render of generator-vNext splits on a pod; resumable on the volume; publishes batches to DDData.
# env: SPLITS (default "train5 test5 challenge5"), JOBS (default 4), MAX_HOURS (default 4, hard stop for cost)
set -u
R=/workspace/probeB/repo_git; A=/workspace/probeB/assets/mh; BV=/workspace/probeB/venv_bpy
W=/workspace/probeB/render5; DD=/workspace/probeB/dddata_pod
DEBIAN_FRONTEND=noninteractive apt-get install -y -qq libx11-6 libxrender1 libxxf86vm1 libxfixes3 libxi6 libxkbcommon0 libsm6 libice6 >/dev/null 2>&1
export PATH=$HOME/.local/bin:$PATH; command -v uv >/dev/null || pip install -q uv
[ -x $BV/bin/python ] || { uv venv -q --python 3.11 $BV && uv pip install -q --python $BV/bin/python bpy numpy; }
[ -f $A/makehuman/data/3dobjs/base.obj ] || { echo "assets missing; run gpu_timing.sh first"; exit 3; }
ln -sfn /workspace/probeB/assets $R/assets
if [ -d $DD/.git ]; then git -C $DD remote set-url origin https://x-access-token:${GH_TOKEN}@github.com/josephleporini/DDData.git; git -C $DD pull -q --rebase origin main
else git clone -q https://x-access-token:${GH_TOKEN}@github.com/josephleporini/DDData.git $DD; fi
nvidia-smi --query-gpu=name --format=csv,noheader; nproc
export CYCLES_DEVICE=OPTIX SPLITS=${SPLITS:-train5 test5 challenge5} JOBS=${JOBS:-4}
timeout $(( ${MAX_HOURS:-4} * 3600 )) bash $R/jobs/render_splits.sh $W $DD $BV/bin/python
echo "RENDER_RC=$?"
for s in $SPLITS; do echo "COUNT $s $(ls $W/$s 2>/dev/null | grep -c _sidecar.json)"; done
