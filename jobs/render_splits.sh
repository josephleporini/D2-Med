#!/usr/bin/env bash
# Phase 3 render driver (workspace or pod): resumable; renders each split in batches of 40 and pushes each batch to DDData.
# usage: bash jobs/render_splits.sh <work_dir> <dddata_clone> <python_with_bpy>
W=$1; DD=$2; BPY=$3; G=$(cd "$(dirname "$0")/../gen" && pwd); B=40
export GEN_COMMIT=$(git -C "$G/.." rev-parse --short HEAD)
for spec in "dev5 480 50000" "train5 600 70000" "test5 480 60000" "challenge5 240 80000 challenge"; do
  set -- $spec; name=$1; n=$2; seed=$3; mode=${4:-}
  O=$W/$name; mkdir -p $O
  [ -f $O/_params_done ] || { python3 $G/sample_batch4.py split $name $n $seed $O $mode && touch $O/_params_done; }
  ids=($(ls $O/*_params.json | xargs -n1 basename | sed 's/_params.json//' | sort))
  for ((b=0; b*B<${#ids[@]}; b++)); do
    bd=$DD/$name/batch_$(printf %03d $b); [ -f $bd/MANIFEST.sha256 ] && continue
    for id in "${ids[@]:b*B:B}"; do
      [ -f $O/${id}_sidecar.json ] && continue
      (cd $G && timeout 900 $BPY scene4.py full $O/${id}_params.json $O 2>&1 | grep -E '^DONE|Traceback' -A3 | cut -c1-120)
    done
    mkdir -p $bd; for id in "${ids[@]:b*B:B}"; do cp $O/${id}* $bd/ 2>/dev/null; done
    (cd $bd && sha256sum $(ls | grep -v MANIFEST) > MANIFEST.sha256)
    (cd $DD && git add -A && git -c user.email=jslepo@gmail.com -c user.name="Joseph Leporini" commit -qm "$name batch $b (generator $GEN_COMMIT)" \
      && for t in 1 2 3; do timeout 300 git push -q origin main && break; sleep 30; done)
    echo "BATCH_DONE $name $b $(date -u +%H:%M:%S)"
  done
  echo "SPLIT_DONE $name $(date -u +%H:%M:%S)"
done
echo ALL_DONE
