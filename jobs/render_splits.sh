#!/usr/bin/env bash
# Phase 3 render driver (workspace or pod): resumable; renders each split in batches of 40 and pushes each batch to DDData.
# usage: [SPLITS="dev5 train5"] [JOBS=4] bash jobs/render_splits.sh <work_dir> <dddata_clone> <python_with_bpy>
#   SPLITS  subset of splits to render (default all four, in order); lets two machines split the work
#   JOBS    concurrent scene4 processes (default 1)
# Each batch is rebased onto origin before push, so two machines can publish to DDData at the same time.
W=$1; DD=$2; BPY=$3; G=$(cd "$(dirname "$0")/../gen" && pwd); B=40; J=${JOBS:-1}
export GEN_COMMIT=$(git -C "$G/.." rev-parse --short HEAD) G BPY
declare -A SPEC=([dev5]="480 50000" [train5]="600 70000" [test5]="480 60000" [challenge5]="240 80000 challenge"
                 [train6]="600 90000 bt2" [dev6]="240 91000 bt2")   # BT-2 mix: gen/sample_batch4.py bt2_mix
render_one() {  # $1 = params path, $2 = out dir
  id=$(basename "$1" _params.json)
  [ -f "$2/${id}_sidecar.json" ] && return 0
  (cd "$G" && timeout 900 "$BPY" scene4.py full "$1" "$2" 2>&1 | grep -E '^DONE|Traceback' -A3 | cut -c1-120)
}
export -f render_one
for name in ${SPLITS:-dev5 train5 test5 challenge5}; do
  set -- ${SPEC[$name]}; n=$1; seed=$2; mode=${3:-}
  O=$W/$name; mkdir -p $O
  [ -f $O/_params_done ] || { "$BPY" $G/sample_batch4.py split $name $n $seed $O $mode && touch $O/_params_done; }
  ids=($(ls $O/*_params.json | xargs -n1 basename | sed 's/_params.json//' | sort))
  for ((b=0; b*B<${#ids[@]}; b++)); do
    bd=$DD/$name/batch_$(printf %03d $b); [ -f $bd/MANIFEST.sha256 ] && continue
    printf "$O/%s_params.json\n" "${ids[@]:b*B:B}" | xargs -P $J -I{} bash -c 'render_one "$1" "$2"' _ {} $O
    mkdir -p $bd; for id in "${ids[@]:b*B:B}"; do cp $O/${id}* $bd/ 2>/dev/null; done
    "$BPY" $G/../infra/guard_public.py $bd || { echo "PUBLISH_GUARD refused $bd"; exit 4; }   # G3
    (cd $bd && sha256sum $(ls | grep -v MANIFEST) > MANIFEST.sha256)
    (cd $DD && git add -A && git -c user.email=jslepo@gmail.com -c user.name="Joseph Leporini" commit -qm "$name batch $b (generator $GEN_COMMIT)" \
      && for t in 1 2 3 4 5; do git -c user.email=jslepo@gmail.com -c user.name="Joseph Leporini" pull -q --rebase origin main && timeout 300 git push -q origin main && break; sleep 30; done)
    echo "BATCH_DONE $name $b $(date -u +%H:%M:%S)"
  done
  echo "SPLIT_DONE $name $(date -u +%H:%M:%S)"
done
echo ALL_DONE
