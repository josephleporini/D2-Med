#!/usr/bin/env bash
# APL-mirror release check (verification level L3). Run on a Linux GPU host with
# Docker + NVIDIA Container Toolkit, ideally once each on A40, L40 and H100.
#
#   harness/apl_mirror.sh <team-name> <team-email> <version> <model_dir> [holdout_images_dir]
#
# Steps: build -> size check -> ClamAV scan -> conformance suite in docker mode
#        -> timed run on holdout images with the exact APL flags -> validator
set -euo pipefail
TEAM="${1:?team name}"; EMAIL="${2:?email}"; VER="${3:?version}"; MODEL="${4:?model dir}"; HOLDOUT="${5:-}"
case "$TEAM" in *TBD*|conformance) echo "Refusing: placeholder team name"; exit 2;; esac
IMG="${TEAM}-d2-qualification-task:${VER}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

echo "== stage model into build context"
rm -rf model && mkdir -p model && cp "$MODEL"/weights.pt "$MODEL"/model_config.json model/

echo "== build $IMG"
docker build --build-arg D2_TEAM_NAME="$TEAM" --build-arg D2_TEAM_EMAIL="$EMAIL" \
             --build-arg D2_SUBMISSION_VERSION="$VER" -t "$IMG" .

echo "== image size (limit < 20 GB)"
BYTES=$(docker image inspect "$IMG" --format '{{.Size}}')
python3 -c "b=$BYTES; print(f'{b/2**30:.2f} GB'); assert b < 19*2**30, 'image too large'"

echo "== GPU visible inside container"
docker run --rm --gpus all --network none --entrypoint python "$IMG" -c \
  "import torch;print('cuda',torch.cuda.is_available(),torch.version.cuda,torch.cuda.get_device_name(0) if torch.cuda.is_available() else '-')"

echo "== ClamAV scan (ICD 3.5)"
SCAN=$(mktemp -d); docker save "$IMG" | tar -xC "$SCAN"
docker run --rm -v "$SCAN":/scandir clamav/clamav:latest clamscan -r --infected --no-summary /scandir && echo "ClamAV: clean"
rm -rf "$SCAN"

echo "== conformance suite (docker mode)"
python3 harness/conformance.py --model-dir "$ROOT/model" --docker "$IMG" --bulk 300

if [ -n "$HOLDOUT" ]; then
  echo "== timed holdout run with APL flags"
  OUT=$(mktemp -d)
  /usr/bin/time -v docker run --rm --gpus all --network none --cpus=8 --memory=32g \
      -v "$(realpath "$HOLDOUT")":/app/input:ro -v "$OUT":/app/output:rw "$IMG" 2> "$OUT.time"
  grep -E "Elapsed|Maximum resident" "$OUT.time" || true
  python3 -c "import sys;sys.path.insert(0,'.');from d2qual.formatter import validate;import json,os;from pathlib import Path
d=json.load(open('$OUT/predictions.json'));ids=sorted(f for f in os.listdir('$HOLDOUT') if Path(f).suffix.lower() in {'.jpg','.jpeg','.png'})
e=validate(d,Path('schema/qual-predictions.schema.json'),ids);print('VALID' if not e else e[:5])"
  echo "   Run APL's validate_predications.py on $OUT/predictions.json as the final check."
fi

echo "== push (manual, after all three GPU types pass)"
echo "docker tag $IMG registry.gitlab.com/<project_name>/$IMG"
echo "docker push registry.gitlab.com/<project_name>/$IMG"
echo "docker inspect --format='{{index .RepoDigests 0}}' registry.gitlab.com/<project_name>/$IMG"
