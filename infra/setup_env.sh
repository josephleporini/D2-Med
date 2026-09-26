#!/usr/bin/env bash
# Idempotent pod setup for Probe B. Source it or run it; safe to repeat.
# Usage: bash /workspace/probeB/repo/infra/setup_env.sh [--lock]
#   --lock  also write an environment lock (python, packages, sam2 source, torch, device) to /workspace/probeB/env/
# Exports nothing; prints PY=<venv python> on the last line for the caller.
set -euo pipefail
VENV=${VENV:-/workspace/probeB/venv_pose}
PYVER=3.11.16
log() { echo "[setup $(date -u +%H:%M:%S)] $*"; }

# 1. system libraries (OpenCV, EGL for Blender headless, OpenMP)
need=""
for p in libgl1 libglib2.0-0 libegl1 libgomp1; do dpkg -s "$p" >/dev/null 2>&1 || need="$need $p"; done
if [ -n "$need" ]; then
  log "apt install$need"
  (apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq $need) >/dev/null
fi

# 2. uv and the interpreter the venv was built on (the venv symlinks into the uv python dir)
if ! "$VENV/bin/python" -c 'import sys' >/dev/null 2>&1; then
  # the venv symlinks into the uv-managed interpreter recorded in pyvenv.cfg; recreate exactly that one
  PYVER=$(sed -n 's/^version_info *= *//p;s/^version *= *//p' "$VENV/pyvenv.cfg" | head -1); PYVER=${PYVER:-3.11.16}
  log "uv python install $PYVER (home: $(sed -n 's/^home *= *//p' "$VENV/pyvenv.cfg"))"
  pip install -q -U uv >/dev/null 2>&1 || true; export PATH=$HOME/.local/bin:$PATH
  uv python install "$PYVER" >/dev/null || { log "FATAL uv python install $PYVER failed"; exit 3; }
fi
PY="$VENV/bin/python"
"$PY" -c 'import numpy, cv2, torch, sklearn' || { log "FATAL venv broken: $VENV"; exit 2; }

# 3. sam2: the J15 jobs replaced the editable install with the git build; accept either, record which
if ! "$PY" -c 'import sam2' >/dev/null 2>&1; then
  log "install sam2 (git, no CUDA ext)"
  command -v uv >/dev/null || pip install -q -U uv >/dev/null; SAM2_BUILD_CUDA=0 uv pip install --python "$PY" --no-deps git+https://github.com/facebookresearch/sam2.git \
    hydra-core iopath omegaconf antlr4-python3-runtime portalocker >/dev/null
fi

if [ "${1:-}" = "--lock" ]; then
  mkdir -p /workspace/probeB/env
  L=/workspace/probeB/env/lock_$(date -u +%Y%m%dT%H%M%SZ).txt
  {
    echo "# probeB environment lock"; echo "date_utc: $(date -u +%FT%TZ)"; echo "pod: ${RUNPOD_POD_ID:-unknown}"
    echo "venv: $VENV"; "$PY" -V
    "$PY" - <<'EOF'
import importlib.metadata as md, torch, json
try:
    d = md.distribution('sam2'); du = d.read_text('direct_url.json')
    print('sam2:', d.version, json.loads(du) if du else 'no direct_url')
except Exception as e:
    print('sam2: ?', e)
print('torch:', torch.__version__, 'cuda', torch.version.cuda, 'device', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')
EOF
    echo "## pip freeze"; uv pip freeze --python "$PY" 2>/dev/null || "$PY" -m pip freeze
  } > "$L"
  log "lock written $L"
fi
echo "PY=$PY"
