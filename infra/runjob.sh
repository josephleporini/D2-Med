#!/usr/bin/env bash
# Standard job wrapper for Probe B pods.
# Usage (as the pod command, after the repo is on the volume):
#   bash /workspace/probeB/repo/infra/runjob.sh <job_name> <command...>
# Behaviour:
#   - runs setup_env.sh, then the command unbuffered (PYTHONUNBUFFERED, stdbuf) from the repo root
#   - tees all output to /workspace/probeB/logs/<job>.log and to the pod log
#   - heartbeat line every 60 s (elapsed, load, last output line) so a silent job is visible
#   - OUT=/workspace/probeB/out/<job> is exported; jobs must write results there (volume, not container disk)
#   - writes OUT/_job.json (commit, command, start, end, exit code, pod)
#   - on exit: KEEP_ALIVE=1 keeps the pod up; otherwise it terminates itself (runpodctl remove pod) so billing ends
set -uo pipefail
JOB=$1; shift
REPO=${REPO:-/workspace/probeB/repo}; export REPO
export OUT=/workspace/probeB/out/$JOB PYTHONUNBUFFERED=1
mkdir -p "$OUT" /workspace/probeB/logs
LOG=/workspace/probeB/logs/$JOB.log
exec > >(tee -a "$LOG") 2>&1
COMMIT=$(cat "$REPO/COMMIT" 2>/dev/null || echo unknown)
START=$(date -u +%FT%TZ)
echo "[job $JOB] start $START commit $COMMIT pod ${RUNPOD_POD_ID:-?} cmd: $*"
SETUP=$(bash "$REPO/infra/setup_env.sh" ${LOCK:+--lock}); SRC=$?; echo "$SETUP"
PY=$(echo "$SETUP" | sed -n 's/^PY=//p' | tail -1); export PY
if [ $SRC -ne 0 ] || [ -z "$PY" ]; then echo "[job $JOB] SETUP_FAILED rc=$SRC"; RC_SETUP=1; fi
( while sleep 60; do echo "[hb $JOB] $(date -u +%H:%M:%S) load $(cut -d' ' -f1 /proc/loadavg) last: $(tail -c 300 "$LOG" | tr '\n' ' ' | tail -c 120)"; done ) &
HB=$!
cd "$REPO"
if [ -z "${RC_SETUP:-}" ]; then stdbuf -oL -eL "$@"; RC=$?; else RC=90; fi
kill $HB 2>/dev/null
END=$(date -u +%FT%TZ)
printf '{"job":"%s","commit":"%s","cmd":"%s","start":"%s","end":"%s","rc":%d,"pod":"%s"}\n' \
  "$JOB" "$COMMIT" "$(echo "$*" | sed 's/"/\\"/g')" "$START" "$END" "$RC" "${RUNPOD_POD_ID:-}" > "$OUT/_job.json"
echo "[job $JOB] JOB_DONE rc=$RC end $END"
if [ "${KEEP_ALIVE:-0}" != "1" ]; then
  sleep 20   # let the log stream flush
  if command -v runpodctl >/dev/null 2>&1 && [ -n "${RUNPOD_POD_ID:-}" ]; then
    # pods with a network volume cannot be stopped, only terminated; outputs are already on the volume
    runpodctl remove pod "$RUNPOD_POD_ID" || runpodctl stop pod "$RUNPOD_POD_ID" || echo "[job $JOB] self-terminate failed; terminate from the console"
  else
    echo "[job $JOB] runpodctl unavailable; terminate from the console"
  fi
fi
sleep infinity
