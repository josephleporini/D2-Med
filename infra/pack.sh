#!/usr/bin/env bash
# Build repo bundle for the volume: probeB_<commit>.tgz containing the tree and a COMMIT file.
set -e; cd "$(dirname "$0")/.."
C=$(git rev-parse --short HEAD)$(git diff --quiet || echo -dirty)
echo "$C" > COMMIT
tar czf "${1:-/tmp}/probeB_$C.tgz" --exclude .git --exclude __pycache__ --exclude tests/stub/__pycache__ .
rm COMMIT; echo "${1:-/tmp}/probeB_$C.tgz"
