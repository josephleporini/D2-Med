#!/usr/bin/env bash
# Publish pending pod results (L1 ledger, TTA mirror, GPU timing) from the volume to DDData. No compute.
set -e
P=/workspace/probeB; F=$P/out/tta_flip; G=$P/out/gpu_timing
cd /tmp && rm -rf dd && git clone -q --depth 1 https://x-access-token:${GH_TOKEN}@github.com/josephleporini/DDData.git dd && cd dd
mkdir -p results/l1 results/tta_flip_dev3 results/gpu_timing
cp -r $P/out/l1/orig $P/out/l1/adopted results/l1/ 2>/dev/null || true
cp $F/tta_dev3.json $F/log_f0.txt results/tta_flip_dev3/ 2>/dev/null || true
cp $G/gpu/*_sidecar.json results/gpu_timing/ 2>/dev/null || true
mkdir -p results/gpu_timing/cpu && cp $G/cpu/*_sidecar.json results/gpu_timing/cpu/ 2>/dev/null || true
cp $P/logs/gpu_timing.log $P/logs/tta_flip.log results/ 2>/dev/null || true
du -sh results
git add -A && git -c user.email=jslepo@gmail.com -c user.name="Joseph Leporini (pod)" commit -qm "Pod results: L1 ledgers, test-time mirror dev3, GPU render timing" && git push -q origin main && echo PUBLISHED
