#!/usr/bin/env bash
# L1: repo install check, split manifests, gt-keypoint coverage, first ledgers (original segmenter and adopted model, dev3)
set -u
P=/workspace/probeB; R=$P/repo
echo "== code drift: repo gen/ vs $P/gen"
for f in $R/gen/*.py $R/gen/*.sh; do b=$(basename $f); if [ -f $P/gen/$b ]; then [ "$(md5sum < $f)" = "$(md5sum < $P/gen/$b)" ] || echo "DRIFT $b"; else echo "NOT_ON_VOLUME $b"; fi; done
for f in $P/gen/*.py; do [ -f $R/gen/$(basename $f) ] || echo "ONLY_ON_VOLUME $(basename $f)"; done
echo "== layout"; ls $P/out | head -40; ls $P/out/checks | head; ls $P/out/s15 | head -20
mkdir -p $P/manifests
for sp in dev3 test4; do
  d=$P/out/$sp; [ -d $d ] || { echo "NO_SPLIT $sp"; continue; }
  $PY $R/score/manifest.py write $d $P/manifests/$sp.sha256 --seeds
  echo "GTKP $sp $(ls $d/*_gtkp.json 2>/dev/null | wc -l) of $(ls $d/C*_sidecar.json | wc -l)"
done
# original-segmenter arm as an extraction-style file (checks.py pred_sites_ceil = original map, rendered side)
$PY - <<'PYEOF'
import json, glob, os
o = open(os.environ['OUT'] + '/orig_dev3.jsonl', 'w'); n = 0
for f in sorted(glob.glob('/workspace/probeB/out/checks/chk_*.jsonl')):
    for l in open(f):
        r = json.loads(l)
        if r.get('split') == 'dev3' and r.get('pred_sites_ceil'):
            car = json.load(open(f"/workspace/probeB/out/dev3/{r['scene']}_sidecar.json"))
            o.write(json.dumps(dict(scene=r['scene'], split='dev3', labels=car['labels_by_threshold'], sites=r['pred_sites_ceil'], kp=None)) + '\n'); n += 1
print('ORIG_ROWS', n)
PYEOF
SC="$PY $R/score/score.py --gen $R/gen --sidecars $P/out/dev3 --truth $P/out/checks/chk_*.jsonl --manifest $P/manifests/dev3.sha256 --images $P/out/dev3 --maps $P/out/s15/maps"
$SC --phase L1_orig_seg_trueside --pred "$OUT/orig_dev3.jsonl" --out $OUT/orig
$SC --phase L1_adopted_gated --pred "$P/out/s15/dev3_t*_sidec.jsonl" --side-truth "$P/out/s15/dev3_t*_ceil.jsonl" --prev $OUT/orig/ledger.jsonl --out $OUT/adopted
for a in orig adopted; do echo "==== RETRO $a"; cat $OUT/$a/retro.md | head -120; echo "==== METRICS $a"; cat $OUT/$a/metrics.json; ls -la $OUT/$a; done
