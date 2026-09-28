"""Write schema v0.2.0 casualty events from a scorer ledger, validate them with the shared validator, and check the
round trip events -> qualification classes against the ledger's own predictions (G-O1 evidence for the image thread).

usage: python tools/ledger_to_events.py LEDGER.jsonl OUT_DIR [--validator D2_DEV/d2-voice] [--model-id probeB-adopted]
       [--pred EXTRACTION.jsonl]   optional extraction records carrying a predicted 'frame' (facing, facing_conf)

Uses only model outputs from the ledger ('p', the 4-class posterior, and 'pred'). Ledger 'truth' fields are generator
truth and are never written into events. Writes OUT_DIR/events.jsonl (one event log per scene) and
OUT_DIR/events_summary.json. Exits non-zero on any validation error, round-trip mismatch or empty input.
"""
import os, sys, json, argparse, glob, hashlib
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'gen'))
import events as E


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('ledger'); ap.add_argument('out')
    ap.add_argument('--validator', default=os.path.join(HERE, '..', '..', 'd2_dev', 'd2-voice'))
    ap.add_argument('--model-id', default='probeB-adopted'); ap.add_argument('--pred', default='')
    A = ap.parse_args()
    sys.path.insert(0, A.validator)
    from d2voice import validate_log as V

    rows = [json.loads(l) for l in open(A.ledger)]
    if not rows:
        sys.exit('ERROR: ledger has zero rows')
    met = os.path.join(os.path.dirname(A.ledger), 'metrics.json')
    commit = json.load(open(met)).get('commit') if os.path.exists(met) else None
    frames = {}
    for f in (glob.glob(A.pred) if A.pred else []):
        for l in open(f):
            r = json.loads(l)
            if r.get('frame'):
                frames[r['scene']] = r['frame']
    scenes = {}
    for r in rows:
        scenes.setdefault(r['scene'], {})[r['site']] = r
    os.makedirs(A.out, exist_ok=True)
    n_err, n_mis, n_sites, bad = 0, 0, 0, []
    counts = {}
    with open(os.path.join(A.out, 'events.jsonl'), 'w') as fh:
        for sid in sorted(scenes):
            S = scenes[sid]
            if set(S) != set(E.SITES):
                n_err += 1; bad.append(f'{sid}: sites {sorted(S)}'); continue
            probs = {s: S[s]['p'] for s in E.SITES}
            tot = {s: sum(p) for s, p in probs.items()}          # ledger rounds to 3 decimals; renormalise
            probs = {s: [v / tot[s] for v in p] for s, p in probs.items()}
            fr = frames.get(sid, {})
            log = E.image_log(f'{sid}.jpg', probs, model_version=f"{A.model_id}@{commit or 'unknown'}", model_id=A.model_id,
                              facing=fr.get('facing') if fr.get('facing') != 'unknown' else None,
                              facing_conf=fr.get('facing_conf'), dataset='team_synthetic', scenario_id=sid)
            errs = V.check(log)
            if errs:
                n_err += 1; bad.append(f'{sid}: {errs[:2]}')
            q = E.to_qual(log)
            for s in E.SITES:
                n_sites += 1
                if q[s] != S[s]['pred']:                     # argmax ties after rounding would show up here
                    n_mis += 1; bad.append(f'{sid} {s}: events {q[s]} vs ledger {S[s]["pred"]} p={S[s]["p"]}')
            for e in log['events']:
                counts[e['type']] = counts.get(e['type'], 0) + 1
            fh.write(json.dumps(log, separators=(',', ':')) + '\n')
    h = hashlib.sha256(open(os.path.join(A.out, 'events.jsonl'), 'rb').read()).hexdigest()
    summ = {'ledger': A.ledger, 'commit': commit, 'schema_version': E.SCHEMA_VERSION, 'scenes': len(scenes),
            'sites': n_sites, 'event_counts': counts, 'logs_invalid': n_err, 'round_trip_mismatches': n_mis,
            'facing_from_model': len(frames), 'events_sha256': h, 'problems': bad[:20]}
    json.dump(summ, open(os.path.join(A.out, 'events_summary.json'), 'w'), indent=1)
    print('EVENTS', json.dumps({k: v for k, v in summ.items() if k != 'problems'}))
    for b in bad[:20]:
        print('  ', b)
    sys.exit(1 if (n_err or n_mis or not n_sites) else 0)


if __name__ == '__main__':
    main()
