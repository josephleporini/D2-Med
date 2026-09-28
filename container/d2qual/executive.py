"""M13 Executive: entrypoint, budgets, watchdog, fallback, health log.

Fail-safe sequence (D-09):
  1. list inputs, then immediately write a complete, valid predictions.json of
     fallback classes, so a valid file exists from second one onward;
  2. load the model; if that fails, keep the fallback file and exit 0;
  3. process in batches, checkpointing the merged file every N batches;
  4. a watchdog thread writes the merged file and exits 0 if the time budget
     expires (hung GPU, pathological input);
  5. final write, self-validation, one-line JSON summary on stdout.
Exit code is 0 whenever a valid predictions.json exists.
"""
import json, os, sys, threading, time, traceback
from concurrent.futures import ThreadPoolExecutor
import numpy as np

from . import __version__
from .config import RunConfig, load_model_config
from .ingest import list_images, decode
from .privacy import check_model_config
from .decision import decide
from .sites import CLASSES
from . import formatter
from .sites import FACING, HEAD_END


def log(msg, **kw):
    print(json.dumps({"t": round(time.time(), 3), "msg": msg, **kw}), file=sys.stderr, flush=True)


class State:
    def __init__(self, ids, fallback_idx):
        self.lock = threading.Lock()
        self.ids = ids
        self.classes = {i: [fallback_idx] * 4 for i in ids}
        self.done = set()
        self.fallback_used = {}
        self.probs = {}
        self.finished = False


def _device(cfg):
    import torch
    if cfg.device != "auto":
        return torch.device(cfg.device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _cap_gpu(device, cap_gb):
    import torch
    if device.type != "cuda":
        return None
    idx = device.index if device.index is not None else torch.cuda.current_device()   # 'cuda' without an index is rejected
    total = torch.cuda.get_device_properties(idx).total_memory / 2**30
    frac = min(1.0, cap_gb / total)
    torch.cuda.set_per_process_memory_fraction(frac, idx)
    return round(total, 1)


def main():
    t0 = time.time()
    cfg = RunConfig()
    os.environ.setdefault("HF_HUB_OFFLINE", "1")          # container has no network; never try
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    mcfg = load_model_config(cfg.model_dir)
    fallback_name = mcfg.get("fallback_class", "no_injury")
    fb = CLASSES.index(fallback_name)

    images, ignored = list_images(cfg.input_dir)
    ids = [p.name for p in images]
    log("inputs", images=len(ids), ignored=len(ignored), ignored_examples=ignored[:5], version=__version__)
    st = State(ids, fb)

    def flush(tag):
        with st.lock:
            doc = formatter.build(ids, st.classes, cfg.team_name, cfg.team_email, cfg.version)
        errs = formatter.validate(doc, cfg.schema_path, ids)
        if errs:
            log("self_validation_failed", tag=tag, errors=errs[:5])
            doc = formatter.build(ids, {i: [fb] * 4 for i in ids}, cfg.team_name, cfg.team_email, cfg.version)
        formatter.write(doc, cfg.output_file)
        return not errs

    flush("initial_fallback")
    log("initial_fallback_written", elapsed_s=round(time.time() - t0, 3))   # M13-01

    def summary(status, **kw):
        s = {"status": status, "images": len(ids), "predicted": len(st.done),
             "fallback_images": len(st.fallback_used), "fallback_reasons": _count(st.fallback_used.values()),
             "elapsed_s": round(time.time() - t0, 2), **kw}
        print(json.dumps(s), flush=True)

    def watchdog():
        deadline = t0 + cfg.budget_s
        while not st.finished:
            if time.time() > deadline:
                with st.lock:
                    for i in ids:
                        if i not in st.done:
                            st.fallback_used.setdefault(i, "time_budget")
                flush("watchdog")
                summary("watchdog_timeout")
                os._exit(0)
            time.sleep(1.0)

    threading.Thread(target=watchdog, daemon=True).start()

    if not ids:
        st.finished = True
        summary("no_images")
        return 0

    try:
        check_model_config(mcfg)
        if not mcfg:
            raise FileNotFoundError(f"no model_config.json in {cfg.model_dir}")
        import torch
        torch.backends.cudnn.benchmark = False
        torch.manual_seed(0)
        device = _device(cfg)
        total_gb = _cap_gpu(device, cfg.gpu_cap_gb)
        from .engines import build as build_engines
        engine = build_engines(cfg.model_dir, mcfg, device, tta_flip=cfg.tta_flip)
        log("model_loaded", device=str(device), gpu_total_gb=total_gb, gpu_cap_gb=cfg.gpu_cap_gb,
            engines=engine.name, backbone=mcfg.get("backbone"), load_s=round(time.time() - t0, 2))
    except Exception as ex:
        log("model_load_failed", error=f"{type(ex).__name__}: {ex}")
        with st.lock:
            for i in ids:
                st.fallback_used[i] = "model_unavailable"
        st.finished = True
        flush("model_unavailable")
        summary("fallback_only")
        return 0

    bias = mcfg.get("decision_bias")
    # Item 3 switch, default OFF: D2_PRIOR_EM=1 and a training class mix in model_config ("train_prior")
    prior_em = mcfg.get("train_prior") if os.environ.get("D2_PRIOR_EM", "0") == "1" else None
    log("prior_em_switch", on=bool(prior_em))
    t_inf = time.time()
    batches = [images[k:k + cfg.batch_size] for k in range(0, len(images), cfg.batch_size)]
    pool = ThreadPoolExecutor(max_workers=max(1, cfg.decode_workers))

    def load(p):
        im, flags = decode(p, engine.size)
        if im is None:
            return p.name, None, flags
        try:
            return p.name, engine.prepare(im), flags
        except Exception as ex:
            return p.name, None, flags + [f"prepare_error:{type(ex).__name__}"]

    # Bounded prefetch: decode at most 3 batches ahead of the GPU so host RAM stays
    # flat regardless of test-set size (32 GB limit).
    from collections import deque
    queue, it = deque(), iter(batches)

    def submit_next():
        b = next(it, None)
        if b is not None:
            queue.append([pool.submit(load, p) for p in b])

    for _ in range(3):
        submit_next()
    bi = -1
    last_ckpt, n_ckpt = time.time(), 0
    while queue:
        futs = queue.popleft()
        submit_next()
        bi += 1
        items = [f.result() for f in futs]
        try:
            ok = [(n, x) for n, x, _ in items if x is not None]
            for n, x, flags in items:
                if x is None:
                    st.fallback_used[n] = flags[-1] if flags else "decode_error"
            if ok:
                for e_, _w in engine.members:            # image ids for engines that log per-image rows (dev runs)
                    e_.current_ids = [n for n, _ in ok]
                probs, aux = engine.predict([x for _, x in ok])
                if not np.all(np.isfinite(probs)):
                    raise FloatingPointError("non-finite probabilities")
                cls = decide(probs, bias)
                if prior_em:
                    with st.lock:
                        for (n, _), pr in zip(ok, probs):
                            st.probs[n] = pr
                with st.lock:
                    for (n, _), c in zip(ok, cls):
                        st.classes[n] = [int(v) for v in c]
                        st.done.add(n)
                if aux is not None:                     # M3-13: intermediate results to the run log
                    for j, (n, _) in enumerate(ok):
                        log("aux", image_id=n,
                            head_end=HEAD_END[int(aux["head_end"][j].argmax())], head_end_p=round(float(aux["head_end"][j].max()), 3),
                            facing=FACING[int(aux["facing"][j].argmax())], facing_p=round(float(aux["facing"][j].max()), 3),
                            vis=[round(float(v), 3) for v in aux["vis"][j]])
        except Exception as ex:
            log("batch_failed", batch=bi, error=f"{type(ex).__name__}: {ex}", trace=traceback.format_exc(limit=2))
            for n, _, _ in items:
                st.fallback_used.setdefault(n, "batch_error")
        if time.time() - last_ckpt >= cfg.checkpoint_s:          # M13-04
            flush("checkpoint"); last_ckpt = time.time(); n_ckpt += 1
    pool.shutdown()
    st.finished = True
    if prior_em and st.probs:                    # Item 3: re-decide every predicted image under the estimated test mix
        from .prior import em_prior
        ids_p = sorted(st.probs)
        P = np.stack([st.probs[i] for i in ids_p])           # (n, 4 sites, 4 classes)
        A, pi = em_prior(P.reshape(-1, 4), prior_em, iters=int(os.environ.get("D2_PRIOR_EM_ITERS", "20")))
        cls = decide(A.reshape(P.shape), bias)
        with st.lock:
            for i, c in zip(ids_p, cls):
                st.classes[i] = [int(v) for v in c]
        log("prior_em", train_prior=[round(float(v), 3) for v in prior_em], test_prior=[round(float(v), 3) for v in pi])
    valid = flush("final")
    n_ok = len(st.done)
    per_img = (time.time() - t_inf) / max(1, n_ok)
    extra = {"per_image_s": round(per_img, 4), "checkpoints": n_ckpt,
             "projected_images_in_1800s": int(1800 / per_img) if per_img > 0 else None,
             "self_valid": valid}
    if device.type == "cuda":
        extra["gpu_peak_gb"] = round(torch.cuda.max_memory_allocated(device) / 2**30, 2)
    summary("ok", **extra)
    return 0


def _count(vals):
    out = {}
    for v in vals:
        k = v.split(":")[0]
        out[k] = out.get(k, 0) + 1
    return out
