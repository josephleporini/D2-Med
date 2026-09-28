import numpy as np
import torch
from ..model import build_from_config
from ..preprocess import letterbox, to_tensor
from ..sites import LR_SWAP, FACING, HEAD_END



class DirectEngine:
    """DirectSiteNet baseline. predict() returns (probs (N,4,4), aux dict or None)."""
    name = "direct"

    def __init__(self, model_dir, cfg: dict, device: torch.device, tta_flip: bool = True):
        self.cfg, self.device, self.tta_flip = cfg, device, tta_flip
        self.size = int(cfg["input_size"])
        self.mean, self.std = cfg.get("mean", [0.485, 0.456, 0.406]), cfg.get("std", [0.229, 0.224, 0.225])
        self.net = build_from_config(cfg, pretrained=False)
        state = torch.load(model_dir / cfg.get("weights", "weights.pt"), map_location="cpu", weights_only=True)
        missing, unexpected = self.net.load_state_dict(state, strict=False)
        bad = [k for k in missing if not k.startswith("aux.")] + list(unexpected)
        if bad:
            raise RuntimeError(f"weights do not match model: {bad[:5]}")
        # Aux outputs are reported only if the weights were trained with them (M3-13).
        self.aux_trained = bool(cfg.get("aux_trained")) and not any(k.startswith("aux.") for k in missing)
        self.net.eval().to(device)
        self.amp = device.type == "cuda"

    def prepare(self, pil_image):
        return to_tensor(letterbox(pil_image, self.size), self.mean, self.std)

    @torch.inference_mode()
    def predict(self, items):
        x = torch.stack(list(items)).to(self.device, non_blocking=True)
        with torch.autocast(self.device.type, dtype=torch.float16, enabled=self.amp):
            logits, aux = self.net.forward_all(x)
            p = logits.float().softmax(-1)
            he, fa, vis = aux["head_end"].float().softmax(-1), aux["facing"].float().softmax(-1), aux["vis"].float().sigmoid()
            if self.tta_flip:
                # Mirror, predict, swap left/right sites back to the original anatomy (D-04, M3-02).
                # Head end and facing do not change under a mirror; visibility swaps like the sites.
                lf, af = self.net.forward_all(torch.flip(x, dims=[3]))
                p = 0.5 * (p + lf.float().softmax(-1)[:, LR_SWAP, :])
                he = 0.5 * (he + af["head_end"].float().softmax(-1))
                fa = 0.5 * (fa + af["facing"].float().softmax(-1))
                vis = 0.5 * (vis + af["vis"].float().sigmoid()[:, LR_SWAP])
        aux_out = None
        if self.aux_trained:
            aux_out = {"head_end": he.cpu().numpy(), "facing": fa.cpu().numpy(), "vis": vis.cpu().numpy()}
        return p.cpu().numpy(), aux_out
