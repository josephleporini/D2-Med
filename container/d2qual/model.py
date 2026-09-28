"""M3 visual perception network for Block T: the direct baseline.

One timm backbone, then a site-query head: four learned queries (LUE, RUE, LLE, RLE)
attend over the backbone's spatial tokens, and each attended vector is classified
into the four ICD classes. Output logits have shape (batch, 4 sites, 4 classes).

This is the hedge model in Development Plan v1.2 P3. The Probe B staged pipeline
(segmenter, body frame, limb-end classifier, learned decision layer) becomes a
second engine and can be ensembled at the probability level.
"""
import torch
import torch.nn as nn
import timm


class SiteQueryHead(nn.Module):
    def __init__(self, dim: int, n_sites: int = 4, n_classes: int = 4, heads: int = 8, dropout: float = 0.1):
        super().__init__()
        heads = heads if dim % heads == 0 else 1
        self.queries = nn.Parameter(torch.randn(n_sites, dim) * 0.02)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norm_kv = nn.LayerNorm(dim)
        self.norm_q = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(dim, n_classes))

    def site_vectors(self, tokens):                  # tokens (B, N, C) -> (B, 4, C)
        kv = self.norm_kv(tokens)
        q = self.queries.unsqueeze(0).expand(tokens.shape[0], -1, -1)
        a, _ = self.attn(self.norm_q(q), kv, kv)
        return a + q

    def forward(self, tokens):
        return self.mlp(self.site_vectors(tokens))  # (B, 4, 4)


class AuxHeads(nn.Module):
    """M3-13 observable intermediate outputs, one per link of the Block T chain (§6.1):
    head end (up, down), facing (front, back, edge-on) -> link L2 body frame;
    per-site visibility fraction -> link L3. Trained when labels exist (synthetic
    data carries them; DARPA data may not), masked otherwise. Logged, never scored."""
    def __init__(self, dim: int):
        super().__init__()
        self.frame = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 2 + 3))
        self.vis = nn.Linear(dim, 1)

    def forward(self, tokens, site_vec):
        f = self.frame(tokens.mean(1))
        return {"head_end": f[:, :2], "facing": f[:, 2:], "vis": self.vis(site_vec).squeeze(-1)}


class DirectSiteNet(nn.Module):
    def __init__(self, backbone: str, pretrained: bool = False, img_size: int | None = None):
        super().__init__()
        kw = {"img_size": img_size} if img_size and "vit" in backbone else {}
        self.backbone = timm.create_model(backbone, pretrained=pretrained, num_classes=0, **kw)
        self.head = SiteQueryHead(self.backbone.num_features)
        self.aux = AuxHeads(self.backbone.num_features)

    def forward_all(self, x):
        t = self.tokens(x)
        sv = self.head.site_vectors(t)
        return self.head.mlp(sv), self.aux(t, sv)

    def tokens(self, x):
        f = self.backbone.forward_features(x)
        if f.ndim == 4:                               # conv: (B, C, H, W) -> (B, HW, C)
            f = f.flatten(2).transpose(1, 2)
        return f                                      # vit: (B, N, C) incl. prefix tokens

    def forward(self, x):
        return self.head(self.tokens(x))


def build_from_config(cfg: dict, pretrained: bool = False) -> DirectSiteNet:
    return DirectSiteNet(cfg["backbone"], pretrained=pretrained, img_size=cfg.get("input_size"))
