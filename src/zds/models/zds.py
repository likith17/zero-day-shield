"""ZeroDayShield model: per-layer temporal encoders, cross-view alignment, fusion.

Training signals (all label-free):
  * masked-event reconstruction per layer
  * temporal-order prediction per layer (was a segment of the window swapped?)
  * cross-view InfoNCE between the host and network view of the same window
  * fused masked reconstruction: rebuild masked events of one layer with the
    other layer visible, which is what trains the fusion block

Ablation switches live in `ModelConfig`.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ModelConfig:
    host_dim: int = 0  # 0 = layer absent
    net_dim: int = 0
    d_model: int = 96
    n_layers: int = 3
    n_heads: int = 4
    fusion_layers: int = 2
    dropout: float = 0.1
    mask_ratio: float = 0.25
    temperature: float = 0.1
    temporal: bool = True  # False: no position/time information (bag of events)
    order_task: bool = True
    contrastive: bool = True
    fusion: bool = True
    w_order: float = 0.5
    w_contrast: float = 0.5
    w_fused: float = 1.0

    @property
    def paired(self) -> bool:
        return self.host_dim > 0 and self.net_dim > 0


def _encoder(d, heads, layers, dropout):
    layer = nn.TransformerEncoderLayer(d, heads, 4 * d, dropout, batch_first=True, norm_first=True)
    return nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)


class TemporalEncoder(nn.Module):
    """Transformer over the events of one layer within one window."""

    def __init__(self, in_dim: int, cfg: ModelConfig, max_len: int = 512):
        super().__init__()
        d = cfg.d_model
        self.cfg = cfg
        self.embed = nn.Linear(in_dim, d)
        self.time = nn.Sequential(nn.Linear(1, d), nn.GELU(), nn.Linear(d, d))
        self.pos = nn.Embedding(max_len, d)
        self.rate = nn.Sequential(nn.Linear(1, d), nn.GELU(), nn.Linear(d, d))  # true event count
        self.cls = nn.Parameter(torch.zeros(1, 1, d))
        self.mask_token = nn.Parameter(torch.zeros(d))
        nn.init.normal_(self.cls, std=0.02)
        nn.init.normal_(self.mask_token, std=0.02)
        self.body = _encoder(d, cfg.n_heads, cfg.n_layers, cfg.dropout)
        self.norm = nn.LayerNorm(d)
        self.recon = nn.Linear(d, in_dim)
        self.order = nn.Linear(d, 1)
        self.proj = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, d))

    def tokens(self, x, dt, hide=None):
        """Event embeddings before the transformer; `hide` marks masked events."""
        h = self.embed(x)
        if hide is not None:
            h = torch.where(hide[..., None], self.mask_token.to(h.dtype), h)
        if self.cfg.temporal:
            pos = torch.arange(x.shape[1], device=x.device)
            h = h + self.pos(pos)[None] + self.time(torch.log1p(dt.clamp(min=0))[..., None])
        return h

    def forward(self, x, dt, mask, hide=None, n=None):
        """Returns (cls [B, d], per-event states [B, L, d]). `n` is log1p of the
        window's event count before thinning, so dense windows stay distinguishable."""
        cls = self.cls.expand(len(x), -1, -1)
        if n is not None:
            cls = cls + self.rate(n[:, None])[:, None]
        h = torch.cat([cls, self.tokens(x, dt, hide)], dim=1)
        pad = torch.cat([torch.zeros_like(mask[:, :1]), ~mask], dim=1)
        h = self.norm(self.body(h, src_key_padding_mask=pad))
        return h[:, 0], h[:, 1:]


def random_hide(mask: torch.Tensor, ratio: float) -> torch.Tensor:
    """Choose events to mask; every non-empty window gets at least one."""
    hide = (torch.rand(mask.shape, device=mask.device) < ratio) & mask
    none = mask.any(1) & ~hide.any(1)
    if none.any():
        first = mask.float().argmax(1)
        hide[none, first[none]] = True
    return hide


def swap_segments(x, dt, mask):
    """For a random half of the batch, swap the two halves of the valid event
    contents (timestamps stay in place). Returns shuffled x and the 0/1 target."""
    n = mask.sum(1)
    flip = (torch.rand(len(x), device=x.device) < 0.5) & (n >= 4)
    idx = torch.arange(x.shape[1], device=x.device)[None].expand(len(x), -1)
    half = (n // 2)[:, None]
    rolled = torch.where(idx < n[:, None], (idx + half) % n.clamp(min=1)[:, None], idx)
    src = torch.where(flip[:, None], rolled, idx)
    return torch.gather(x, 1, src[..., None].expand_as(x)), flip.float()


def masked_mse(pred, target, hide):
    """Per-window mean squared error over the hidden events. [B]"""
    err = ((pred - target) ** 2).mean(-1) * hide
    return err.sum(1) / hide.sum(1).clamp(min=1)


class ZeroDayShield(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.layers = [n for n, d in (("host", cfg.host_dim), ("net", cfg.net_dim)) if d > 0]
        if not self.layers:
            raise ValueError("model needs at least one telemetry layer")
        self.enc = nn.ModuleDict({n: TemporalEncoder(getattr(cfg, f"{n}_dim"), cfg) for n in self.layers})
        self.use_fusion = cfg.fusion and cfg.paired
        self.use_contrast = cfg.contrastive and cfg.paired
        if self.use_fusion:
            d = cfg.d_model
            self.kind = nn.Embedding(2, d)
            self.fuse_cls = nn.Parameter(torch.zeros(1, 1, d))
            nn.init.normal_(self.fuse_cls, std=0.02)
            self.fuser = _encoder(d, cfg.n_heads, cfg.fusion_layers, cfg.dropout)
            self.fuse_norm = nn.LayerNorm(d)
            self.fuse_recon = nn.ModuleDict({n: nn.Linear(d, getattr(cfg, f"{n}_dim")) for n in self.layers})

    def _fuse(self, states, masks):
        parts = [states[n] + self.kind.weight[i] for i, n in enumerate(self.layers)]
        h = torch.cat([self.fuse_cls.expand(len(parts[0]), -1, -1)] + parts, dim=1)
        pad = torch.cat([torch.zeros_like(masks["host"][:, :1])] + [~masks[n] for n in self.layers], dim=1)
        h = self.fuse_norm(self.fuser(h, src_key_padding_mask=pad))
        out, at = {}, 1
        for n in self.layers:
            length = states[n].shape[1]
            out[n] = h[:, at:at + length]
            at += length
        return h[:, 0], out

    def _contrast(self, cls):
        a = F.normalize(self.enc["host"].proj(cls["host"]), dim=-1)
        b = F.normalize(self.enc["net"].proj(cls["net"]), dim=-1)
        return a, b

    def loss(self, batch: dict) -> tuple[torch.Tensor, dict]:
        """Self-supervised training loss on a batch of (benign) windows."""
        cfg = self.cfg
        parts, states, masks, hides, cls = {}, {}, {}, {}, {}
        for n in self.layers:
            x, dt, mask, cnt = batch[f"{n}_x"], batch[f"{n}_dt"], batch[f"{n}_mask"], batch.get(f"{n}_n")
            enc = self.enc[n]
            hide = random_hide(mask, cfg.mask_ratio)
            c, s = enc(x, dt, mask, hide, cnt)
            parts[f"{n}_recon"] = masked_mse(enc.recon(s), x, hide).mean()
            states[n], masks[n], hides[n], cls[n] = s, mask, hide, c
            if cfg.order_task and cfg.temporal:
                xs, target = swap_segments(x, dt, mask)
                parts[f"{n}_order"] = cfg.w_order * F.binary_cross_entropy_with_logits(
                    enc.order(enc(xs, dt, mask, None, cnt)[0]).squeeze(-1), target)
        if self.use_contrast:
            a, b = self._contrast(cls)
            logits = a @ b.T / cfg.temperature
            target = torch.arange(len(a), device=a.device)
            parts["contrast"] = cfg.w_contrast * 0.5 * (F.cross_entropy(logits, target) + F.cross_entropy(logits.T, target))
        if self.use_fusion:
            _, fused = self._fuse(states, masks)
            for n in self.layers:
                parts[f"{n}_fused"] = cfg.w_fused * masked_mse(self.fuse_recon[n](fused[n]), batch[f"{n}_x"], hides[n]).mean()
        return sum(parts.values()), {k: float(v.detach()) for k, v in parts.items()}

    @torch.no_grad()
    def describe(self, batch: dict, n_masks: int = 4, seed: int = 0) -> dict:
        """Per-window anomaly evidence and embedding. Deterministic given `seed`.

        Keys: `embed_<view>` [B, D] plus raw (uncalibrated) components, higher = stranger:
        `<layer>_recon`, `<layer>_fused`, `<layer>_order`, `mismatch`.
        """
        gen_state = torch.random.get_rng_state()
        torch.manual_seed(seed)
        out: dict[str, torch.Tensor] = {}
        clean_states, masks, cls = {}, {}, {}
        for n in self.layers:
            x, dt, mask = batch[f"{n}_x"], batch[f"{n}_dt"], batch[f"{n}_mask"]
            cls[n], clean_states[n] = self.enc[n](x, dt, mask, None, batch.get(f"{n}_n"))
            masks[n] = mask
            if self.cfg.order_task and self.cfg.temporal:
                # Logit that the untouched window looks out of order.
                out[f"{n}_order"] = self.enc[n].order(cls[n]).squeeze(-1)
        for _ in range(n_masks):
            states, hides = {}, {}
            for n in self.layers:
                x, dt, mask = batch[f"{n}_x"], batch[f"{n}_dt"], batch[f"{n}_mask"]
                hides[n] = random_hide(mask, self.cfg.mask_ratio)
                _, states[n] = self.enc[n](x, dt, mask, hides[n], batch.get(f"{n}_n"))
                err = masked_mse(self.enc[n].recon(states[n]), x, hides[n])
                out[f"{n}_recon"] = out.get(f"{n}_recon", 0) + err / n_masks
            if self.use_fusion:
                _, fused = self._fuse(states, masks)
                for n in self.layers:
                    err = masked_mse(self.fuse_recon[n](fused[n]), batch[f"{n}_x"], hides[n])
                    out[f"{n}_fused"] = out.get(f"{n}_fused", 0) + err / n_masks
        if self.use_contrast:
            a, b = self._contrast(cls)
            out["mismatch"] = 1 - (a * b).sum(-1)
        # One embedding per view; each gets its own benign-neighbour distance so a
        # single-layer anomaly is not diluted inside the fused embedding. The summary
        # token is joined with the mean event state (which is what reconstruction
        # trains) and with the event count.
        def pooled(states, mask):
            m = mask[..., None].float()
            return (states * m).sum(1) / m.sum(1).clamp(min=1)

        for n in self.layers:
            parts = [cls[n], pooled(clean_states[n], masks[n])]
            if f"{n}_n" in batch:
                parts.append(batch[f"{n}_n"][:, None])
            out[f"embed_{n}"] = torch.cat(parts, dim=-1)
        if self.use_fusion:
            f_cls, f_states = self._fuse(clean_states, masks)
            out["embed_fused"] = torch.cat([f_cls] + [pooled(f_states[n], masks[n]) for n in self.layers], dim=-1)
        torch.random.set_rng_state(gen_state)
        return out
