#!/usr/bin/env python3
"""MMAHH-Mamba: single-file executable research implementation.

Commands: demo, smoke, prepare, splits, train, evaluate, predict, config.
This release is based on the manuscript design, not the code or weights that
produced its reported experiments. Read README.md for the precise scope.
"""

import sys
import copy
import subprocess
import torch
import torch.nn.functional as F
from torch import nn
from dataclasses import dataclass
import json
from pathlib import Path
import numpy as np
from torch.utils.data import Dataset
from itertools import product
from scipy.ndimage import (
    binary_erosion,
    distance_transform_edt,
    generate_binary_structure,
)
import random
import argparse
import os
import math
from dataclasses import asdict
from torch.utils.data import DataLoader
import csv


# ============================================================================
# GRAPH
# ============================================================================


OFFSETS = (
    (0, 0, 0),
    (1, 0, 0),
    (-1, 0, 0),
    (0, 1, 0),
    (0, -1, 0),
    (0, 0, 1),
    (0, 0, -1),
)
INVERSE = (0, 2, 1, 4, 3, 6, 5)


def topology(shape, device):
    axes = [torch.arange(s, device=device) for s in shape]
    coords = torch.stack(torch.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 3)
    neighbors = coords[:, None] + coords.new_tensor(OFFSETS)[None]
    valid = ((neighbors >= 0) & (neighbors < coords.new_tensor(shape))).all(-1)
    bounded = torch.minimum(neighbors.clamp_min(0), coords.new_tensor(shape) - 1)
    index = (bounded[..., 0] * shape[1] + bounded[..., 1]) * shape[2] + bounded[..., 2]
    return coords.float(), index.long(), valid


def masked_softmax(logits, mask, dim=-1):
    """All-masked neighborhoods return zeros rather than NaNs."""
    mask = mask.to(torch.bool)
    scores = logits.float().masked_fill(~mask, -1e30)
    result = scores.softmax(dim) * mask
    return (result / result.sum(dim, keepdim=True).clamp_min(1e-8)).to(logits.dtype)


def gather(x, index):
    """B,N,C -> B,E,K,C; E=N (one prior anchor per spatial token)."""
    return x[:, index]


def incoming(outgoing, index, valid):
    """Transpose edge->node incidence into node->candidate-edge sparse slots."""
    inv = torch.tensor(INVERSE, device=index.device)
    return outgoing[:, index, inv[None, :]] * valid[None]


def region_pool(x, outgoing, index):
    return (gather(x, index) * outgoing[..., None]).sum(2)


def scatter_regions(regions, outgoing, index):
    b, n, c = regions.shape
    source = (regions[:, :, None, :] * outgoing[..., None]).reshape(b, -1, c)
    target = index.reshape(1, -1, 1).expand(b, -1, c)
    return torch.zeros_like(regions).scatter_add(1, target, source)


def shared_membership(h_in, omega, index, previous):
    """Weighted hyperedge intersection between each token and its predecessor."""
    left = index[:, :, None]
    right = index[previous, None, :]
    equal = (left == right).to(h_in.dtype)
    weights = h_in * omega[:, index]
    value = (weights[..., None] * h_in[:, previous, None, :] * equal[None]).sum(
        (-1, -2)
    )
    return value


def rope3d(x, coords):
    """Three independent rotary coordinate groups; head width is divisible by 6."""
    b, n, heads, width = x.shape
    pairs = width // 6
    freq = 10000.0 ** (-torch.arange(pairs, device=x.device).float() / max(pairs, 1))
    angle = coords[:, :, None] * freq[None, None, :]
    angle = angle[None, :, None].to(x.dtype)
    y = x.reshape(b, n, heads, 3, pairs, 2)
    first, second = y[..., 0], y[..., 1]
    return torch.stack(
        (
            first * angle.cos() - second * angle.sin(),
            first * angle.sin() + second * angle.cos(),
        ),
        -1,
    ).reshape_as(x)


def cosine(x, y):
    return F.cosine_similarity(x.float(), y.float(), dim=-1, eps=1e-6).to(x.dtype)


# ============================================================================
# SSM
# ============================================================================


class SelectiveSSM(nn.Module):
    def __init__(self, width, state_size=8, backend="torch"):
        super().__init__()
        if backend not in ("torch", "cuda"):
            raise ValueError("backend must be torch or cuda")
        self.backend = backend
        self.state_size = state_size
        self.in_proj = nn.Linear(width, width * 2)
        self.depthwise = nn.Conv1d(width, width, 3, groups=width, padding=2)
        self.params = nn.Linear(width, width + 2 * state_size)
        self.A_log = nn.Parameter(
            torch.arange(1, state_size + 1).float().log().repeat(width, 1)
        )
        self.D = nn.Parameter(torch.ones(width))
        self.out_proj = nn.Linear(width, width)

    def forward(self, sequence, modulation=None):
        b, length, width = sequence.shape
        x, gate = self.in_proj(sequence).chunk(2, -1)
        x = F.silu(self.depthwise(x.transpose(1, 2))[..., :length].transpose(1, 2))
        dt, B, C = torch.split(
            self.params(x), [width, self.state_size, self.state_size], -1
        )
        dt = F.softplus(dt.float())
        if modulation is not None:
            dt = dt * (1 + modulation.float()[..., None])
        A = -self.A_log.float().exp()
        if self.backend == "cuda":
            if not sequence.is_cuda:
                raise RuntimeError(
                    "cuda scan requested for a CPU tensor; use backend=torch"
                )
            try:
                from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
            except ImportError as exc:
                raise RuntimeError(
                    "Install mamba-ssm with its CUDA selective_scan extension, or use backend=torch"
                ) from exc
            y = selective_scan_fn(
                x.float().transpose(1, 2).contiguous(),
                dt.transpose(1, 2).contiguous(),
                A,
                B.float().transpose(1, 2).contiguous(),
                C.float().transpose(1, 2).contiguous(),
                self.D.float(),
                delta_softplus=False,
            ).transpose(1, 2)
        else:
            state = torch.zeros(
                b, width, self.state_size, device=x.device, dtype=torch.float32
            )
            outputs = []
            for t in range(length):
                delta = dt[:, t, :, None]
                state = (
                    torch.exp(delta * A) * state
                    + delta * B[:, t, None, :].float() * x[:, t, :, None].float()
                )
                outputs.append(
                    (state * C[:, t, None, :].float()).sum(-1)
                    + self.D.float() * x[:, t].float()
                )
            y = torch.stack(outputs, 1)
        return self.out_proj(y.to(sequence.dtype) * F.silu(gate))


# ============================================================================
# MODEL
# ============================================================================


@dataclass
class ModelConfig:
    width: int = 24
    heads: int = 2
    state_size: int = 8
    frequency_scales: int = 2
    threshold: float = 0.55
    missing_threshold_shift: float = 0.2
    backend: str = "torch"
    use_align: bool = True
    use_atg: bool = True
    use_refine: bool = True

    def __post_init__(self):
        if self.width <= 0 or self.heads <= 0 or self.width % (6 * self.heads):
            raise ValueError("width must be a positive multiple of 6*heads for 3D RoPE")
        if self.state_size < 1 or self.frequency_scales < 1:
            raise ValueError("state_size and frequency_scales must be positive")
        if not 0 < self.threshold < 1 or not 0 <= self.missing_threshold_shift < 1:
            raise ValueError("invalid retention threshold")


def mlp(in_size, out_size, hidden=None):
    return nn.Sequential(
        nn.Linear(in_size, hidden or out_size),
        nn.GELU(),
        nn.Linear(hidden or out_size, out_size),
    )


def conv(in_size, out_size, stride=1):
    return nn.Sequential(
        nn.Conv3d(in_size, out_size, 3, stride, 1), nn.GroupNorm(1, out_size), nn.GELU()
    )


def laplacian_pyramid(x, scales):
    """Recursive low-pass and residuals at native level; reconstruction is exact."""
    current, residuals = x, []
    for _ in range(scales):
        size = tuple(max(1, s // 2) for s in current.shape[-3:])
        low = F.adaptive_avg_pool3d(current, size)
        restored = F.interpolate(
            low, size=current.shape[-3:], mode="trilinear", align_corners=False
        )
        residuals.append(current - restored)
        current = low
    return current, residuals


def reconstruct_pyramid(low, residuals):
    for residual in reversed(residuals):
        low = (
            F.interpolate(
                low, size=residual.shape[-3:], mode="trilinear", align_corners=False
            )
            + residual
        )
    return low


class FrequencyPrior(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        c = cfg.width
        self.cfg = cfg
        self.type_embedding = nn.Parameter(torch.randn(4, c) * 0.02)
        self.mask_embedding = nn.Parameter(torch.randn(4, c) * 0.02)
        self.position = mlp(3, c)
        self.scale_score = nn.ModuleList(
            [mlp(3 * c, 1) for _ in range(cfg.frequency_scales)]
        )
        self.freq = nn.Linear(2 * c, c)
        self.qkv = nn.ModuleList([nn.Linear(2 * c + 4, 3 * c) for _ in range(4)])
        self.pair_bias = nn.Parameter(torch.zeros(4, 4, cfg.heads))
        self.high_score = mlp(2 * c + 1, 1)
        self.prior = mlp(3 * c, c)

    def forward(self, features, available, visible, coords, index, valid):
        b, m, c, d, h, w = features.shape
        n = d * h * w
        all_low, all_high = [], []
        for modality in range(m):
            low, residuals = laplacian_pyramid(
                features[:, modality], self.cfg.frequency_scales
            )
            low = (
                F.interpolate(low, (d, h, w), mode="trilinear", align_corners=False)
                .flatten(2)
                .transpose(1, 2)
            )
            bands = [
                F.interpolate(t, (d, h, w), mode="trilinear", align_corners=False)
                .flatten(2)
                .transpose(1, 2)
                for t in residuals
            ]
            em = self.type_embedding[modality].expand(b, n, c)
            weights = torch.cat(
                [
                    score(torch.cat((band, low, em), -1))
                    for band, score in zip(bands, self.scale_score)
                ],
                -1,
            ).softmax(-1)
            high = sum(band * weights[..., i, None] for i, band in enumerate(bands))
            all_low.append(low)
            all_high.append(high)
        low, high = torch.stack(all_low, 1), torch.stack(all_high, 1)
        freq = self.freq(torch.cat((low, high), -1))
        # Patch visibility is explicit. It never makes an invisible token an edge source.
        norm_coords = (
            coords
            / coords.new_tensor([max(d - 1, 1), max(h - 1, 1), max(w - 1, 1)])
            * 2
            - 1
        )
        pos = self.position(norm_coords)[None].expand(b, -1, -1)
        z = (
            torch.where(visible[..., None], freq, self.mask_embedding[None, :, None, :])
            + pos[:, None]
        )
        queries, keys, values = [], [], []
        for modality in range(4):
            inp = torch.cat(
                (
                    z[:, modality],
                    self.type_embedding[modality].expand(b, n, c),
                    available[:, None].expand(-1, n, -1),
                ),
                -1,
            )
            q, k, v = self.qkv[modality](inp).chunk(3, -1)
            q = rope3d(q.reshape(b, n, self.cfg.heads, -1), coords)
            k = rope3d(k.reshape(b, n, self.cfg.heads, -1), coords)
            queries.append(q)
            keys.append(k)
            values.append(v.reshape(b, n, self.cfg.heads, -1))
        # N x (M K) attention, not global all-pairs attention.
        keys = torch.stack(keys, 2)[:, index].permute(0, 1, 4, 3, 2, 5).flatten(3, 4)
        values = (
            torch.stack(values, 2)[:, index].permute(0, 1, 4, 3, 2, 5).flatten(3, 4)
        )
        candidates = (
            (visible[:, :, index] & valid[None, None]).permute(0, 2, 1, 3).flatten(2, 3)
        )
        scores = []
        for modality, q in enumerate(queries):
            attention = (q[:, :, :, None, :] * keys).sum(-1) / (
                c // self.cfg.heads
            ) ** 0.5
            bias = (
                self.pair_bias[modality]
                .transpose(0, 1)
                .repeat_interleave(index.shape[1], -1)
            )
            attention = masked_softmax(
                attention + bias[None, None], candidates[:, :, None], -1
            )
            consensus = (attention[..., None] * values).sum(-2)
            scores.append(cosine(q, consensus).mean(-1))
        reliability = masked_softmax(torch.stack(scores, 1), visible, 1)
        shared_low = (low * reliability[..., None]).sum(1)
        high_weights = masked_softmax(
            self.high_score(torch.cat((high, z, reliability[..., None]), -1)).squeeze(
                -1
            ),
            visible,
            1,
        )
        shared_high = (high * high_weights[..., None]).sum(1)
        prior = self.prior(torch.cat((shared_low, shared_high, pos), -1))
        return z, reliability, prior


class MFHAlign(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        c = cfg.width
        self.frequency = FrequencyPrior(cfg)
        self.node = nn.ModuleList([nn.Linear(c, c) for _ in range(4)])
        self.to_edge = nn.ModuleList([nn.Linear(c, c) for _ in range(4)])
        self.from_edge = nn.ModuleList([nn.Linear(c, c) for _ in range(4)])
        self.to_virtual = nn.ModuleList([nn.Linear(c, c) for _ in range(4)])
        self.virtual = nn.Parameter(torch.randn(4, c) * 0.02)
        self.prior_node = nn.Linear(c, c)
        self.query = nn.Linear(c, c)
        self.key = nn.Linear(c, c)
        self.edge_conf = mlp(2 * c, 1)
        self.virtual_conf = mlp(2 * c, 1)
        self.comp_score = mlp(3 * c, 1)
        self.fuse = mlp(2 * c, c)

    def forward(self, features, available, visible, coords, index, valid):
        z, r, prior = self.frequency(features, available, visible, coords, index, valid)
        b, m, n, c = z.shape
        nodes = torch.stack([self.node[i](z[:, i]) for i in range(m)], 1)
        anchor = self.prior_node(prior)
        distance = (coords[index] - coords[:, None]).square().sum(-1) / 2
        logits = []
        for i in range(m):
            score = (
                gather(self.query(nodes[:, i]), index) * self.key(anchor)[:, :, None]
            ).sum(-1) / c**0.5
            logits.append(
                score.float()
                - distance[None]
                + torch.log(r[:, i, index].float().clamp_min(1e-8))
            )
        logits = torch.stack(logits, 2)  # B,E,M,K
        mask = (visible[:, :, index] & valid[None, None]).permute(0, 2, 1, 3)
        incidence = masked_softmax(logits.flatten(2), mask.flatten(2), -1).reshape_as(
            logits
        )
        edge = anchor + sum(
            (
                gather(self.to_edge[i](nodes[:, i]), index)
                * incidence[:, :, i, :, None]
            ).sum(2)
            for i in range(m)
        )
        omega = self.edge_conf(torch.cat((edge, anchor), -1)).sigmoid().squeeze(-1)
        aligned = []
        confidences = []
        for i in range(m):
            observed = nodes[:, i] + scatter_regions(
                self.from_edge[i](edge) * omega[..., None], incidence[:, :, i], index
            )
            virt = self.virtual[i].expand(b, n, c)
            candidate_edge = gather(edge, index)
            compensation = masked_softmax(
                self.comp_score(
                    torch.cat(
                        (
                            virt[:, :, None].expand_as(candidate_edge),
                            candidate_edge,
                            prior[:, :, None].expand_as(candidate_edge),
                        ),
                        -1,
                    )
                ).squeeze(-1)
                - distance[None],
                valid[None],
                -1,
            )
            virt = virt + (
                compensation[..., None]
                * gather(self.to_virtual[i](edge) * omega[..., None], index)
            ).sum(2)
            aligned.append(torch.where(visible[:, i, :, None], observed, virt))
            confidences.append(
                torch.where(
                    visible[:, i],
                    r[:, i],
                    self.virtual_conf(torch.cat((virt, prior), -1))
                    .sigmoid()
                    .squeeze(-1),
                )
            )
        confidence = torch.stack(confidences, 1)
        confidence = confidence / confidence.sum(1, keepdim=True).clamp_min(1e-8)
        fused = (torch.stack(aligned, 1) * confidence[..., None]).sum(1)
        outgoing = incidence.sum(2)
        return self.fuse(torch.cat((prior, fused), -1)), {
            "prior": prior,
            "reliability": r,
            "visible": visible,
            "incidence": incidence,
            "outgoing": outgoing,
            "h_in": incoming(outgoing, index, valid),
            "edge": edge,
            "omega": omega,
            "index": index,
            "valid": valid,
            "coords": coords,
        }


class ATGMamba(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        c = cfg.width
        self.cfg = cfg
        self.context = mlp(3 * c, c)
        self.direction = nn.Linear(c, 4)
        self.scale = nn.Linear(c, 2)
        self.scans = nn.ModuleList(
            [SelectiveSSM(c, cfg.state_size, cfg.backend) for _ in range(4)]
        )
        self.direction_gate = mlp(5 * c + 4, 4)
        self.feature_gate = mlp(3 * c, c)
        self.output = nn.Linear(c, c)
        self.score = mlp(3 * c, 1)
        self.deep_scan = SelectiveSSM(c, cfg.state_size, cfg.backend)
        self.deep_ff = nn.Sequential(nn.LayerNorm(c), mlp(c, c, 2 * c))

    def forward(self, x, graph, available, shape):
        b, n, c = x.shape
        d, h, w = shape
        prior, index = graph["prior"], graph["index"]
        context = self.context(
            torch.cat(
                (
                    x,
                    prior,
                    scatter_regions(
                        graph["edge"] * graph["omega"][..., None],
                        graph["outgoing"],
                        index,
                    ),
                ),
                -1,
            )
        )
        directions = self.direction(context).softmax(-1)
        scales = F.softplus(self.scale(context)) + 1e-3
        grid = torch.arange(n, device=x.device).reshape(d, h, w)
        horizontal = grid.reshape(d, -1)
        vertical = grid.transpose(1, 2).reshape(d, -1)
        orders = [horizontal, horizontal.flip(-1), vertical, vertical.flip(-1)]
        outputs = []
        for k, order in enumerate(orders):
            previous = order.roll(1, -1)
            previous[:, 0] = order[:, 0]
            predecessor = torch.empty(n, device=x.device, dtype=torch.long)
            predecessor[order.flatten()] = previous.flatten()
            affinity = shared_membership(
                graph["h_in"], graph["omega"], index, predecessor
            )
            displacement = (graph["coords"] - graph["coords"][predecessor])[:, 1:]
            geometry = torch.exp(-0.5 * (displacement[None] / scales).square().sum(-1))
            modulation = directions[..., k] * affinity * geometry
            sequence = x[:, order].reshape(b * d, h * w, c)
            result = self.scans[k](
                sequence, modulation[:, order].reshape(b * d, h * w)
            ).reshape(b, n, c)
            restore = order.flatten().argsort()
            outputs.append(result[:, restore])
        packed = torch.cat(outputs, -1)
        gate = self.direction_gate(
            torch.cat((packed, context, available[:, None].expand(-1, n, -1)), -1)
        ).softmax(-1)
        direction = sum(y * gate[..., i, None] for i, y in enumerate(outputs))
        feature_gate = self.feature_gate(torch.cat((direction, x, prior), -1)).sigmoid()
        u = x + feature_gate * self.output(direction)
        score = self.score(torch.cat((u, context, prior), -1)).sigmoid().squeeze(-1)
        mu = 1 - available.mean(-1)
        threshold = (self.cfg.threshold - self.cfg.missing_threshold_shift * mu).clamp(
            0.05, 0.95
        )
        hard = score >= threshold[:, None]
        # At least one retained token per axial plane. This prevents empty scan calls.
        planes = hard.reshape(b, d, h * w)
        best = score.reshape(b, d, h * w).argmax(-1, keepdim=True)
        planes = planes.scatter(2, best, True)
        hard = planes.reshape(b, n)
        keep = (
            hard.to(score.dtype) + score - score.detach()
            if self.training
            else hard.to(score.dtype)
        )
        if self.training:
            # Dense training is required for the straight-through path of dropped tokens.
            deep = self.deep_scan(u.reshape(b * d, h * w, c)).reshape(b, n, c)
            deep = deep + self.deep_ff(deep)
        else:
            # Actual gather -> expensive branch -> scatter (not just multiplying by zero).
            deep = torch.zeros_like(u)
            for sample in range(b):
                for plane in range(d):
                    ids = (
                        torch.nonzero(
                            hard[sample, plane * h * w : (plane + 1) * h * w],
                            as_tuple=False,
                        ).flatten()
                        + plane * h * w
                    )
                    chosen = self.deep_scan(u[sample : sample + 1, ids])
                    chosen = chosen + self.deep_ff(chosen)
                    deep[sample, ids] = chosen[0]
        return x + keep[..., None] * deep, {
            "score": score,
            "keep": keep,
            "hard_keep": hard,
            "scales": scales,
            "threshold": threshold,
        }


class SDHRefine(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.region = mlp(3 * c, c)
        self.p = nn.Linear(c, c)
        self.r = nn.Linear(c, c)
        self.gate = mlp(3 * c + 2, c)
        self.output = nn.Linear(c, c)

    def forward(self, x, graph, state):
        index = graph["index"]
        out = graph["outgoing"]
        prior = graph["prior"]
        weights = out * state["keep"][:, index] * graph["omega"][..., None]
        total = weights.sum(-1, keepdim=True)
        # Empty retained edges fall back to their prior/old consensus. A 1/eps
        # gradient through zero membership would otherwise overflow ST + AMP.
        weights = torch.where(
            total > 1e-6, weights / total.clamp_min(1e-6), torch.zeros_like(weights)
        )
        region = self.region(
            torch.cat(
                (
                    region_pool(x, weights, index),
                    region_pool(prior, out, index),
                    graph["edge"],
                ),
                -1,
            )
        )
        similarity = (
            cosine(self.p(prior)[:, :, None], gather(self.r(region), index)) / 0.2
        )
        logweights = (
            similarity.float()
            + torch.log(graph["h_in"].float().clamp_min(1e-8))
            + torch.log(graph["omega"][:, index].float().clamp_min(1e-8))
        )
        weights_back = masked_softmax(
            logweights, (graph["h_in"] > 0) & graph["valid"][None], -1
        )
        back = (gather(region, index) * weights_back[..., None]).sum(2)
        gate = self.gate(
            torch.cat(
                (x, back, prior, state["score"][..., None], state["keep"][..., None]),
                -1,
            )
        ).sigmoid()
        return x + (1 - state["score"][..., None]) * gate * self.output(back)


class MMAHHMamba(nn.Module):
    def __init__(self, cfg=None):
        super().__init__()
        self.cfg = cfg or ModelConfig()
        c = self.cfg.width
        self.stems = nn.ModuleList(
            [nn.Sequential(conv(1, c // 2, 2), conv(c // 2, c, 2)) for _ in range(4)]
        )
        self.align = MFHAlign(self.cfg)
        self.atg = ATGMamba(self.cfg)
        self.refine = SDHRefine(c)
        self.skip = conv(8, c // 2)
        self.decode = nn.Sequential(conv(c + c // 2, c), nn.Conv3d(c, 4, 1))

    def forward(self, image, available=None, patch_visible=None, return_aux=False):
        if image.ndim != 5 or image.shape[1] != 4 or min(image.shape[2:]) < 4:
            raise ValueError("image must have shape [B,4,D,H,W], spatial sizes >=4")
        b = image.shape[0]
        if available is None:
            available = image.new_ones((b, 4))
        available = available.to(device=image.device, dtype=image.dtype)
        if (
            available.shape != (b, 4)
            or not torch.all((available == 0) | (available == 1))
            or torch.any(available.sum(-1) == 0)
        ):
            raise ValueError(
                "available must be binary [B,4] with at least one observed modality per sample"
            )
        # where also discards NaN placeholders in absent channels.
        masked = torch.where(
            available[:, :, None, None, None].bool(), image, torch.zeros_like(image)
        )
        features = torch.stack(
            [stem(masked[:, i : i + 1]) for i, stem in enumerate(self.stems)], 1
        )
        shape = features.shape[-3:]
        n = shape[0] * shape[1] * shape[2]
        visible = available[:, :, None].bool().expand(-1, -1, n)
        if patch_visible is not None:
            if not self.training:
                raise ValueError("patch visibility is training-only")
            if patch_visible.shape != (b, 4, *shape):
                raise ValueError("patch_visible must match the latent grid")
            visible = visible & patch_visible.flatten(2).bool()
        coords, index, valid = topology(shape, image.device)
        aligned, graph = self.align(features, available, visible, coords, index, valid)
        if not self.cfg.use_align:
            aligned = (features.flatten(3).transpose(2, 3) * visible[..., None]).sum(
                1
            ) / visible.sum(1).clamp_min(1)[..., None]
        if self.cfg.use_atg:
            deep, state = self.atg(aligned, graph, available, shape)
        else:
            deep = aligned
            state = {
                "score": aligned.new_ones((b, n)),
                "keep": aligned.new_ones((b, n)),
                "hard_keep": torch.ones(b, n, dtype=torch.bool, device=image.device),
                "scales": aligned.new_ones((b, n, 2)),
            }
        refined = self.refine(deep, graph, state) if self.cfg.use_refine else deep
        decoded = F.interpolate(
            refined.transpose(1, 2).reshape(b, -1, *shape),
            image.shape[2:],
            mode="trilinear",
            align_corners=False,
        )
        avail_grid = available[:, :, None, None, None].expand_as(image)
        skip = self.skip(torch.cat((masked, avail_grid), 1))
        logits = self.decode(torch.cat((decoded, skip), 1))
        if return_aux:
            return logits, dict(graph, **state, available=available)
        return logits


# ============================================================================
# LOSSES
# ============================================================================


def region_probabilities(logits):
    p = logits.softmax(1)
    return torch.stack((p[:, 3], p[:, 1] + p[:, 3], 1 - p[:, 0]), 1)  # ET, TC, WT


def target_regions(target):
    return torch.stack((target == 3, (target == 1) | (target == 3), target > 0), 1)


def soft_dice_per_case(logits, target):
    p = region_probabilities(logits).float()
    y = target_regions(target).float()
    dims = tuple(range(2, p.ndim))
    return ((2 * (p * y).sum(dims) + 1e-5) / (p.sum(dims) + y.sum(dims) + 1e-5)).mean(1)


def segmentation_loss(logits, target):
    return (
        1
        - soft_dice_per_case(logits, target).mean()
        + F.cross_entropy(logits.float(), target)
    )


def auxiliary_losses(aux, rho_min=0.45, rho_max=0.85):
    index = aux["index"]
    out = aux["outgoing"]
    keep = aux["keep"]
    weights = out * keep[:, index] * aux["omega"][..., None]
    total = weights.sum(-1, keepdim=True)
    weights = torch.where(
        total > 1e-6, weights / total.clamp_min(1e-6), torch.zeros_like(weights)
    )
    # Coordinates and scales use the same latent-token units (H,W).
    points = aux["coords"][index, 1:][None]
    center = (weights[..., None] * points).sum(2)
    variance = (
        (weights[..., None] * (points - center[:, :, None]).square()).sum(2).detach()
    )
    predicted = region_pool(aux["scales"].square(), weights, index)
    active = (total.squeeze(-1) > 1e-6).float()
    geo = (
        (predicted - variance).square().mean(-1) * active
    ).sum() / active.sum().clamp_min(1)
    target = rho_min + (rho_max - rho_min) * (1 - aux["available"].mean(-1))
    budget = (keep.mean(-1) - target).square().mean()
    return geo, budget


def contribution_loss(
    model, image, target, available, logits, aux, patch_visible=None, temperature=0.2
):
    """Exact within-batch contribution distribution; up to four no-grad extra passes.

    Singleton cases are excluded: removing their only observation is undefined.
    Unlike the manuscript's ambiguous one-sample estimator, no unmeasured modal
    contribution is silently treated as zero. See METHOD_MAPPING.md.
    """
    eligible = available.sum(-1) > 1
    if not eligible.any():
        return logits.sum() * 0
    delta = torch.zeros_like(available)
    with torch.no_grad():
        baseline = soft_dice_per_case(logits, target)
        for m in range(4):
            chosen = eligible & available[:, m].bool()
            if not chosen.any():
                continue
            removed = available[chosen].clone()
            removed[:, m] = 0
            patch = patch_visible[chosen] if patch_visible is not None else None
            counterfactual = model(image[chosen], removed, patch_visible=patch)
            delta[chosen, m] = baseline[chosen] - soft_dice_per_case(
                counterfactual, target[chosen]
            )
    visibility = aux["visible"].float()
    rel = (aux["reliability"] * visibility).sum(-1) / visibility.sum(-1).clamp_min(1)
    rel = rel[eligible].float().masked_fill(~available[eligible].bool(), -1e9)
    target_distribution = (
        (delta[eligible] / temperature)
        .masked_fill(~available[eligible].bool(), -1e9)
        .softmax(-1)
    )
    return F.kl_div(rel.log_softmax(-1), target_distribution, reduction="batchmean")


# ============================================================================
# DATA
# ============================================================================


MODALITIES = ("t1", "t1ce", "t2", "flair")


def read_manifest(path):
    path = Path(path).resolve()
    obj = json.loads(path.read_text(encoding="utf-8"))
    if obj.get("modalities") != list(MODALITIES):
        raise ValueError("manifest modalities must be [t1,t1ce,t2,flair]")
    cases = obj["cases"]
    ids = [c["id"] for c in cases]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate case IDs")
    for case in cases:
        for key in ("npz", "label"):
            if case.get(key):
                case[key] = str((path.parent / case[key]).resolve())
        if "images" in case:
            case["images"] = {
                k: str((path.parent / v).resolve())
                for k, v in case["images"].items()
                if v
            }
    return cases


def normalize(image):
    output = np.zeros_like(image, dtype=np.float32)
    for i, channel in enumerate(image):
        mask = (channel != 0) & np.isfinite(channel)
        if mask.any():
            values = channel[mask]
            output[i, mask] = (values - values.mean()) / max(float(values.std()), 1e-6)
    return output


def load_case(case, require_label=True):
    if "npz" in case:
        with np.load(case["npz"], allow_pickle=False) as data:
            image = data["image"].astype(np.float32)
            raw = data["label"].astype(np.int64) if "label" in data else None
            available = (
                data["available"].astype(np.float32)
                if "available" in data
                else np.ones(4, np.float32)
            )
            spacing = data["spacing"].astype(float) if "spacing" in data else np.ones(3)
            affine = (
                data["affine"].astype(float)
                if "affine" in data
                else np.diag([*spacing, 1])
            )
    else:
        import nibabel as nib

        loaded = {
            key: nib.load(value)
            for key, value in case["images"].items()
            if key in MODALITIES
        }
        if not loaded:
            raise ValueError("no images for " + case["id"])
        ref = next(iter(loaded.values()))
        shape = ref.shape
        if len(shape) != 3:
            raise ValueError("expected 3D NIfTI")
        for volume in loaded.values():
            if volume.shape != shape or not np.allclose(
                volume.affine, ref.affine, atol=1e-4
            ):
                raise ValueError("modalities must be co-registered: " + case["id"])
        image = np.stack(
            [
                loaded[m].get_fdata(dtype=np.float32)
                if m in loaded
                else np.zeros(shape, np.float32)
                for m in MODALITIES
            ]
        )
        available = np.asarray([m in loaded for m in MODALITIES], np.float32)
        spacing = np.asarray(ref.header.get_zooms()[:3], float)
        affine = ref.affine
        raw = None
        if case.get("label"):
            seg = nib.load(case["label"])
            if seg.shape != shape or not np.allclose(seg.affine, ref.affine, atol=1e-4):
                raise ValueError("label not registered to images")
            raw = seg.get_fdata().astype(np.int64)
    if (
        image.ndim != 4
        or image.shape[0] != 4
        or available.shape != (4,)
        or not np.isin(available, [0, 1]).all()
        or available.sum() == 0
    ):
        raise ValueError("invalid image/availability: " + case["id"])
    if (
        np.asarray(spacing).shape != (3,)
        or not np.isfinite(spacing).all()
        or np.any(spacing <= 0)
    ):
        raise ValueError("invalid voxel spacing")
    if not np.isfinite(image[available.astype(bool)]).all():
        raise ValueError("nonfinite observed image")
    image[available == 0] = 0
    if require_label and raw is None:
        raise ValueError("missing label: " + case["id"])
    if raw is not None:
        if raw.shape != image.shape[1:] or not np.isin(raw, [0, 1, 2, 4]).all():
            raise ValueError(
                "expected BraTS 2019-2021 raw labels {0,1,2,4}; do not pass pre-remapped labels"
            )
        raw = raw.copy()
        raw[raw == 4] = 3
    return {
        "image": normalize(image),
        "target": raw,
        "available": available,
        "spacing": spacing,
        "affine": affine,
        "id": case["id"],
    }


def random_modality_mask(available, drop_probability=0.25):
    if not 0 <= drop_probability <= 1:
        raise ValueError("invalid drop probability")
    keep = (torch.rand_like(available) > drop_probability) & available.bool()
    for i in range(len(keep)):
        if not keep[i].any():
            candidates = torch.nonzero(available[i], as_tuple=False).flatten()
            if len(candidates) == 0:
                raise ValueError("empty acquisition")
            keep[
                i,
                candidates[
                    torch.randint(len(candidates), (1,), device=available.device)
                ],
            ] = True
    return keep.to(available.dtype)


def patch_mask(available, shape, probability=0.1):
    mask = torch.rand(len(available), 4, *shape, device=available.device) > probability
    # An observed anchor modality at every position avoids an entirely empty local field.
    for b in range(len(available)):
        anchor = torch.nonzero(available[b], as_tuple=False)[0, 0]
        mask[b, anchor] = True
    return mask


class PatchDataset(Dataset):
    def __init__(
        self, cases, patch_size=(96, 96, 96), samples_per_case=1, augment=True
    ):
        if not cases:
            raise ValueError("empty dataset")
        self.cases = cases
        self.patch = tuple(patch_size)
        self.samples = samples_per_case
        self.augment = augment

    def __len__(self):
        return len(self.cases) * self.samples

    def __getitem__(self, i):
        case = load_case(self.cases[i % len(self.cases)])
        x = torch.from_numpy(case["image"])
        y = torch.from_numpy(case["target"])
        pad = []
        for size, minimum in reversed(list(zip(y.shape, self.patch))):
            pad.extend([0, max(0, minimum - size)])
        x = F.pad(x, pad)
        y = F.pad(y, pad)
        # Half of crops are centered on a foreground voxel, if one exists.
        fg = torch.nonzero(y > 0, as_tuple=False)
        if len(fg) and torch.rand(()) < 0.5:
            center = fg[torch.randint(len(fg), (1,)).item()]
            starts = [
                int(max(0, min(int(c) - p // 2, s - p)))
                for c, s, p in zip(center, y.shape, self.patch)
            ]
        else:
            starts = [
                int(torch.randint(s - p + 1, (1,))) for s, p in zip(y.shape, self.patch)
            ]
        sl = tuple(slice(a, a + p) for a, p in zip(starts, self.patch))
        x = x[(slice(None),) + sl]
        y = y[sl]
        if self.augment:
            for dim in range(3):
                if torch.rand(()) < 0.5:
                    x = x.flip(dim + 1)
                    y = y.flip(dim)
            # Small-angle 3D rotation, shared across all modalities and the label.
            angles = (torch.rand(3) - 0.5) * 0.2
            a, b, c = angles
            zero = angles.new_tensor(0)
            one = angles.new_tensor(1)
            rx = torch.stack(
                (one, zero, zero, zero, a.cos(), -a.sin(), zero, a.sin(), a.cos())
            ).reshape(3, 3)
            ry = torch.stack(
                (b.cos(), zero, b.sin(), zero, one, zero, -b.sin(), zero, b.cos())
            ).reshape(3, 3)
            rz = torch.stack(
                (c.cos(), -c.sin(), zero, c.sin(), c.cos(), zero, zero, zero, one)
            ).reshape(3, 3)
            theta = torch.cat((rz @ ry @ rx, torch.zeros(3, 1)), 1)[None]
            grid = F.affine_grid(theta, (1, 4, *self.patch), align_corners=False)
            x = F.grid_sample(x[None], grid, mode="bilinear", align_corners=False)[0]
            y = F.grid_sample(
                y[None, None].float(), grid, mode="nearest", align_corners=False
            )[0, 0].long()
            valid = x != 0
            x = torch.where(
                valid,
                x * (0.9 + 0.2 * torch.rand(4, 1, 1, 1))
                + (torch.rand(4, 1, 1, 1) - 0.5) * 0.2
                + 0.01 * torch.randn_like(x),
                x,
            )
        return {
            "image": x.contiguous(),
            "target": y.contiguous(),
            "available": torch.from_numpy(case["available"]),
        }


# ============================================================================
# INFERENCE
# ============================================================================


def modality_combinations():
    return [tuple((code >> i) & 1 for i in range(4)) for code in range(1, 16)]


@torch.no_grad()
def sliding_window(
    model, image, available, patch_size=(96, 96, 96), overlap=0.5, device="cpu"
):
    if image.ndim != 4 or image.shape[0] != 4:
        raise ValueError("expected image [4,D,H,W]")
    if not 0 <= overlap < 1 or min(patch_size) < 4:
        raise ValueError("invalid window parameters")
    original = image.shape[1:]
    pad = []
    for a, b in reversed(list(zip(original, patch_size))):
        pad.extend([0, max(0, b - a)])
    image = F.pad(image.cpu(), pad)
    starts = []
    for size, patch in zip(image.shape[1:], patch_size):
        stride = max(1, int(patch * (1 - overlap)))
        axis = list(range(0, size - patch + 1, stride))
        if axis[-1] != size - patch:
            axis.append(size - patch)
        starts.append(axis)
    coords = torch.meshgrid(
        *[torch.linspace(-1, 1, s) for s in patch_size], indexing="ij"
    )
    weight = torch.exp(-sum(c * c for c in coords) / (0.5**2)).clamp_min(1e-3)
    result = torch.zeros(4, *image.shape[1:])
    normalizer = torch.zeros(image.shape[1:])
    was_training = model.training
    model.eval()
    try:
        for start in product(*starts):
            sl = tuple(slice(a, a + p) for a, p in zip(start, patch_size))
            logits = model(
                image[(slice(None),) + sl][None].to(device), available[None].to(device)
            )
            result[(slice(None),) + sl] += logits.softmax(1)[0].float().cpu() * weight
            normalizer[sl] += weight
    finally:
        model.train(was_training)
    result /= normalizer.clamp_min(1e-8)
    return result[(slice(None),) + tuple(slice(0, s) for s in original)]


# ============================================================================
# METRICS
# ============================================================================


def region_masks(label):
    return {"ET": label == 3, "TC": (label == 1) | (label == 3), "WT": label > 0}


def binary_metrics(pred, target, spacing=(1, 1, 1)):
    pred = np.asarray(pred, dtype=bool)
    target = np.asarray(target, dtype=bool)
    p = int(pred.sum())
    t = int(target.sum())
    tp = int((pred & target).sum())
    dice = 2 * tp / (p + t) if p + t else 1.0
    sensitivity = tp / t if t else None  # undefined for an empty reference region
    if not p and not t:
        hd95 = 0.0
    elif not p or not t:
        hd95 = float("inf")
    else:
        st = generate_binary_structure(3, 1)
        ps = pred ^ binary_erosion(pred, st, border_value=0)
        ts = target ^ binary_erosion(target, st, border_value=0)
        # Maximum of the two directed 95th percentiles (documented convention).
        a = distance_transform_edt(~ts, sampling=spacing)[ps]
        b = distance_transform_edt(~ps, sampling=spacing)[ts]
        hd95 = float(max(np.percentile(a, 95), np.percentile(b, 95)))
    return {"dice": float(dice), "sensitivity": sensitivity, "hd95_mm": hd95}


def evaluate_regions(pred, target, spacing):
    return {
        key: binary_metrics(value, region_masks(target)[key], spacing)
        for key, value in region_masks(pred).items()
    }


# ============================================================================
# UTILS
# ============================================================================


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def worker_seed(worker_id):
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )


def load_checkpoint(path, device="cpu", backend=None):
    # Only load your own/trusted checkpoints: optimizer/RNG states use pickle.
    saved = torch.load(path, map_location="cpu", weights_only=False)
    config = dict(saved["model_config"])
    if backend:
        config["backend"] = backend
    model = MMAHHMamba(ModelConfig(**config))
    model.load_state_dict(saved["model"])
    model.to(device).eval()
    return model, saved


def select_cases(manifest, split_path, fold, partition):
    cases = read_manifest(manifest)
    splits = json.loads(Path(split_path).read_text(encoding="utf-8"))
    item = splits["folds"][fold]
    groups = [set(item[key]) for key in ("train", "val", "test")]
    if any(groups[i] & groups[j] for i in range(3) for j in range(i + 1, 3)):
        raise ValueError("patient overlap between train/val/test")
    ids = set(c["id"] for c in cases)
    if not set.union(*groups).issubset(ids):
        raise ValueError("split references unknown patient IDs")
    chosen = set(item[partition])
    result = [c for c in cases if c["id"] in chosen]
    if not result:
        raise ValueError("empty " + partition + " partition")
    return result


# ============================================================================
# PREPARE
# ============================================================================


def prepare_main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--allow-missing-label", action="store_true")
    args = parser.parse_args()
    root = Path(args.root).resolve()
    output = Path(args.output).resolve()
    if not root.is_dir():
        parser.error("root is not a directory")
    cases = []
    for folder in sorted(root.rglob("*")):
        if not folder.is_dir():
            continue
        images = {}
        for modality in MODALITIES:
            files = sorted(folder.glob("*_" + modality + ".nii*"))
            if len(files) > 1:
                raise ValueError("ambiguous modality files in " + str(folder))
            if files:
                images[modality] = os.path.relpath(files[0], output.parent).replace(
                    "\\", "/"
                )
        if not images:
            continue
        labels = sorted(folder.glob("*_seg.nii*"))
        if len(labels) > 1:
            raise ValueError("ambiguous labels")
        if not labels and not args.allow_missing_label:
            raise ValueError("no segmentation in " + str(folder))
        case = {"id": folder.name, "images": images}
        if labels:
            case["label"] = os.path.relpath(labels[0], output.parent).replace("\\", "/")
        cases.append(case)
    if not cases:
        raise ValueError("no BraTS cases found; expected <patient>_t1.nii.gz etc.")
    if len({c["id"] for c in cases}) != len(cases):
        raise ValueError("duplicate patient folder names")
    save_json(output, {"modalities": list(MODALITIES), "cases": cases})
    print("Wrote", len(cases), "cases to", output)


# ============================================================================
# SPLITS
# ============================================================================


def make_splits(ids, n_folds=5, seed=2026, val_fraction=0.1):
    if len(ids) != len(set(ids)) or n_folds < 2 or len(ids) < 2 * n_folds:
        raise ValueError("need unique IDs, >=2 folds and at least 2 patients per fold")
    if not 0 < val_fraction < 0.5:
        raise ValueError("val_fraction must be between 0 and .5")
    rng = np.random.default_rng(seed)
    ids = np.asarray(sorted(ids))
    rng.shuffle(ids)
    chunks = np.array_split(ids, n_folds)
    folds = []
    for k in range(n_folds):
        rest = np.concatenate([x for i, x in enumerate(chunks) if i != k]).copy()
        rng.shuffle(rest)
        n_val = max(1, int(round(len(rest) * val_fraction)))
        if n_val >= len(rest):
            raise ValueError("too few training patients")
        folds.append(
            {
                "train": sorted(rest[n_val:].tolist()),
                "val": sorted(rest[:n_val].tolist()),
                "test": sorted(chunks[k].tolist()),
            }
        )
    return {"seed": seed, "val_fraction": val_fraction, "folds": folds}


def splits_main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--val-fraction", type=float, default=0.1)
    a = p.parse_args()
    cases = read_manifest(a.manifest)
    save_json(
        a.output, make_splits([c["id"] for c in cases], a.folds, a.seed, a.val_fraction)
    )
    print("Saved", a.folds, "disjoint outer folds to", a.output)


# ============================================================================
# TRAIN
# ============================================================================


@torch.no_grad()
def validate(model, cases, patch, device):
    scores = []
    for case in cases:
        item = load_case(case)
        pred = (
            sliding_window(
                model,
                torch.from_numpy(item["image"]),
                torch.from_numpy(item["available"]),
                patch,
                0.5,
                device,
            )
            .argmax(0)
            .numpy()
        )
        metrics = evaluate_regions(pred, item["target"], item["spacing"])
        scores.append(np.mean([v["dice"] for v in metrics.values()]))
    return float(np.mean(scores))


def train_main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="reference")
    p.add_argument("--manifest", required=True)
    p.add_argument("--splits", required=True)
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--output", required=True)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--backend", choices=["torch", "cuda"])
    p.add_argument("--epochs", type=int)
    p.add_argument(
        "--max-steps",
        type=int,
        help="Optional debug cap per epoch, not for paper experiments",
    )
    p.add_argument(
        "--resume", help="Trusted checkpoint from this training configuration"
    )
    a = p.parse_args()
    config = load_config(a.config)
    train = config["training"]
    mc = dict(config["model"])
    if a.backend:
        mc["backend"] = a.backend
    if a.epochs:
        train["epochs"] = a.epochs
    seed_all(train["seed"])
    torch.set_num_threads(train.get("cpu_threads", 4))
    device = torch.device(a.device)
    model = MMAHHMamba(ModelConfig(**mc)).to(device)
    cases = select_cases(a.manifest, a.splits, a.fold, "train")
    val = select_cases(a.manifest, a.splits, a.fold, "val")
    ds = PatchDataset(
        cases, train["patch_size"], train.get("samples_per_case", 1), augment=True
    )
    generator = torch.Generator().manual_seed(train["seed"])
    loader = DataLoader(
        ds,
        batch_size=train["batch_size"],
        shuffle=True,
        num_workers=train["workers"],
        pin_memory=device.type == "cuda",
        worker_init_fn=worker_seed,
        generator=generator,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=train["lr"], weight_decay=train["weight_decay"]
    )
    amp = bool(train.get("amp", False) and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=amp, init_scale=128.0)
    steps = min(len(loader), a.max_steps) if a.max_steps else len(loader)
    if steps < 1 or train["epochs"] < 1:
        raise ValueError("need positive steps/epochs")
    total = train["epochs"] * steps
    warm = max(1, round(total * 0.05))
    global_step = 0
    start = 0
    best = -1.0
    output = Path(a.output)
    output.mkdir(parents=True, exist_ok=True)
    resolved = {
        "model": asdict(model.cfg),
        "training": train,
        "fold": a.fold,
        "train_ids": [c["id"] for c in cases],
        "val_ids": [c["id"] for c in val],
    }
    save_json(output / "config.json", resolved)
    if a.resume:
        saved = torch.load(a.resume, map_location=device, weights_only=False)
        if saved["model_config"] != asdict(model.cfg):
            raise ValueError("resume model config differs")
        if (
            saved["fold"] != a.fold
            or saved["train_ids"] != resolved["train_ids"]
            or saved["val_ids"] != resolved["val_ids"]
        ):
            raise ValueError("resume data partition differs")
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        scaler.load_state_dict(saved["scaler"])
        start = saved["epoch"] + 1
        global_step = saved["global_step"]
        best = saved["best_val_dice"]
        random.setstate(saved["rng_python"])
        np.random.set_state(saved["rng_numpy"])
        torch.set_rng_state(saved["rng_torch"].cpu())
        generator.set_state(saved["rng_loader"].cpu())
        if device.type == "cuda" and saved["rng_cuda"] is not None:
            torch.cuda.set_rng_state_all([x.cpu() for x in saved["rng_cuda"]])
    for epoch in range(start, train["epochs"]):
        model.train()
        totals = []
        for step, batch in enumerate(loader):
            if step >= steps:
                break
            progress = max(0, (global_step - warm) / max(1, total - warm))
            lr = (
                train["lr"] * (global_step + 1) / warm
                if global_step < warm
                else train["min_lr"]
                + (train["lr"] - train["min_lr"])
                * 0.5
                * (1 + math.cos(math.pi * progress))
            )
            for group in optimizer.param_groups:
                group["lr"] = lr
            x = batch["image"].to(device)
            y = batch["target"].to(device)
            available = random_modality_mask(
                batch["available"].to(device), train["modality_drop"]
            )
            shape = tuple((s + 3) // 4 for s in x.shape[-3:])
            local = patch_mask(available, shape, train["patch_drop"])
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type, enabled=amp, dtype=torch.float16
            ):
                logits, aux = model(x, available, patch_visible=local, return_aux=True)
                seg = segmentation_loss(logits, y)
                geo, budget = auxiliary_losses(aux, train["rho_min"], train["rho_max"])
                con = (
                    contribution_loss(model, x, y, available, logits, aux, local)
                    if train["lambda_contribution"] > 0
                    else seg * 0
                )
                ramp = min(
                    1.0, (epoch + 1) / max(1, train.get("aux_warmup_epochs", 10))
                )
                loss = seg + ramp * (
                    train["lambda_contribution"] * con
                    + train["lambda_geometry"] * geo
                    + train["lambda_budget"] * budget
                )
            if not torch.isfinite(loss):
                raise FloatingPointError("nonfinite training loss")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), 1.0, error_if_nonfinite=not amp
            )
            scale_before = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            skipped = scaler.get_scale() < scale_before
            global_step += 1
            totals.append(
                {
                    "loss": float(loss.detach()),
                    "seg": float(seg.detach()),
                    "con": float(con.detach()),
                    "geo": float(geo.detach()),
                    "budget": float(budget.detach()),
                    "keep": float(aux["hard_keep"].float().mean()),
                    "amp_skipped": float(skipped),
                }
            )
        if all(t["amp_skipped"] for t in totals):
            raise FloatingPointError(
                "All optimizer updates overflowed; disable AMP or inspect the data/loss"
            )
        val_score = validate(model, val, tuple(train["patch_size"]), device)
        improved = val_score > best
        best = max(best, val_score)
        log = {
            "epoch": epoch,
            "global_step": global_step,
            "lr": lr,
            "val_mean_dice": val_score,
            **{key: float(np.mean([t[key] for t in totals])) for key in totals[0]},
        }
        with (output / "train.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(log, allow_nan=False) + "\n")
        checkpoint = {
            "model": model.state_dict(),
            "model_config": asdict(model.cfg),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "epoch": epoch,
            "global_step": global_step,
            "best_val_dice": best,
            "training": train,
            "fold": a.fold,
            "train_ids": resolved["train_ids"],
            "val_ids": resolved["val_ids"],
            "rng_python": random.getstate(),
            "rng_numpy": np.random.get_state(),
            "rng_torch": torch.get_rng_state(),
            "rng_loader": generator.get_state(),
            "rng_cuda": torch.cuda.get_rng_state_all()
            if device.type == "cuda"
            else None,
        }
        torch.save(checkpoint, output / "last.pt")
        if improved:
            torch.save(checkpoint, output / "best.pt")
        print(json.dumps(log), flush=True)


# ============================================================================
# EVALUATE
# ============================================================================


def summarize(rows):
    result = {}
    for name in sorted({r["combination"] for r in rows}):
        group = [r for r in rows if r["combination"] == name]
        regions = {}
        for region in ("ET", "TC", "WT"):
            cases = [r for r in group if r["region"] == region]
            hd = [r["hd95_mm"] for r in cases if np.isfinite(r["hd95_mm"])]
            sens = [r["sensitivity"] for r in cases if r["sensitivity"] is not None]
            regions[region] = {
                "n": len(cases),
                "dice": float(np.mean([r["dice"] for r in cases])),
                "sensitivity": float(np.mean(sens)) if sens else None,
                "sensitivity_defined_n": len(sens),
                "hd95_finite_mean_mm": float(np.mean(hd)) if hd else None,
                "hd95_infinite_n": len(cases) - len(hd),
            }
        result[name] = {
            "regions": regions,
            "mean_dice": float(np.mean([r["dice"] for r in group])),
            "n_cases": len(group) // 3,
            "available_modalities": group[0]["k"],
        }
    by_k = {
        str(k): float(np.mean([r["dice"] for r in rows if r["k"] == k]))
        for k in range(1, 5)
        if any(r["k"] == k for r in rows)
    }
    return {
        "combinations": result,
        "mean_dice_by_k": by_k,
        "all_evaluated_case_combination_region_mean_dice": float(
            np.mean([r["dice"] for r in rows])
        )
        if rows
        else None,
        "note": "HD95 finite mean is reported together with failure count; infinities are not silently treated as successes.",
    }


def evaluate_main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--splits", required=True)
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--output", required=True)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--backend", choices=["torch", "cuda"])
    p.add_argument("--full-only", action="store_true")
    p.add_argument("--overlap", type=float, default=0.5)
    a = p.parse_args()
    torch.set_num_threads(4)
    model, saved = load_checkpoint(a.checkpoint, a.device, a.backend)
    cases = select_cases(a.manifest, a.splits, a.fold, "test")
    training = set(saved.get("train_ids", [])) | set(saved.get("val_ids", []))
    if any(c["id"] in training for c in cases):
        raise ValueError("test patients overlap checkpoint train/val patients")
    patch = tuple(saved["training"]["patch_size"])
    rows = []
    skipped = []
    combos = [(1, 1, 1, 1)] if a.full_only else modality_combinations()
    for case in cases:
        item = load_case(case)
        for combination in combos:
            if any(
                requested and not actual
                for requested, actual in zip(combination, item["available"])
            ):
                skipped.append({"id": item["id"], "combination": list(combination)})
                continue
            mask = torch.tensor(combination, dtype=torch.float32)
            prediction = (
                sliding_window(
                    model,
                    torch.from_numpy(item["image"]),
                    mask,
                    patch,
                    a.overlap,
                    a.device,
                )
                .argmax(0)
                .numpy()
            )
            for region, metrics in evaluate_regions(
                prediction, item["target"], item["spacing"]
            ).items():
                rows.append(
                    {
                        "id": item["id"],
                        "fold": a.fold,
                        "combination": "+".join(
                            m for m, v in zip(MODALITIES, combination) if v
                        ),
                        "k": sum(combination),
                        "region": region,
                        **metrics,
                    }
                )
        print("Evaluated", item["id"], flush=True)
    if not rows:
        raise ValueError("no feasible evaluation combinations")
    output = Path(a.output)
    output.mkdir(parents=True, exist_ok=True)
    with (output / "cases.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    save_json(output / "summary.json", dict(summarize(rows), skipped=skipped))
    print("Saved", len(rows), "region-level rows")


# ============================================================================
# PREDICT
# ============================================================================


def predict_main():
    p = argparse.ArgumentParser(
        description="Predict NIfTI segmentations with original affine and BraTS labels"
    )
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--backend", choices=["torch", "cuda"])
    p.add_argument(
        "--modalities",
        nargs="+",
        choices=MODALITIES,
        help="Defaults to actually available modalities",
    )
    a = p.parse_args()
    import nibabel as nib

    model, saved = load_checkpoint(a.checkpoint, a.device, a.backend)
    output = Path(a.output)
    output.mkdir(parents=True, exist_ok=True)
    for case in read_manifest(a.manifest):
        item = load_case(case, require_label=False)
        mask = item["available"].copy()
        if a.modalities:
            requested = np.asarray([m in a.modalities for m in MODALITIES], np.float32)
            if np.any(requested > mask):
                raise ValueError("requested an unobserved modality for " + item["id"])
            mask = requested
        result = sliding_window(
            model,
            torch.from_numpy(item["image"]),
            torch.from_numpy(mask),
            tuple(saved["training"]["patch_size"]),
            0.5,
            a.device,
        )
        label = result.argmax(0).numpy().astype(np.uint8)
        label[label == 3] = 4
        image = nib.Nifti1Image(label, item["affine"])
        image.set_data_dtype(np.uint8)
        nib.save(image, output / (item["id"] + "_seg.nii.gz"))
        print("Saved", item["id"], flush=True)


BUILTIN_CONFIGS = {
    "reference": {
        "model": {
            "width": 24,
            "heads": 2,
            "state_size": 8,
            "frequency_scales": 2,
            "threshold": 0.55,
            "missing_threshold_shift": 0.2,
            "backend": "torch",
        },
        "training": {
            "seed": 2026,
            "patch_size": [96, 96, 96],
            "batch_size": 1,
            "epochs": 300,
            "workers": 4,
            "cpu_threads": 4,
            "samples_per_case": 4,
            "amp": True,
            "lr": 0.0002,
            "min_lr": 1e-06,
            "weight_decay": 0.01,
            "modality_drop": 0.25,
            "patch_drop": 0.1,
            "lambda_contribution": 0.1,
            "lambda_geometry": 0.01,
            "lambda_budget": 0.05,
            "rho_min": 0.45,
            "rho_max": 0.85,
            "aux_warmup_epochs": 10,
        },
    },
    "smoke": {
        "model": {
            "width": 12,
            "heads": 1,
            "state_size": 2,
            "frequency_scales": 2,
            "backend": "torch",
        },
        "training": {
            "seed": 2026,
            "patch_size": [8, 8, 8],
            "batch_size": 2,
            "epochs": 1,
            "workers": 0,
            "cpu_threads": 2,
            "samples_per_case": 1,
            "amp": False,
            "lr": 0.001,
            "min_lr": 1e-05,
            "weight_decay": 0.01,
            "modality_drop": 0.25,
            "patch_drop": 0.1,
            "lambda_contribution": 0.1,
            "lambda_geometry": 0.01,
            "lambda_budget": 0.05,
            "rho_min": 0.45,
            "rho_max": 0.85,
            "aux_warmup_epochs": 1,
        },
    },
}


def load_config(name):
    """Return a fresh built-in configuration, or read an explicit JSON file."""
    if name in BUILTIN_CONFIGS:
        return copy.deepcopy(BUILTIN_CONFIGS[name])
    return json.loads(Path(name).read_text(encoding="utf-8"))


def build_mmahh_mamba(**kwargs):
    """Construct the network; kwargs are fields of ModelConfig."""
    return MMAHHMamba(ModelConfig(**kwargs))


def config_main():
    p = argparse.ArgumentParser(
        description="Export an editable built-in training configuration"
    )
    p.add_argument("--preset", choices=list(BUILTIN_CONFIGS), default="reference")
    p.add_argument("--output", required=True)
    a = p.parse_args()
    save_json(a.output, load_config(a.preset))
    print("Saved configuration to", a.output)


def demo_main():
    p = argparse.ArgumentParser(
        description="Synthetic shape / gradient / missing-modality check"
    )
    p.add_argument("--image-size", type=int, default=16)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--backend", choices=["torch", "cuda"], default="torch")
    p.add_argument("--backward", action="store_true")
    p.add_argument(
        "--all-modalities", action="store_true", help="Check all 15 nonempty subsets"
    )
    a = p.parse_args()
    seed_all(2026)
    torch.set_num_threads(2)
    if a.image_size < 4 or a.batch_size < 1:
        p.error("image size >=4 and batch size >=1 required")
    model = build_mmahh_mamba(backend=a.backend).to(a.device)
    x = torch.randn(
        a.batch_size, 4, a.image_size, a.image_size, a.image_size, device=a.device
    )
    available = torch.ones(a.batch_size, 4, device=a.device)
    model.train(a.backward)
    with torch.set_grad_enabled(a.backward):
        logits, aux = model(x, available, return_aux=True)
        if a.backward:
            target = torch.randint(
                4,
                (a.batch_size, a.image_size, a.image_size, a.image_size),
                device=a.device,
            )
            geo, budget = auxiliary_losses(aux)
            loss = segmentation_loss(logits, target) + 0.01 * geo + 0.05 * budget
            loss.backward()
            assert all(
                torch.isfinite(p.grad).all()
                for p in model.parameters()
                if p.grad is not None
            )
            print("Backward passed; synthetic loss:", float(loss.detach()))
    print("Input:", list(x.shape), "Logits:", list(logits.shape))
    print("Parameters:", sum(p.numel() for p in model.parameters()))
    if a.all_modalities:
        model.eval()
        with torch.no_grad():
            for combination in modality_combinations():
                mask = torch.tensor(combination, device=a.device, dtype=x.dtype)[
                    None
                ].expand(a.batch_size, -1)
                y = model(x, mask)
                poisoned = torch.where(
                    mask[:, :, None, None, None].bool(),
                    x,
                    torch.full_like(x, float("nan")),
                )
                torch.testing.assert_close(
                    model(poisoned, mask), y, atol=5e-5, rtol=1e-4
                )
                assert torch.isfinite(y).all()
        print("All 15 subsets passed; absent-channel invariance verified.")


def smoke_main():
    p = argparse.ArgumentParser(
        description="Synthetic training -> checkpoint -> 15-subset evaluation"
    )
    p.add_argument("--device", default="cpu")
    p.add_argument("--output", default="runs/smoke")
    a = p.parse_args()
    out = Path(a.output).resolve()
    folder = out / "synthetic"
    folder.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(2026)
    cases = []
    coords = np.stack(np.meshgrid(*[np.arange(8)] * 3, indexing="ij"))
    for i in range(10):
        center = np.array([4, 4, 4]) + rng.integers(-1, 2, 3)
        distance = ((coords - center[:, None, None, None]) ** 2).sum(0)
        label = np.zeros((8, 8, 8), np.uint8)
        label[distance < 12] = 2
        label[distance < 6] = 1
        label[distance < 2] = 4
        image = np.stack(
            [
                rng.normal(0, 0.1, label.shape)
                + 0.5 * (label > 0)
                + m * 0.2 * (label == 4)
                for m in range(4)
            ]
        ).astype(np.float32)
        name = "synthetic_%02d" % i
        np.savez_compressed(
            folder / (name + ".npz"),
            image=image,
            label=label,
            spacing=np.ones(3),
            affine=np.eye(4),
        )
        cases.append({"id": name, "npz": name + ".npz"})
    manifest = folder / "manifest.json"
    save_json(manifest, {"modalities": list(MODALITIES), "cases": cases})
    script = Path(__file__).resolve()

    def run(*args):
        subprocess.run([sys.executable, str(script), *map(str, args)], check=True)

    run("splits", "--manifest", manifest, "--output", out / "splits.json")
    run(
        "train",
        "--config",
        "smoke",
        "--manifest",
        manifest,
        "--splits",
        out / "splits.json",
        "--device",
        a.device,
        "--output",
        out / "train",
        "--max-steps",
        "2",
    )
    run(
        "evaluate",
        "--checkpoint",
        out / "train/best.pt",
        "--manifest",
        manifest,
        "--splits",
        out / "splits.json",
        "--device",
        a.device,
        "--output",
        out / "eval",
    )
    report = json.loads((out / "eval/summary.json").read_text(encoding="utf-8"))
    assert len(report["combinations"]) == 15
    assert all(r["n_cases"] == 2 for r in report["combinations"].values())
    print("PASS: synthetic training, checkpoint reload, and all 15 subsets.")


def main():
    commands = {
        "demo": demo_main,
        "smoke": smoke_main,
        "prepare": prepare_main,
        "splits": splits_main,
        "train": train_main,
        "evaluate": evaluate_main,
        "predict": predict_main,
        "config": config_main,
    }
    if len(sys.argv) == 1:
        demo_main()
        return
    if sys.argv[1] in ("-h", "--help"):
        print(
            "Usage: python MMAHH-Mamba.py {demo,smoke,prepare,splits,train,evaluate,predict,config} [options]"
        )
        print("Run any command with --help for options. No command runs a shape demo.")
        return
    name = sys.argv.pop(1)
    if name not in commands:
        raise SystemExit("Unknown command: " + name)
    commands[name]()


if __name__ == "__main__":
    main()
