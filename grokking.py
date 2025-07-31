import argparse
import math
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Transformer(nn.Module):
    # One layer, no LayerNorm, input is "a b =" and we read the logits at "=".
    def __init__(self, p, d=64, heads=4, d_mlp=256):
        super().__init__()
        self.heads = heads
        self.embed = nn.Parameter(torch.randn(p + 1, d) / math.sqrt(d))
        self.pos = nn.Parameter(torch.randn(3, d) / math.sqrt(d))
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.out = nn.Linear(d, d, bias=False)
        self.mlp_in = nn.Linear(d, d_mlp)
        self.mlp_out = nn.Linear(d_mlp, d)
        self.unembed = nn.Linear(d, p, bias=False)

    def forward(self, x):
        # Only the "=" position feeds the output, so the query and MLP run there alone.
        h = self.embed[x] + self.pos
        B, T, D = h.shape
        q, k, v = self.qkv(h).view(B, T, 3, self.heads, D // self.heads).permute(2, 0, 3, 1, 4)
        att = F.scaled_dot_product_attention(q[:, :, -1:], k, v)
        h = h[:, -1] + self.out(att.reshape(B, D))
        h = h + self.mlp_out(F.relu(self.mlp_in(h)))
        return self.unembed(h)


def make_data(p, frac, seed):
    a, b = torch.meshgrid(torch.arange(p), torch.arange(p), indexing="ij")
    x = torch.stack([a.flatten(), b.flatten(), torch.full((p * p,), p)], 1)
    y = (x[:, 0] + x[:, 1]) % p
    perm = torch.randperm(p * p, generator=torch.Generator().manual_seed(seed))
    n = int(frac * p * p)
    return x, y, perm[:n], perm[n:]












def train(p=59, frac=0.4, steps=5000, lr=3e-3, wd=1.0, seed=0, log_every=100):
    torch.manual_seed(seed)
    x, y, tr, te = make_data(p, frac, seed)
    model = Transformer(p)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd, betas=(0.9, 0.98))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 10))
    hist = {"step": [], "train_loss": [], "test_loss": [], "train_acc": [], "test_acc": [], "logits": []}
    for step in range(steps + 1):
        if step % log_every == 0 or step == steps:
            with torch.no_grad():
                logits = model(x)
            for split, idx in (("train", tr), ("test", te)):
                hist[f"{split}_loss"].append(F.cross_entropy(logits[idx], y[idx]).item())
                hist[f"{split}_acc"].append((logits[idx].argmax(1) == y[idx]).float().mean().item())
            hist["step"].append(step)
            hist["logits"].append(logits)
        if step == steps:
            break
        loss = F.cross_entropy(model(x[tr]), y[tr])
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
    return model, hist, (x, y, tr, te)


def first_step(hist, key, thresh):
    return next((s for s, a in zip(hist["step"], hist[key]) if a >= thresh), None)




