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


def fourier_basis(p):
    # Orthonormal rows: constant, then cos/sin pairs for k = 1..(p-1)/2.
    n = torch.arange(p, dtype=torch.float64)
    rows = [torch.ones(p, dtype=torch.float64)]
    for k in range(1, p // 2 + 1):
        rows += [torch.cos(2 * math.pi * k * n / p), torch.sin(2 * math.pi * k * n / p)]
    basis = torch.stack(rows)
    return (basis / basis.norm(dim=1, keepdim=True)).float()


def embed_freq_norms(W_E):
    # Norm of the embedding along each frequency (cos and sin combined), k = 1..(p-1)/2.
    p = W_E.shape[0]
    coef = fourier_basis(p) @ W_E
    return (coef[1::2].pow(2).sum(1) + coef[2::2].pow(2).sum(1)).sqrt()


def key_frequencies(norms):
    # Frequencies holding well above their fair share of embedding energy.
    energy = norms.pow(2)
    return [k + 1 for k in torch.nonzero(energy > 3 * energy.mean()).flatten().tolist()]


def key_components(logits, p, freqs):
    # logits: (p*p, p) over all (a, b). Project onto cos(w(a+b)) and sin(w(a+b)) for each key w.
    s = (torch.arange(p)[:, None] + torch.arange(p)[None, :]).flatten().float()
    comp = torch.zeros_like(logits)
    for k in freqs:
        for v in (torch.cos(2 * math.pi * k * s / p), torch.sin(2 * math.pi * k * s / p)):
            v = v / v.norm()
            comp += v[:, None] * (v @ logits)[None, :]
    return comp


def progress_measures(logits, y, train_idx, p, freqs):
    comp = key_components(logits, p, freqs)
    restricted = logits.mean(0, keepdim=True) + comp
    excluded = logits - comp
    return (F.cross_entropy(restricted, y).item(),
            F.cross_entropy(excluded[train_idx], y[train_idx]).item())


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--p", type=int, default=59)
    ap.add_argument("--frac", type=float, default=0.4)
    ap.add_argument("--steps", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    if args.quick:
        args.steps = 500

    t0 = time.time()
    model, hist, (x, y, tr, te) = train(args.p, args.frac, args.steps, seed=args.seed)
    p = args.p
    norms = embed_freq_norms(model.embed.detach()[:p])
    freqs = key_frequencies(norms)
    measures = [progress_measures(l, y, tr, p, freqs) for l in hist["logits"]]
    restricted, excluded = map(list, zip(*measures))

    s_train, s_test = first_step(hist, "train_acc", 0.99), first_step(hist, "test_acc", 0.99)
    print(f"p={p} train={len(tr)} test={len(te)} steps={args.steps} time={time.time() - t0:.0f}s")
    print(f"train acc >=99% at step {s_train}, test acc >=99% at step {s_test}")
    print(f"final train acc {hist['train_acc'][-1]:.4f}  test acc {hist['test_acc'][-1]:.4f}  "
          f"test loss {hist['test_loss'][-1]:.2e}")
    print(f"key frequencies {freqs}, share of embedding energy "
          f"{norms[[k - 1 for k in freqs]].pow(2).sum() / norms.pow(2).sum():.3f}")
    print(f"final restricted loss {restricted[-1]:.2e}  excluded loss {excluded[-1]:.2f}  "
          f"(uniform guess = {math.log(p):.2f})")

    steps = hist["step"][1:]  # skip step 0 on log axes
    plt.figure(figsize=(6, 4))
    plt.plot(steps, hist["train_acc"][1:], label="train")
    plt.plot(steps, hist["test_acc"][1:], label="test")
    plt.xscale("log")
    plt.xlabel("step")
    plt.ylabel("accuracy")
    plt.title(f"(a + b) mod {p}, {args.frac:.0%} train")
    plt.legend()
    plt.tight_layout()
    plt.savefig("accuracy.png", dpi=120)

    plt.figure(figsize=(7, 3.5))
    ks = np.arange(1, len(norms) + 1)
    plt.bar(ks, norms.numpy(), color=["C3" if k in freqs else "C0" for k in ks])
    plt.xlabel("frequency k")
    plt.ylabel("norm of W_E component")
    plt.title("Embedding in the Fourier basis (red = key frequencies)")
    plt.tight_layout()
    plt.savefig("fourier_embed.png", dpi=120)

    plt.figure(figsize=(6.5, 4))
    plt.plot(steps, hist["train_loss"][1:], label="train loss")
    plt.plot(steps, hist["test_loss"][1:], label="test loss")
    plt.plot(steps, restricted[1:], label="restricted loss")
    plt.plot(steps, excluded[1:], label="excluded loss")
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("step")
    plt.ylabel("cross-entropy")
    plt.title(f"Progress measures (key freqs {freqs})")
    plt.legend()
    plt.tight_layout()
    plt.savefig("progress_measures.png", dpi=120)


if __name__ == "__main__":
    main()
