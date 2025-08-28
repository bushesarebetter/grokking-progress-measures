import math

import torch
import torch.nn.functional as F

from grokking import embed_freq_norms, fourier_basis, key_frequencies, make_data, progress_measures, train


def test_fourier_basis_is_orthonormal():
    B = fourier_basis(23)
    assert B.shape == (23, 23)
    assert torch.allclose(B @ B.T, torch.eye(23), atol=1e-5)


def test_embedding_with_one_frequency_is_found():
    p = 23
    n = torch.arange(p).float()
    W_E = torch.randn(p, 16) * 0.01
    W_E[:, 0] += torch.cos(2 * math.pi * 5 * n / p)
    W_E[:, 1] += torch.sin(2 * math.pi * 5 * n / p)
    norms = embed_freq_norms(W_E)
    assert norms.argmax().item() + 1 == 5
    assert key_frequencies(norms) == [5]


def test_restricted_and_excluded_loss_on_ideal_logits():
    # Logits that implement the paper's algorithm with a single frequency.
    p, k = 23, 4
    x, y, tr, _ = make_data(p, 0.3, 0)
    c = torch.arange(p)
    logits = 10 * torch.cos(2 * math.pi * k * (x[:, :1] + x[:, 1:2] - c) / p)
    full = F.cross_entropy(logits, y).item()
    restricted, excluded = progress_measures(logits, y, tr, p, [k])
    assert abs(restricted - full) < 1e-3
    assert abs(excluded - math.log(p)) < 1e-3
    restricted_wrong, _ = progress_measures(logits, y, tr, p, [k + 1])
    assert abs(restricted_wrong - math.log(p)) < 1e-3


def test_quick_training_loss_decreases():
    _, hist, _ = train(p=23, steps=150, log_every=50)
    assert hist["train_loss"][-1] < 0.5 * hist["train_loss"][0]
