import torch

from qecml.decoders.neural.mini_alphaqubit import MiniAlphaQubitDecoder
from qecml.decoders.neural.mlp import FlatMLPDecoder
from qecml.decoders.neural.temporal_cnn import TemporalCNNDecoder


def _assert_backward(model, x, y):
    logits = model(x)
    assert logits.shape == y.shape
    loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, y)
    loss.backward()
    grad_norm = sum(
        float(p.grad.detach().abs().sum()) for p in model.parameters() if p.grad is not None
    )
    assert grad_norm > 0


def test_mlp_forward_backward():
    model = FlatMLPDecoder(num_detectors=12, num_observables=1, hidden_dim=16)
    _assert_backward(model, torch.randint(0, 2, (8, 12)).float(), torch.zeros(8, 1))


def test_temporal_cnn_forward_backward():
    model = TemporalCNNDecoder(num_sites=5, num_features=1, hidden_dim=16)
    _assert_backward(model, torch.randint(0, 2, (8, 3, 5, 1)).float(), torch.zeros(8, 1))


def test_mini_alphaqubit_forward_backward_and_tiny_overfit():
    torch.manual_seed(0)
    x = torch.randint(0, 2, (32, 3, 5, 1)).float()
    y = (x[:, :, 0, :].sum(dim=(1, 2), keepdim=False) % 2).reshape(32, 1).float()
    model = MiniAlphaQubitDecoder(
        num_sites=5,
        num_features=1,
        hidden_dim=16,
        num_layers=1,
        num_heads=4,
        dropout=0.0,
    )
    _assert_backward(model, x[:8], y[:8])
    opt = torch.optim.AdamW(model.parameters(), lr=0.02)
    first = None
    last = None
    for step in range(25):
        opt.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(model(x), y)
        if step == 0:
            first = float(loss.detach())
        loss.backward()
        opt.step()
        last = float(loss.detach())
    assert first is not None and last is not None
    assert last < first
