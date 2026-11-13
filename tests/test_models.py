import torch

from gpumodel.models import MODELS, make_model
from gpumodel.traffic import LogicalTrafficMode


def test_model_shapes_and_backward():
    for name in MODELS:
        model = make_model(name)
        out = model(torch.randn(2, 3, 64, 64))
        assert out.shape == (2, 10)
        out.mean().backward()
        assert any(p.grad is not None for p in model.parameters())


def test_logical_traffic():
    x = torch.ones(4)
    with LogicalTrafficMode() as c:
        _ = x + x
    assert c.logical_bytes > 0
