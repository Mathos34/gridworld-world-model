"""Unit tests for the VICReg-style regularizers."""
import torch

from src.losses import covariance_loss, variance_loss


def test_variance_loss_zero_on_unit_std():
    torch.manual_seed(0)
    z = torch.randn(512, 8)
    z = (z - z.mean(0)) / z.std(0)
    assert variance_loss(z).item() < 1e-3


def test_variance_loss_positive_on_collapsed_dim():
    z = torch.zeros(64, 4)
    z[:, 0] = torch.randn(64)
    loss = variance_loss(z).item()
    assert loss > 0.5


def test_covariance_loss_zero_on_diagonal():
    torch.manual_seed(0)
    z = torch.randn(2048, 4)
    z = (z - z.mean(0)) / z.std(0)
    assert covariance_loss(z).item() < 0.1


def test_covariance_loss_positive_on_correlated_dims():
    torch.manual_seed(0)
    base = torch.randn(512, 1)
    z = torch.cat([base, base, base + 0.01 * torch.randn(512, 1)], dim=1)
    assert covariance_loss(z).item() > 0.1
