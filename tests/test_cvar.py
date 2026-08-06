"""Tests for the CVaR / Expected Shortfall loss (src/loss/cvar.py)."""

import torch

from loss.cvar import CVaRLoss


def _optimize_h(
    loss_fn: CVaRLoss, x: torch.Tensor, steps: int = 1000, lr: float = 0.5
) -> float:
    """Optimize the learnable VaR threshold h to convergence (infimum over h).

    The objective is piecewise-linear in h with a kink at the optimum, so a
    constant learning rate (e.g. plain Adam) oscillates around the minimum
    indefinitely. An exponentially decaying learning rate lets the optimizer
    settle tightly onto the true infimum.
    """
    optimizer = torch.optim.Adam(loss_fn.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
    for _ in range(steps):
        optimizer.zero_grad()
        loss = loss_fn(x)
        loss.backward()
        optimizer.step()
        scheduler.step()
    with torch.no_grad():
        return loss_fn(x).item()


def test_cvar_zero_variance_matches_expected_value() -> None:
    # Constant P&L (zero variance): CVaR_alpha should equal -c, since the
    # worst-case tail is identical to the expected outcome (no dispersion).
    c = 2.5
    x = torch.full((256,), c)

    loss_fn = CVaRLoss(alpha=0.95)
    converged_loss = _optimize_h(loss_fn, x)

    assert abs(converged_loss - (-c)) < 1e-2


def test_cvar_equal_for_two_zero_variance_batches() -> None:
    # Two different constant-P&L batches with the same constant value should
    # converge to the same (equal) CVaR loss regardless of batch size.
    c = -1.0
    x1 = torch.full((64,), c)
    x2 = torch.full((512,), c)

    loss1 = _optimize_h(CVaRLoss(alpha=0.99), x1)
    loss2 = _optimize_h(CVaRLoss(alpha=0.99), x2)

    assert abs(loss1 - loss2) < 1e-2


def test_cvar_increases_monotonically_with_tail_losses() -> None:
    # Fix a benign "body" of outcomes and progressively worsen the tail
    # (bottom 5% of samples). CVaR_0.95 should strictly increase (worsen)
    # as the tail losses become more severe.
    torch.manual_seed(0)
    batch_size = 1000
    n_tail = int(batch_size * 0.05)

    tail_severities = [0.0, 1.0, 2.0, 5.0, 10.0]
    losses = []
    for severity in tail_severities:
        pnl = torch.ones(batch_size)
        pnl[:n_tail] = -severity  # worsening tail losses
        loss_fn = CVaRLoss(alpha=0.95)
        losses.append(_optimize_h(loss_fn, pnl))

    for earlier, later in zip(losses, losses[1:]):
        assert later > earlier
