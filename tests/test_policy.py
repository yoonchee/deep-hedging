"""Tests for the Deep Hedging policy pipeline (policy + environment + CVaR)."""

import torch

from environment.market_env import MarketEnvironment
from generator.market_gan import Generator
from loss.cvar import CVaRLoss
from policy.hedging_agent import HedgingAgent
from policy.train_policy import PolicyTrainer


def test_hedging_agent_output_shape_and_range() -> None:
    batch_size = 32
    agent = HedgingAgent(hidden_dim=16, num_hidden_layers=2)

    state = torch.randn(batch_size, HedgingAgent.STATE_DIM)
    delta_t = agent(state)

    assert delta_t.shape == (batch_size, 1)
    assert torch.all(delta_t >= 0.0) and torch.all(delta_t <= 1.0)


def test_market_environment_wealth_shape() -> None:
    batch_size, seq_len = 8, 10
    agent = HedgingAgent(hidden_dim=16, num_hidden_layers=2)
    env = MarketEnvironment(strike=1.0, proportional_fee=0.01, dt=1.0)

    prices = torch.rand(batch_size, seq_len, 1) + 0.5  # strictly positive prices
    wealth = env.simulate(agent, prices, implied_vol=0.2)

    assert wealth.shape == (batch_size,)
    assert torch.all(torch.isfinite(wealth))


def test_gradient_propagates_to_every_price_timestep() -> None:
    # Every S_t (0..N) should receive a nonzero gradient from Wealth_T: S_0
    # through the first hedge decision and P&L term, intermediate S_t through
    # both the previous hedge P&L and the current step's state/cost, and S_N
    # through both the last hedge P&L and the option payoff. This confirms
    # gradients flow through the full recurrent rollout, not just the last step.
    torch.manual_seed(0)
    batch_size, seq_len = 8, 10
    agent = HedgingAgent(hidden_dim=16, num_hidden_layers=2)
    env = MarketEnvironment(strike=1.0, proportional_fee=0.01, dt=1.0)

    prices = (torch.rand(batch_size, seq_len, 1) + 0.5).requires_grad_(True)
    wealth = env.simulate(agent, prices, implied_vol=0.2)
    wealth.sum().backward()

    assert prices.grad is not None
    # [Batch, Time_Steps, 1] -> [Time_Steps] (gradient magnitude per step)
    grad_per_step = prices.grad.abs().sum(dim=(0, 2))
    assert grad_per_step.shape == (seq_len,)
    assert torch.all(grad_per_step > 0), grad_per_step


def test_gradient_propagates_to_all_policy_parameters() -> None:
    # End-to-end check: after one CVaR training step over synthetic
    # market_gan.py paths, every policy parameter must have received a
    # nonzero gradient, confirming the shared policy network is trained
    # using signal accumulated across all time steps of the rollout.
    torch.manual_seed(0)
    batch_size, seq_len = 16, 12

    policy = HedgingAgent(hidden_dim=16, num_hidden_layers=2)
    generator = Generator(noise_dim=8, hidden_dim=16, num_layers=1)
    environment = MarketEnvironment(strike=1.0, proportional_fee=0.01, dt=1.0)
    cvar_loss = CVaRLoss(alpha=0.95)

    trainer = PolicyTrainer(
        policy,
        environment,
        generator,
        cvar_loss,
        implied_vol=0.2,
        lr=1e-2,
        device=torch.device("cpu"),
    )
    stats = trainer.train_step(batch_size, seq_len)

    assert isinstance(stats["loss"], float)
    for name, param in policy.named_parameters():
        assert param.grad is not None, f"no gradient for {name}"
        assert torch.any(param.grad != 0), f"zero gradient for {name}"
