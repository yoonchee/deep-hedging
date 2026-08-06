"""Tests for the Deep Hedging policy pipeline (policy + environment + CVaR)."""

import pytest
import torch

from environment.market_env import MarketEnvironment
from generator.market_gan import Generator
from loss.cvar import CVaRLoss
from policy.hedging_agent import HedgingAgent, RecurrentHedgingAgent
from policy.train_policy import PolicyTrainer


def test_hedging_agent_output_shape_and_range() -> None:
    batch_size = 32
    agent = HedgingAgent(hidden_dim=16, num_hidden_layers=2)

    state = torch.randn(batch_size, HedgingAgent.STATE_DIM)
    delta_t = agent(state)

    assert delta_t.shape == (batch_size, 1)
    assert torch.all(delta_t >= 0.0) and torch.all(delta_t <= 1.0)


def test_hedging_agent_strike_normalization_matches_equivalent_moneyness() -> None:
    # An agent hedging S~100 against strike=100 should behave the same as an
    # agent hedging S~1 against strike=1, given the same weights -- i.e. the
    # network should only ever see moneyness-scale inputs internally.
    torch.manual_seed(0)
    agent_raw = HedgingAgent(hidden_dim=16, num_hidden_layers=2, strike=1.0)
    agent_scaled = HedgingAgent(hidden_dim=16, num_hidden_layers=2, strike=100.0)
    agent_scaled.load_state_dict(agent_raw.state_dict())

    delta_prev = torch.rand(8, 1)
    time_to_maturity = torch.rand(8, 1) * 0.08
    implied_vol = torch.full((8, 1), 0.15)

    S_normalized = torch.rand(8, 1) * 0.4 + 0.8  # moneyness in [0.8, 1.2]
    state_raw = torch.cat([S_normalized, delta_prev, time_to_maturity, implied_vol], dim=-1)
    state_scaled = torch.cat(
        [S_normalized * 100.0, delta_prev, time_to_maturity, implied_vol], dim=-1
    )

    delta_raw = agent_raw(state_raw)
    delta_scaled = agent_scaled(state_scaled)

    assert torch.allclose(delta_raw, delta_scaled, atol=1e-5)


def test_hedging_agent_default_strike_is_backward_compatible_noop() -> None:
    # strike=1.0 (the default) must be a pure no-op: S_t / 1.0 == S_t.
    torch.manual_seed(0)
    agent = HedgingAgent(hidden_dim=16, num_hidden_layers=2)
    state = torch.rand(8, HedgingAgent.STATE_DIM) + 0.5

    with torch.no_grad():
        expected = agent.net(state)
    delta_t = agent(state)

    assert torch.allclose(delta_t, torch.sigmoid(expected))


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


@pytest.mark.parametrize("cell_type", ["rnn", "lstm", "gru"])
def test_recurrent_agent_output_shape_and_range(cell_type: str) -> None:
    batch_size, seq_len = 8, 10
    agent = RecurrentHedgingAgent(cell_type=cell_type, hidden_dim=16, num_layers=1)

    prices = torch.rand(batch_size, seq_len, 1) + 0.5
    delta_path = agent(prices)

    assert delta_path.shape == (batch_size, seq_len - 1, 1)
    assert torch.all(delta_path >= 0.0) and torch.all(delta_path <= 1.0)


def test_recurrent_agent_rejects_unknown_cell_type() -> None:
    with pytest.raises(ValueError):
        RecurrentHedgingAgent(cell_type="transformer")


def test_recurrent_agent_output_head_matches_paper_node_counts() -> None:
    # Kim (2021)'s stated "128, 64, 64, 1" node counts, read as
    # RNN(128) -> FC(64) -> FC(64) -> 1 since nn.RNN/LSTM/GRU require a
    # single hidden_size per recurrent layer.
    batch_size, seq_len = 8, 10
    agent = RecurrentHedgingAgent(
        cell_type="gru", hidden_dim=128, num_layers=1, output_hidden_dims=[64, 64]
    )

    prices = torch.rand(batch_size, seq_len, 1) + 0.5
    delta_path = agent(prices)

    assert delta_path.shape == (batch_size, seq_len - 1, 1)
    assert torch.all(delta_path >= 0.0) and torch.all(delta_path <= 1.0)
    linear_layers = [m for m in agent.output_layer if isinstance(m, torch.nn.Linear)]
    assert [layer.out_features for layer in linear_layers] == [64, 64, 1]


def test_recurrent_agent_strike_normalization_matches_equivalent_moneyness() -> None:
    torch.manual_seed(0)
    agent_raw = RecurrentHedgingAgent(cell_type="gru", hidden_dim=16, num_layers=1, strike=1.0)
    agent_scaled = RecurrentHedgingAgent(cell_type="gru", hidden_dim=16, num_layers=1, strike=100.0)
    agent_scaled.load_state_dict(agent_raw.state_dict())

    prices_normalized = torch.rand(8, 10, 1) * 0.4 + 0.8  # moneyness in [0.8, 1.2]
    delta_raw = agent_raw(prices_normalized)
    delta_scaled = agent_scaled(prices_normalized * 100.0)

    assert torch.allclose(delta_raw, delta_scaled, atol=1e-5)


def test_recurrent_agent_rnn_input_is_standardized_log_moneyness() -> None:
    # Captures the actual tensor fed into self.rnn via a forward hook and
    # checks it matches log(S_t/K) / (implied_vol * sqrt(T)) exactly -- a
    # regression test protecting the fix for the DC-dominance bug (raw S/K
    # has ~3% signal buried under a ~1.0 constant offset for realistic
    # option params; this transform is what makes the RNN's input O(1)-scaled
    # instead of dominated by that offset). See RESULTS.md.
    strike, implied_vol, time_to_maturity = 100.0, 0.15, 1.0 / 12.0
    agent = RecurrentHedgingAgent(
        cell_type="gru",
        hidden_dim=8,
        num_layers=1,
        strike=strike,
        implied_vol=implied_vol,
        time_to_maturity=time_to_maturity,
    )

    captured = {}

    def hook(module, args):
        captured["rnn_input"] = args[0]

    agent.rnn.register_forward_pre_hook(hook)

    prices = torch.tensor([[[95.0], [100.0], [105.0], [110.0]]])  # [1, 4, 1]
    agent(prices)

    expected = torch.log(prices[:, :-1, :] / strike) / (implied_vol * time_to_maturity**0.5)
    assert torch.allclose(captured["rnn_input"], expected, atol=1e-6)


def test_recurrent_agent_moneyness_input_is_order_one_scale_for_realistic_params() -> None:
    # The regression this fix targets: raw S/K for Part I's params (S0=K=100,
    # vol=0.15, T=1/12) has std ~0.03 (measured directly on real GBM paths --
    # see RESULTS.md) -- a signal 97% dominated by a constant DC offset of
    # 1.0. The standardized log-moneyness transform must turn this into an
    # O(1)-scaled quantity instead.
    torch.manual_seed(0)
    strike, implied_vol, time_to_maturity = 100.0, 0.15, 1.0 / 12.0
    agent = RecurrentHedgingAgent(
        cell_type="gru", hidden_dim=8, num_layers=1, strike=strike,
        implied_vol=implied_vol, time_to_maturity=time_to_maturity,
    )

    # GBM-like price path around S0=100 at this vol/maturity, matching
    # replicate_part1.py's actual training distribution.
    log_returns = torch.cumsum(torch.randn(500, 29) * implied_vol * (time_to_maturity / 29) ** 0.5, dim=1)
    prices = 100.0 * torch.exp(torch.cat([torch.zeros(500, 1), log_returns], dim=1)).unsqueeze(-1)

    captured = {}
    agent.rnn.register_forward_pre_hook(lambda module, args: captured.update(rnn_input=args[0]))
    agent(prices)

    # Raw S/K would have std ~0.03 (the bug); standardized log-moneyness
    # should land within an order of magnitude of 1.0.
    assert 0.1 < captured["rnn_input"].std().item() < 10.0


@pytest.mark.parametrize("cell_type", ["rnn", "lstm", "gru"])
def test_recurrent_agent_gradient_propagates_to_every_price_timestep(cell_type: str) -> None:
    # Same rationale as the stepwise HedgingAgent test above, but for a
    # sequence policy evaluated through MarketEnvironment's vectorized
    # sequence_policy=True rollout: every S_t should still receive gradient.
    torch.manual_seed(0)
    batch_size, seq_len = 8, 10
    agent = RecurrentHedgingAgent(cell_type=cell_type, hidden_dim=16, num_layers=1)
    env = MarketEnvironment(strike=1.0, proportional_fee=0.01, dt=1.0)

    prices = (torch.rand(batch_size, seq_len, 1) + 0.5).requires_grad_(True)
    wealth = env.simulate(agent, prices, implied_vol=0.2, sequence_policy=True)
    wealth.sum().backward()

    assert prices.grad is not None
    grad_per_step = prices.grad.abs().sum(dim=(0, 2))
    assert grad_per_step.shape == (seq_len,)
    assert torch.all(grad_per_step > 0), grad_per_step


@pytest.mark.parametrize("cell_type", ["rnn", "lstm", "gru"])
def test_recurrent_agent_gradient_propagates_to_all_policy_parameters(cell_type: str) -> None:
    torch.manual_seed(0)
    batch_size, seq_len = 16, 12

    policy = RecurrentHedgingAgent(cell_type=cell_type, hidden_dim=16, num_layers=1)
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
        sequence_policy=True,
    )
    stats = trainer.train_step(batch_size, seq_len)

    assert isinstance(stats["loss"], float)
    for name, param in policy.named_parameters():
        assert param.grad is not None, f"no gradient for {name}"
        assert torch.any(param.grad != 0), f"zero gradient for {name}"
