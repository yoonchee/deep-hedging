"""Tests for the Deep Hedging policy pipeline (policy + environment + CVaR)."""

import pytest
import torch

from common.black_scholes import BlackScholesDeltaPolicy, black_scholes_call_price
from environment.market_env import MarketEnvironment, estimate_premium_monte_carlo
from generator.data import sample_real_prices
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


def test_black_scholes_call_price_matches_hand_computed_value_for_part1_params() -> None:
    # S0=K=100, vol=0.15, T=1/12 (Part I's exact params); hand-derived
    # reference d1 ~ 0.02165 -> C0 ~ 1.72, independently cross-checked
    # against the measured Black-Scholes mean PnL of -1.729 in
    # results/part1_replication/part1_summary.json (see RESULTS.md).
    price = black_scholes_call_price(S0=100.0, K=100.0, tau=1.0 / 12.0, sigma=0.15)
    assert price == pytest.approx(1.72, abs=0.01)


def test_black_scholes_call_price_reduces_to_intrinsic_value_at_near_zero_volatility() -> None:
    price = black_scholes_call_price(S0=110.0, K=100.0, tau=1.0, sigma=1e-6)
    assert price == pytest.approx(10.0, abs=1e-3)


def test_estimate_premium_monte_carlo_matches_closed_form_for_gbm() -> None:
    torch.manual_seed(0)
    s0, strike, vol, seq_len = 100.0, 100.0, 0.15, 30
    dt = (1.0 / 12.0) / (seq_len - 1)

    def sample_prices(batch_size: int) -> torch.Tensor:
        return sample_real_prices(batch_size, seq_len, s0=s0, vol=vol, dt=dt)

    premium = estimate_premium_monte_carlo(sample_prices, strike=strike, num_paths=200_000)

    # Closed-form Black-Scholes price at these params (Part I's own setup) is ~1.727.
    assert premium == pytest.approx(1.727, abs=0.05)


def test_estimate_premium_monte_carlo_is_nonnegative() -> None:
    def sample_prices(batch_size: int) -> torch.Tensor:
        return sample_real_prices(batch_size, seq_len=10, s0=1.0, vol=0.3)

    premium = estimate_premium_monte_carlo(sample_prices, strike=1.0, num_paths=10_000)

    assert premium >= 0.0


def test_market_environment_premium_defaults_to_zero_and_is_backward_compatible() -> None:
    batch_size, seq_len = 8, 10
    agent = HedgingAgent(hidden_dim=16, num_hidden_layers=2)
    torch.manual_seed(0)
    prices = torch.rand(batch_size, seq_len, 1) + 0.5

    env_default = MarketEnvironment(strike=1.0, proportional_fee=0.01, dt=1.0)
    env_explicit_zero = MarketEnvironment(strike=1.0, proportional_fee=0.01, dt=1.0, premium=0.0)

    torch.manual_seed(1)
    wealth_default = env_default.simulate(agent, prices, implied_vol=0.2)
    torch.manual_seed(1)
    wealth_explicit_zero = env_explicit_zero.simulate(agent, prices, implied_vol=0.2)

    assert torch.allclose(wealth_default, wealth_explicit_zero)


def test_market_environment_premium_shifts_wealth_by_exactly_premium() -> None:
    batch_size, seq_len = 8, 10
    agent = HedgingAgent(hidden_dim=16, num_hidden_layers=2)
    prices = torch.rand(batch_size, seq_len, 1) + 0.5
    premium = 1.72

    env_no_premium = MarketEnvironment(strike=1.0, proportional_fee=0.01, dt=1.0)
    env_with_premium = MarketEnvironment(strike=1.0, proportional_fee=0.01, dt=1.0, premium=premium)

    torch.manual_seed(2)
    wealth_no_premium = env_no_premium.simulate(agent, prices, implied_vol=0.2)
    torch.manual_seed(2)
    wealth_with_premium = env_with_premium.simulate(agent, prices, implied_vol=0.2)

    assert torch.allclose(wealth_with_premium, wealth_no_premium + premium, atol=1e-6)


def test_market_environment_chunked_matches_unchunked() -> None:
    batch_size, seq_len = 37, 10  # deliberately not a multiple of chunk_size
    agent = HedgingAgent(hidden_dim=16, num_hidden_layers=2)
    env = MarketEnvironment(strike=1.0, proportional_fee=0.01, dt=1.0, premium=0.5)
    prices = torch.rand(batch_size, seq_len, 1) + 0.5

    wealth_unchunked, cost_unchunked = env.simulate_with_costs(agent, prices, implied_vol=0.2)
    wealth_chunked, cost_chunked = env.simulate_with_costs(
        agent, prices, implied_vol=0.2, chunk_size=10
    )

    assert torch.allclose(wealth_unchunked, wealth_chunked, atol=1e-6)
    assert torch.allclose(cost_unchunked, cost_chunked, atol=1e-6)


def test_market_environment_chunked_matches_unchunked_for_sequence_policy() -> None:
    batch_size, seq_len, cell_type = 23, 10, "lstm"
    agent = RecurrentHedgingAgent(cell_type=cell_type, hidden_dim=16, num_layers=1, strike=1.0)
    agent.eval()
    env = MarketEnvironment(strike=1.0, proportional_fee=0.01, dt=1.0, premium=0.5)
    prices = torch.rand(batch_size, seq_len, 1) + 0.5

    with torch.no_grad():
        wealth_unchunked = env.simulate(agent, prices, implied_vol=0.2, sequence_policy=True)
        wealth_chunked = env.simulate(
            agent, prices, implied_vol=0.2, sequence_policy=True, chunk_size=7
        )

    assert torch.allclose(wealth_unchunked, wealth_chunked, atol=1e-6)


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


def test_grad_clip_norm_actually_clips_the_applied_gradient() -> None:
    # This is the mechanism (a) fix (RESULTS.md): a single CVaR-amplified
    # gradient step can push HedgingAgent's sigmoid output into permanent
    # numerical saturation. grad_clip_norm exists specifically to cap step
    # size, but had no test at all before -- only real training runs
    # exercised it. Two trainers, identical seed/setup, differing only in
    # grad_clip_norm: the unclipped run's own pre-clip grad_norm confirms
    # the natural gradient is larger than the clip target (otherwise this
    # test wouldn't be exercising clipping at all), and the clipped run's
    # actual post-step parameter gradients (not the pre-clip grad_norm
    # PolicyTrainer returns) must have total norm at or below the target.
    clip_target = 1e-3

    def make_trainer(grad_clip_norm) -> PolicyTrainer:
        torch.manual_seed(0)
        policy = HedgingAgent(hidden_dim=16, num_hidden_layers=2)
        generator = Generator(noise_dim=8, hidden_dim=16, num_layers=1)
        environment = MarketEnvironment(strike=1.0, proportional_fee=0.01, dt=1.0)
        cvar_loss = CVaRLoss(alpha=0.95)
        return PolicyTrainer(
            policy, environment, generator, cvar_loss,
            implied_vol=0.2, lr=1e-2, device=torch.device("cpu"),
            grad_clip_norm=grad_clip_norm,
        )

    torch.manual_seed(0)
    unclipped = make_trainer(grad_clip_norm=None)
    stats_unclipped = unclipped.train_step(batch_size=16, seq_len=12)
    assert stats_unclipped["grad_norm"] > clip_target, (
        "natural gradient norm must exceed the clip target for this test to "
        "actually exercise clipping -- if this fails, raise clip_target"
    )

    torch.manual_seed(0)
    clipped = make_trainer(grad_clip_norm=clip_target)
    clipped.train_step(batch_size=16, seq_len=12)
    applied_grad_norm = torch.sqrt(
        sum((p.grad**2).sum() for p in clipped.policy.parameters() if p.grad is not None)
    )
    assert applied_grad_norm.item() <= clip_target * 1.001


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


def test_use_bs_baseline_gradient_still_flows_to_policy() -> None:
    # The Black-Scholes baseline is a fixed analytic function with no
    # trainable parameters -- gradients must still reach every policy
    # parameter when use_bs_baseline is enabled.
    torch.manual_seed(0)
    batch_size, seq_len = 16, 12

    policy = RecurrentHedgingAgent(cell_type="rnn", hidden_dim=16, num_layers=1)
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
        use_bs_baseline=True,
    )
    stats = trainer.train_step(batch_size, seq_len)

    assert isinstance(stats["loss"], float)
    for name, param in policy.named_parameters():
        assert param.grad is not None, f"no gradient for {name}"
        assert torch.any(param.grad != 0), f"zero gradient for {name}"


def test_use_bs_baseline_reports_raw_wealth_not_advantage() -> None:
    # mean_wealth must reflect the policy's actual raw wealth regardless of
    # use_bs_baseline, so results stay comparable across configurations.
    torch.manual_seed(0)
    batch_size, seq_len = 16, 10

    policy = HedgingAgent(hidden_dim=16, num_hidden_layers=2)
    generator = Generator(noise_dim=8, hidden_dim=16, num_layers=1)
    environment = MarketEnvironment(strike=1.0, proportional_fee=0.01, dt=1.0)
    initial_state = {k: v.clone() for k, v in policy.state_dict().items()}

    trainer_raw = PolicyTrainer(
        policy, environment, generator, CVaRLoss(alpha=0.95),
        implied_vol=0.2, lr=1e-2, device=torch.device("cpu"), use_bs_baseline=False,
    )
    torch.manual_seed(0)
    stats_raw = trainer_raw.train_step(batch_size, seq_len)

    policy2 = HedgingAgent(hidden_dim=16, num_hidden_layers=2)
    policy2.load_state_dict(initial_state)
    trainer_baseline = PolicyTrainer(
        policy2, environment, generator, CVaRLoss(alpha=0.95),
        implied_vol=0.2, lr=1e-2, device=torch.device("cpu"), use_bs_baseline=True,
    )
    torch.manual_seed(0)
    stats_baseline = trainer_baseline.train_step(batch_size, seq_len)

    # Same seed, same starting weights, same sampled batch -> raw policy
    # wealth (pre-update) should match regardless of what the loss trains on.
    assert stats_raw["mean_wealth"] == pytest.approx(stats_baseline["mean_wealth"], abs=1e-4)


def test_use_bs_baseline_reduces_loss_variance_across_batches() -> None:
    # The core empirical claim of the control-variate technique: across
    # independent batches, CVaR of (wealth - bs_wealth) should vary less
    # batch to batch than CVaR of raw wealth, since both wealths share the
    # same market-driven noise on the same price draw and that shared noise
    # cancels in the subtraction.
    torch.manual_seed(0)
    batch_size, seq_len = 500, 15

    policy = HedgingAgent(hidden_dim=16, num_hidden_layers=2)
    generator = Generator(noise_dim=8, hidden_dim=16, num_layers=1)
    environment = MarketEnvironment(strike=1.0, proportional_fee=0.01, dt=1.0)
    bs_policy = BlackScholesDeltaPolicy(strike=1.0)

    raw_losses = []
    advantage_losses = []
    for seed in range(20):
        torch.manual_seed(seed)
        with torch.no_grad():
            z = generator.sample_noise(batch_size, seq_len)
            prices = generator(z)
            wealth = environment.simulate(policy, prices, 0.2, sequence_policy=False)
            bs_wealth = environment.simulate(bs_policy, prices, 0.2, sequence_policy=False)

        raw_losses.append(CVaRLoss(alpha=0.95)(wealth).item())
        advantage_losses.append(CVaRLoss(alpha=0.95)(wealth - bs_wealth).item())

    raw_std = torch.tensor(raw_losses).std().item()
    advantage_std = torch.tensor(advantage_losses).std().item()
    assert advantage_std < raw_std
