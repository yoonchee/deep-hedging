# Deep Hedging

A PyTorch implementation of **Kim (2021), "Deep Hedging, Generative Adversarial
Networks, and Beyond"** (itself building on Buehler et al.'s 2018 Deep Hedging
paper and Yoon et al.'s TimeGAN), plus a validation/backtesting harness built
around it. Formulas are in [`math_spec.md`](math_spec.md); the full story of
what matches the paper, what doesn't, and why is in
[`RESULTS.md`](RESULTS.md).

## What's here

- A **WGAN-GP market generator** that learns to simulate synthetic asset price
  paths, trainable on either a synthetic GBM placeholder or real historical
  data (via `yfinance`), with an automatic **fidelity checker** that catches
  mode collapse, mean bias, and tail-shape (skew/kurtosis) mismatches before
  they cascade into policy training.
- **Four hedging policy architectures** trained by direct policy search to
  minimize CVaR of terminal wealth: a feed-forward MLP (Buehler et al.'s
  `delta_k = f(I_k, delta_{k-1})` formulation) and three genuine recurrent
  policies (RNN/LSTM/GRU) that consume the whole price path in one pass,
  matching the paper's own architecture comparison.
- A **differentiable CVaR loss** (Rockafellar-Uryasev) with a jointly-learned
  auxiliary threshold.
- A **stress-test backtester** comparing all four architectures against
  analytic Black-Scholes delta hedging under regime-switching volatility and
  transaction friction.
- A **frictionless Part I replication** reproducing the paper's controlled
  GBM experiment (no transaction costs) across three risk-aversion levels.

## Project structure

```
math_spec.md                  Core formulas: payoff, transaction cost, CVaR, WGAN-GP loss
src/
  common/stats.py             Shared skewness/excess-kurtosis helpers
  generator/
    market_gan.py             WGAN-GP Generator (GRU) + Discriminator (LSTM)
    data.py                   Real (yfinance) and synthetic GBM data sources
    train_gan.py              WGAN-GP training CLI
    validate.py               GAN fidelity checker (mode collapse / mean bias / tail shape)
  loss/cvar.py                Differentiable CVaR loss (learnable threshold h)
  policy/
    hedging_agent.py          HedgingAgent (MLP) + RecurrentHedgingAgent (RNN/LSTM/GRU)
    train_policy.py           Policy training CLI (single alpha or alpha sweep)
  environment/market_env.py   MarketEnvironment: wealth/transaction-cost simulation
  backtester/
    evaluate.py                Stress-test backtest vs. Black-Scholes
    replicate_part1.py         Frictionless Part I paper replication
    plotting.py                Shared chart library
tests/                         37 tests
results/                       Generated plots + JSON summaries (gitignored inputs: data/, checkpoints/)
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Quickstart

Each stage writes a checkpoint (to `checkpoints/`, gitignored) that the next
stage loads automatically.

```bash
# 1. Train the market generator. Synthetic (fast, offline, default):
python src/generator/train_gan.py --epochs 200
# ...or on real S&P 500 data (needs more epochs/higher lr to converge well):
python src/generator/train_gan.py --epochs 1500 --lr 3e-4 --data-source yfinance

# 2. Check it didn't collapse (also runs automatically after step 1)
python src/generator/validate.py --generator-checkpoint checkpoints/market_gan.pt

# 3. Train each policy architecture against the generator
python src/policy/train_policy.py --architecture mlp  --epochs 200
python src/policy/train_policy.py --architecture rnn  --epochs 200
python src/policy/train_policy.py --architecture lstm --epochs 200
python src/policy/train_policy.py --architecture gru  --epochs 200

# 4. Backtest all trained policies vs. Black-Scholes under stress conditions
python src/backtester/evaluate.py

# 5. Replicate the paper's frictionless Part I experiment
python src/backtester/replicate_part1.py --epochs 500
```

Run the tests with:

```bash
pytest tests/ -v
```

## Results, in brief

Two independent experiments this session, with two very different (and
equally honest) outcomes — see [`RESULTS.md`](RESULTS.md) for the full
numbers, plots, and diagnostic trail:

- **Frictionless replication (Part I)**: MLP and especially GRU learn a
  genuine, Black-Scholes-like delta response and post competitive CVaR.
  Basic RNN and LSTM do not — despite five targeted interventions (gradient
  clipping, extended training, learning-rate sweeps, orthogonal
  initialization, and removing an over-deep readout head), they converge to
  a near-constant policy in this exact setup. That failure is real,
  reproducible, and documented, not swept under the rug.
- **Stress-test backtest**: all four trained policies currently underperform
  Black-Scholes on tail risk. This traces directly to a known, *diagnosed*
  gap in the real-data market generator — it captures the right mean and
  spread but not real markets' fat-tail crash risk (skewness/kurtosis) — caught
  by the fidelity checker itself, not discovered by accident.

## Known limitations

- Real-data generator doesn't capture tail asymmetry (skew/kurtosis) yet —
  the single biggest open item; see `RESULTS.md`.
- Basic RNN and LSTM policies don't converge in this setup; GRU and MLP do.
- Market generator is a single-feature WGAN-GP, not the paper's full
  multi-variate TimeGAN (embedder/recovery/supervisor architecture).
- No option premium (P₀) term in the wealth formula.
- Toy-scale networks and training budgets throughout, not the paper's scale.

## References

- Kim, H. (2021). *Deep Hedging, Generative Adversarial Networks, and Beyond.*
- Buehler, H. et al. (2018). *Deep Hedging.* arXiv:1802.03042.
- Yoon, J. et al. (2019). *Time-series Generative Adversarial Networks.* NeurIPS 32.
