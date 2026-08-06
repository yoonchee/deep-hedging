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
  they cascade into policy training, plus an explicit **moment-matching loss
  term** that trains the generator to reproduce real markets' skew/kurtosis
  directly, not just what fools the adversarial critic.
- A **TimeGAN market generator** (Yoon et al. 2019, the paper's actual Part
  II architecture) -- embedder/recovery/generator/supervisor/discriminator
  over multi-variate OHLCV data, trained via the paper's 3-phase procedure.
  Its synthetic-data diversity has been hard to calibrate correctly (first
  too low, then -- after a tanh fix -- too high), and which policy
  architecture benefits from that miscalibration changed completely after
  a separate RNN/LSTM bug fix -- see `RESULTS.md` for the full story.
- **Four hedging policy architectures** trained by direct policy search to
  minimize CVaR of terminal wealth: a feed-forward MLP (Buehler et al.'s
  `delta_k = f(I_k, delta_{k-1})` formulation) and three genuine recurrent
  policies (RNN/LSTM/GRU) that consume the whole price path in one pass,
  matching the paper's own architecture comparison. All three recurrent
  cell types now replicate the paper's frictionless Part I result after a
  standardized-log-moneyness input fix; Basic RNN's stress-test performance
  is seed-sensitive, substantially improved by a CVaR control-variate
  baseline (`--use-bs-baseline`, an adaptation of the paper's suggested
  actor-critic variance reduction to this codebase's direct-backprop
  training).
- A **differentiable CVaR loss** (Rockafellar-Uryasev) with a jointly-learned
  auxiliary threshold.
- A **stress-test backtester** comparing all four architectures against
  analytic Black-Scholes delta hedging under regime-switching volatility and
  transaction friction.
- A **frictionless Part I replication** reproducing the paper's controlled
  GBM experiment (no transaction costs) across three risk-aversion levels.

## Project structure

```
math_spec.md                  Core formulas: payoff, transaction cost, CVaR, WGAN-GP + TimeGAN losses, CVaR baseline
src/
  common/
    stats.py                   Shared skewness/excess-kurtosis/terminal-log-return helpers (tensor + float)
    black_scholes.py           Analytic Black-Scholes call delta (also used as a CVaR control-variate baseline)
  generator/
    market_gan.py             WGAN-GP Generator (GRU) + Discriminator (LSTM), single-feature
    timegan.py                TimeGAN: Embedder/Recovery/Generator/Supervisor/Discriminator, multi-feature
    data.py                   Real (yfinance, single- and multi-feature) and synthetic GBM data sources
    train_gan.py              WGAN-GP training CLI (+ moment-matching loss)
    train_timegan.py          TimeGAN 3-phase training CLI (+ moment-matching loss, policy-training adapter)
    validate.py               GAN fidelity checker (mode collapse / mean bias / tail shape), used by both
  loss/cvar.py                Differentiable CVaR loss (learnable threshold h)
  policy/
    hedging_agent.py          HedgingAgent (MLP) + RecurrentHedgingAgent (RNN/LSTM/GRU)
    train_policy.py           Policy training CLI (--generator-type wgan|timegan, single alpha or alpha sweep)
  environment/market_env.py   MarketEnvironment: wealth/transaction-cost simulation
  backtester/
    evaluate.py                Stress-test backtest vs. Black-Scholes (WGAN-GP- and TimeGAN-trained policies)
    replicate_part1.py         Frictionless Part I paper replication
    plotting.py                Shared chart library
tests/                         71 tests
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
# Basic RNN's stress-test convergence is seed-sensitive; --use-bs-baseline
# (a CVaR control-variate variance-reduction technique, see RESULTS.md)
# substantially improves and stabilizes it:
python src/policy/train_policy.py --architecture rnn  --epochs 200 --use-bs-baseline
python src/policy/train_policy.py --architecture lstm --epochs 200
python src/policy/train_policy.py --architecture gru  --epochs 200

# 4. Backtest all trained policies vs. Black-Scholes under stress conditions
python src/backtester/evaluate.py

# 5. Replicate the paper's frictionless Part I experiment
python src/backtester/replicate_part1.py --epochs 500

# 6. Optional: train TimeGAN instead (the paper's actual Part II generator)
# and compare its policies against the WGAN-GP ones from steps 1-4 -- see
# RESULTS.md for why the two currently disagree on which is "better".
python src/generator/train_timegan.py --phase1-epochs 500 --phase2-epochs 500 \
    --phase3-epochs 1500 --data-source yfinance
python src/policy/train_policy.py --architecture mlp --epochs 200 --generator-type timegan
python src/backtester/evaluate.py  # also runs the TimeGAN comparison if those checkpoints exist
```

Run the tests with:

```bash
pytest tests/ -v
```

## Results, in brief

See [`RESULTS.md`](RESULTS.md) for the full numbers, plots, and diagnostic
trail:

- **Frictionless replication (Part I)**: initially, MLP and GRU worked while
  Basic RNN/LSTM converged to a near-constant, input-insensitive policy
  despite five targeted interventions. The real root cause turned out to be
  a DC-dominated RNN input (the raw price channel was 97% a constant offset,
  3% signal) that a naive strike-normalization didn't fix. A standardized
  log-moneyness transform closed the gap completely — **all four
  architectures now replicate Black-Scholes-level CVaR at every
  risk-aversion level**, some slightly beating it.
- **GAN tail-shape fidelity**: the real-data generator originally captured
  the right mean and spread but not real markets' fat-tail crash risk
  (skew/kurtosis), caught by the built-in fidelity checker. Fixed with an
  explicit moment-matching loss term (`generator/train_gan.py`) that pulls
  the generator's terminal-return skew/kurtosis toward the real data's.
- **Stress-test backtest**: after the RNN/LSTM input fix, LSTM's stress-test
  tail risk also improved dramatically (CVaR₉₉ 18.4 → 5.0, now matching
  GRU). Basic RNN looked architecturally stuck at first (single-seed
  testing), but an 8-seed sweep revealed it's actually highly
  seed-sensitive and bimodal, not deterministically broken — every prior
  experiment in this project happened to use an unlucky default seed. A
  CVaR control-variate baseline (`--use-bs-baseline`) cuts cross-seed CVaR₉₉
  variance 5.7x and turns the canonical checkpoint's CVaR₉₉ from 19.3 to
  7.3 — real progress, though still short of LSTM/GRU's ~5.0.
- **TimeGAN vs. WGAN-GP**: TimeGAN's synthetic-data diversity was hard to
  calibrate — first only ~31% of real data's standard deviation (sigmoid
  latent space), then 214-224% after switching to tanh to fix it. That
  overshoot produces a "beats Black-Scholes" result for exactly one
  architecture each time — but *which* architecture changed completely once
  the RNN/LSTM input fix was applied (Basic RNN inherited the effect GRU
  used to show, and GRU's own result got worse), showing the original
  "GRU-specific" explanation was never real. Neither generator is
  straightforwardly "better"; see `RESULTS.md` for the full story.
- **Why every mean wealth in this repo is negative**: this project's wealth
  formula omits the option premium (P₀). Verified directly against a
  closed-form Black-Scholes price in Part I and a Monte Carlo estimate in
  the stress test — both match the measured negative mean wealth to within
  simulation noise — and this is very likely why the paper's own Part II
  results report large *positive* mean PnL where this repo's numbers are
  negative. See `RESULTS.md` for the full derivation and both checks.

## Known limitations

- **No option premium (P₀) term in the wealth formula** — this is why every
  mean wealth in this repo is negative, and verified (not just suspected)
  to be the likely reason the paper's own Part II results show large
  *positive* mean PnL where this repo's analogous numbers are negative; see
  `RESULTS.md` for the direct numerical check.
- Part I trains for 500 epochs, not the paper's stated 50 — checked
  directly: 50 epochs is verified insufficient in this implementation
  (CVaR roughly 2-3x worse across every architecture).
- Generator tail-shape fidelity is improved but not exact — synthetic skew
  now overshoots real data's, kurtosis still runs a bit low; see `RESULTS.md`.
- Basic RNN's stress-test convergence is seed-sensitive (bimodal: some
  seeds converge well, others get stuck) — substantially improved by a
  CVaR control-variate baseline, but not fully closed to LSTM/GRU's level.
- TimeGAN's diversity is miscalibrated (first too low, then too high after
  a fix) and its fidelity checker has no upper-bound diversity warning to
  catch the latter — see `RESULTS.md`.
- Toy-scale networks and training budgets throughout, not the paper's scale.

## References

- Kim, H. (2021). *Deep Hedging, Generative Adversarial Networks, and Beyond.*
- Buehler, H. et al. (2018). *Deep Hedging.* arXiv:1802.03042.
- Yoon, J. et al. (2019). *Time-series Generative Adversarial Networks.* NeurIPS 32.
