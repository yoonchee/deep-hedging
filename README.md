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
  over multi-variate OHLCV data, trained via the paper's 3-phase procedure,
  now with the paper's own hyperparameters (31 hidden nodes, 3 layers) and
  its own binary cross-entropy discriminator loss as the default
  (`--discriminator-loss wgan-gp` keeps this repo's earlier deviation
  available). Its synthetic-data diversity has been hard to calibrate
  correctly (31% too low, then 214-224% too high, now 130.2% with an
  explicit diversity-matching loss) -- see `RESULTS.md` for the full,
  three-attempt story.
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
    black_scholes.py           Analytic Black-Scholes call delta + price (delta used as a CVaR control-variate baseline; price used as Part I's P0 premium)
  generator/
    market_gan.py             WGAN-GP Generator (GRU) + Discriminator (LSTM), single-feature
    timegan.py                TimeGAN: Embedder/Recovery/Generator/Supervisor/Discriminator, multi-feature
    data.py                   Real (yfinance, single- and multi-feature) and synthetic GBM data sources
    train_gan.py              WGAN-GP training CLI (+ moment-matching loss)
    train_timegan.py          TimeGAN 3-phase training CLI (BCE or WGAN-GP discriminator loss, moment- + diversity-matching loss, policy-training adapter)
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
tests/                         84 tests
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

# 3. Train each policy architecture against the generator. Defaults now
# match paper Table 3 (batch 1000, 25,000 gradient steps -- expect this to
# take longer than it used to; see RESULTS.md for timing):
python src/policy/train_policy.py --architecture mlp
# Basic RNN's stress-test convergence is seed-sensitive; --use-bs-baseline
# (a CVaR control-variate variance-reduction technique, see RESULTS.md)
# substantially improves and stabilizes it:
python src/policy/train_policy.py --architecture rnn --use-bs-baseline
python src/policy/train_policy.py --architecture lstm
python src/policy/train_policy.py --architecture gru

# 4. Backtest all trained policies vs. Black-Scholes under stress conditions.
# Evaluation batch is now 500,000 paths (matching Part I's paper-specified
# test scale), chunked internally to stay fast on CPU.
python src/backtester/evaluate.py

# 5. Replicate the paper's frictionless Part I experiment. Defaults now
# match paper Table 1 exactly (25,000 gradient steps, 500,000-path test set):
python src/backtester/replicate_part1.py

# 6. Optional: train TimeGAN instead (the paper's actual Part II generator,
# now with the paper's own hyperparameters, BCE discriminator loss, batch
# size, iteration count, and temporal train/test split) and compare its
# policies against the WGAN-GP ones from steps 1-4 -- see RESULTS.md for
# why the two currently disagree on which is "better". Defaults now match
# paper Table 2 (batch 178, 10,000 total iterations across 3 phases,
# training data through 2010, fidelity-checked against data through 2021);
# expect this to take hours, not minutes -- see RESULTS.md for timing.
python src/generator/train_timegan.py --data-source yfinance
python src/policy/train_policy.py --architecture mlp --generator-type timegan
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
  6.6 (P₀-inclusive) — real progress, though still short of LSTM/GRU's ~4.4.
- **TimeGAN vs. WGAN-GP**: TimeGAN's synthetic-data diversity was hard to
  calibrate — first only ~31% of real data's standard deviation (sigmoid
  latent space), then 214-224% after switching to tanh to fix it, then
  130.2% after adding an explicit diversity-matching loss alongside the
  paper's own hyperparameters and BCE discriminator loss. The 214-224%
  overshoot produced a "beats Black-Scholes" result for exactly one
  architecture each time — but *which* architecture changed completely once
  the RNN/LSTM input fix was applied (Basic RNN inherited the effect GRU
  used to show, and GRU's own result got worse). With diversity brought
  down to 130.2%, that same architecture (Basic RNN) is still separated
  from the pack by the same signature (lower transaction cost, lower std),
  just no longer landing exactly at Black-Scholes — a weaker version of
  the effect, not its absence; suggestive (not conclusive) evidence the
  attractor's strength scales with how extreme the overshoot is. Neither
  generator is straightforwardly "better" for hedging; see `RESULTS.md`
  for the full three-attempt story.
- **The option premium (P₀), and why mean wealth used to be negative
  everywhere**: this project's wealth formula omitted the option premium
  (P₀), verified directly against a closed-form Black-Scholes price in
  Part I and a Monte Carlo estimate in the stress test. P₀ is now
  implemented (`MarketEnvironment(premium=...)`) everywhere — Part I uses
  the exact closed-form price, the stress test and GAN-driven training
  estimate it via Monte Carlo through whatever simulator/generator is in
  use. Mean wealth is now ≈0 throughout, matching the paper's convention,
  and Part I's CVaR numbers match the paper's own absolute figures to
  within 2-9% for three of four architectures. No retraining was needed
  for existing checkpoints — a constant additive wealth shift doesn't
  change the CVaR-minimizing optimal policy. See `RESULTS.md` for the full
  derivation and both original checks.

## Known limitations

- ~~No option premium (P₀)~~ — resolved everywhere (Part I: exact
  closed-form; stress test / GAN-driven training: Monte Carlo estimate).
  Mean wealth is now ≈0 throughout, tightening further at paper scale; see
  `RESULTS.md`.
- ~~Part I's training budget doesn't match the paper~~ — resolved, with a
  mixed result: the paper's "50 epochs" is 25,000 gradient steps over a
  fixed 500,000-scenario dataset at batch=1000, now this repo's default
  (train and test scale both match Table 1 exactly). Black-Scholes then
  matches the paper's absolute CVaR to 2-3% at every α; LSTM/GRU are
  mixed (0.4% at α=0.75, up to 34% at α=0.5); Basic RNN's mismatch got
  *worse* at the corrected scale (58-59% at α=0.5/0.99), not better — see
  `RESULTS.md` for the full table and why that's still informative.
- Generator tail-shape fidelity is improved but not exact — synthetic skew
  now overshoots real data's, kurtosis still runs a bit low; see `RESULTS.md`.
- Basic RNN's stress-test convergence is seed-sensitive (bimodal: some
  seeds converge well, others get stuck) — substantially improved by a
  CVaR control-variate baseline, but not fully closed to LSTM/GRU's level.
- TimeGAN's diversity is improved but still overshoots (31% → 214-224% →
  130.2% across three calibration attempts) — its fidelity checker now has
  an upper-bound diversity warning (`DIVERSITY_OVERSHOOT_WARNING_THRESHOLD`)
  that would have caught the 214-224% overshoot, but 130.2% is under it;
  see `RESULTS.md`.
- Toy-scale networks and training budgets throughout, not the paper's scale
  (the paper's 500k Monte Carlo scenarios and larger networks are out of
  reach on this project's compute budget, stated plainly rather than chased).

## References

- Kim, H. (2021). *Deep Hedging, Generative Adversarial Networks, and Beyond.*
- Buehler, H. et al. (2018). *Deep Hedging.* arXiv:1802.03042.
- Yoon, J. et al. (2019). *Time-series Generative Adversarial Networks.* NeurIPS 32.
