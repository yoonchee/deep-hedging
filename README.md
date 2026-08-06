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
  too low, then -- after a tanh fix -- too high), and the results are
  architecture-dependent: it produced this project's single best hedging
  policy (GRU) and its worst (MLP) -- see `RESULTS.md` for the full story.
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
math_spec.md                  Core formulas: payoff, transaction cost, CVaR, WGAN-GP + TimeGAN losses
src/
  common/stats.py             Shared skewness/excess-kurtosis/terminal-log-return helpers (tensor + float)
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
tests/                         66 tests
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

- **Frictionless replication (Part I)**: MLP and especially GRU learn a
  genuine, Black-Scholes-like delta response and post competitive CVaR.
  Basic RNN and LSTM do not — despite five targeted interventions (gradient
  clipping, extended training, learning-rate sweeps, orthogonal
  initialization, and removing an over-deep readout head), they converge to
  a near-constant policy in this exact setup. That failure is real,
  reproducible, and documented, not swept under the rug.
- **GAN tail-shape fidelity**: the real-data generator originally captured
  the right mean and spread but not real markets' fat-tail crash risk
  (skew/kurtosis), caught by the built-in fidelity checker. Fixed with an
  explicit moment-matching loss term (`generator/train_gan.py`) that pulls
  the generator's terminal-return skew/kurtosis toward the real data's.
- **Stress-test backtest**: retrained against the fixed generator, MLP and
  GRU's tail risk dropped sharply (GRU's CVaR₉₉ 18.4 → 4.5, now close to
  Black-Scholes) — but Basic RNN and LSTM barely moved, confirming their
  failure is structural (they don't condition on market state at all) and
  independent of the generator fix.
- **TimeGAN vs. WGAN-GP**: TimeGAN's synthetic-data diversity was hard to
  calibrate — first only ~31% of real data's standard deviation (sigmoid
  latent space), then 214-224% after switching to tanh to fix it. That
  overshoot happened to produce the single best-performing policy in this
  project: GRU trained against the tanh-fixed TimeGAN nearly matches
  Black-Scholes' mean/std exactly and *beats* it on stress-test CVaR
  (confirmed stable across 4 backtest seeds) — while MLP trained the same
  way is the worst-performing policy in the whole comparison. Neither
  generator is straightforwardly "better"; see `RESULTS.md` for the full
  three-act story.

## Known limitations

- Generator tail-shape fidelity is improved but not exact — synthetic skew
  now overshoots real data's, kurtosis still runs a bit low; see `RESULTS.md`.
- Basic RNN and LSTM policies don't converge in this setup; GRU and MLP do.
- TimeGAN's diversity is miscalibrated (first too low, then too high after
  a fix) and its fidelity checker has no upper-bound diversity warning to
  catch the latter — see `RESULTS.md`.
- No option premium (P₀) term in the wealth formula.
- Toy-scale networks and training budgets throughout, not the paper's scale.

## References

- Kim, H. (2021). *Deep Hedging, Generative Adversarial Networks, and Beyond.*
- Buehler, H. et al. (2018). *Deep Hedging.* arXiv:1802.03042.
- Yoon, J. et al. (2019). *Time-series Generative Adversarial Networks.* NeurIPS 32.
