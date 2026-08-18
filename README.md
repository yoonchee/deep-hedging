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
  now with the paper's own hyperparameters (31 hidden nodes, 3 layers, batch
  178, ~10,000 iterations) and its own temporal train/test split. A
  **fidelity checker** (`generator/validate.py`) checks both terminal-return
  statistics (diversity, mean bias, skew, kurtosis) and path-level dynamics
  (per-step volatility, return autocorrelation) -- see `RESULTS.md` for why
  both matter and the calibration story behind each.
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
  training). Seed sensitivity is the dominant effect for the recurrent
  policies under stress — large enough that single-seed comparisons on GRU
  measure the seed rather than the change, which is why two previously
  promoted GRU fixes were retracted after 5-seed reruns.
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
    lstm_introspection.py      Manual LSTM unroll exposing per-step cell state/gates nn.LSTM.forward() doesn't return; backs RESULTS.md's mechanism (b) diagnosis
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
tests/                         147 tests
results/                       Generated plots + JSON summaries (gitignored inputs: data/, checkpoints/)
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Every training CLI (`train_gan.py`, `train_timegan.py`, `train_policy.py`)
auto-detects the fastest available device (CUDA > MPS > CPU) via
`common/device.py`, or force one with `--device cpu|cuda|mps`. Benchmarked
directly on Apple Silicon (M5 Pro): MPS is ~5.5x faster than CPU for policy
training (the dominant cost in this project's own training runs -- a
paper-scale LSTM policy run drops from ~78 minutes to ~14), though TimeGAN's
own training loop (much smaller batch size) benchmarks slightly *slower* on
MPS than CPU. `--smoothness-penalty-weight`'s double-backward through an
`nn.LSTM` isn't supported on MPS as of torch 2.8 and fails fast with a clear
error if combined with an MPS device -- pass `--device cpu` for that flag.
`train_gan.py` hits the same torch limitation unconditionally (WGAN-GP's
gradient penalty *is* a double-backward through an LSTM discriminator), so it
auto-detects down to CPU on Apple Silicon rather than crashing; an explicit
`--device mps` there raises with the reason.

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
# Basic RNN (TimeGAN) needs both halves of its promoted fix: --lr 1e-3
# de-saturates the hidden state, and only then can --moneyness-clip help
# (clipping an input a saturated state ignores is a no-op). 5-seed validated.
python src/policy/train_policy.py --architecture rnn --generator-type timegan \
    --use-bs-baseline --lr 1e-3 --moneyness-clip -0.15 0.10
python src/policy/train_policy.py --architecture lstm --generator-type timegan --slow-ramp-fraction 0.05
# GRU (TimeGAN) takes no extra flags: --moneyness-clip was promoted for it on
# one seed and retracted after a 5-seed rerun found it harmful.
python src/policy/train_policy.py --architecture gru --generator-type timegan
python src/backtester/evaluate.py  # also runs the TimeGAN comparison if those checkpoints exist
```

Run the tests with:

```bash
pytest tests/ -v
```

## Results, in brief

Full numbers, plots, and diagnostic trail are in
[`RESULTS.md`](RESULTS.md#summary--current-state). Headline results, at the
paper's own scale throughout:

**Part I: frictionless replication** (500,000 train/test scenarios, 25,000
gradient steps). CVaR of terminal PnL, lower is better:

| α | Black-Scholes | MLP | Basic RNN | LSTM | GRU |
|---|---|---|---|---|---|
| 0.50 | 0.207 | 0.212 | 0.205 | 0.193 | 0.268 |
| 0.75 | 0.343 | 0.347 | 0.333 | 0.312 | 0.311 |
| 0.99 | 0.947 | 0.843 | 0.780 | 0.697 | 0.699 |

All four architectures beat Black-Scholes at every α, matching the paper's
own absolute CVaR figures closely (2-3% for Black-Scholes, 0.1-2.3% for
Basic RNN, 0.4-34% for LSTM/GRU depending on α).

**Stress test: regime-switching volatility + transaction costs** (500,000
paths):

| Strategy | CVaR 95% | CVaR 99% | Excess kurtosis |
|---|---|---|---|
| Black-Scholes | 1.20 | 1.85 | 7.6 |
| MLP | 2.24 | 3.48 | 7.5 |
| Basic RNN | 1.59 | 2.46 | 7.4 |
| LSTM | 2.24 | 3.58 | 8.6 |
| GRU | 3.05 | 8.31 | 116,510.8 |

These are freshly measured from checkpoints rebuilt from scratch, so the
first four rows differ by a few percent from earlier published ones (the
generator had to be regenerated too). Black-Scholes, which has no checkpoint,
reproduces exactly — a control confirming the scenario is unchanged.

Every architecture except GRU behaves like an ordinary fat-tailed P&L
distribution. **GRU's row is one draw from a wide distribution, not a
property of the architecture**: rerun across 5 seeds, baseline GRU's CVaR₉₉
spans 5.37-13.11 with no intervention at all. Its previously-advertised
`--grad-clip-norm 1.0` fix has been retracted — at 5 paired seeds it improves
2/5 and is bit-for-bit inert at a third.

**TimeGAN-driven policies** (the paper's actual Part II generator, same
scale):

| Architecture | Status |
|---|---|
| MLP | Not reproducible — the generator its "clean" row was measured against was never preserved |
| Basic RNN | Substantially fixed and promoted (`--lr 1e-3 --moneyness-clip -0.15 0.10`): mean CVaR₉₉ 20.65 → 3.77 across 5 seeds, 5/5 improved, 2/5 fully clean |
| LSTM | Fixed and promoted (`--slow-ramp-fraction 0.05`): CVaR₉₉ 42.13 → 3.24 |
| GRU | Fix retracted — `--moneyness-clip` improves only 1/5 seeds and doubles mean CVaR₉₉ |

The option premium (P₀) is now correctly included everywhere — Part I uses
the exact closed-form Black-Scholes price, the stress test and GAN-driven
settings estimate it via Monte Carlo — so mean wealth is ≈0 throughout,
matching the paper's own convention.

## Known limitations

Roughly in priority order — see
[`RESULTS.md`](RESULTS.md#summary--current-state) for the diagnostic trail
behind each:

1. **GRU, on both generators, is dominated by seed variance.** Baseline
   CVaR₉₉ spans 5.37-13.11 (WGAN-GP) and 2.78-39.67 (TimeGAN) across seeds
   with no intervention; that spread dwarfs every measured effect of every
   fix tried on it. Both previously-promoted GRU fixes failed multi-seed
   validation — one inert, one actively harmful — and are retracted. GRU
   needs its variance explained, not another fix attempt.
2. **The TimeGAN rows can't be reproduced from repo state** — attempt 4's
   generator was never preserved, and MLP (TimeGAN) is no longer clean
   against the one that survives.
3. **Basic RNN (TimeGAN)** — substantially fixed (5/5 seeds improved), but
   3/5 seeds still show 15-50 catastrophic paths per 500,000, the clip
   bound was inherited from GRU rather than tuned, and it's one generator.
4. A dedicated TimeGAN loss term that matches path-level dynamics (not just
   terminal statistics) produced the first generator to pass every fidelity
   check — but a policy trained against it had dramatically worse tail
   risk. Open, single-seed, not pursued further — now the main outstanding
   single-seed claim in the project.
5. Minor: WGAN-GP's synthetic skew/kurtosis still slightly miss real
   data's; `^GSPC` has no dividend/split adjustments.

## References

- Kim, H. (2021). *Deep Hedging, Generative Adversarial Networks, and Beyond.*
- Buehler, H. et al. (2018). *Deep Hedging.* arXiv:1802.03042.
- Yoon, J. et al. (2019). *Time-series Generative Adversarial Networks.* NeurIPS 32.
