# Results and Findings

This documents what was implemented against Kim (2021), "Deep Hedging,
Generative Adversarial Networks, and Beyond," what the actual experiments
found, and — deliberately — what didn't work and why, not just what did.

## Contents

- [What's implemented vs. the paper](#whats-implemented-vs-the-paper)
- [Part I: frictionless replication](#part-i-frictionless-replication)
- [The GAN fidelity story](#the-gan-fidelity-story)
- [Stress-test backtest](#stress-test-backtest)
- [Known limitations](#known-limitations)
- [Ideas for future work](#ideas-for-future-work)

## What's implemented vs. the paper

| Paper component | This repo | Status |
|---|---|---|
| CVaR-minimizing direct policy search | `loss/cvar.py`, `policy/train_policy.py` | Matches |
| Basic RNN / LSTM / GRU comparison vs. Black-Scholes | `policy/hedging_agent.py` (`RecurrentHedgingAgent`) | Matches architecturally; RNN/LSTM don't converge in practice (see below) |
| Frictionless Part I (GBM, no transaction costs) | `backtester/replicate_part1.py` | Matches paper's exact Table 1 params (S₀=K=100, vol=0.15, T=1/12, 30 steps) |
| Part II: GAN-driven nonparametric scenarios | `generator/market_gan.py`, `generator/data.py` | Partial — single-feature WGAN-GP on real data, not the paper's multi-variate TimeGAN |
| Multi-alpha risk-return sweep | `train_policy.py --alpha-sweep`, `evaluate.py::run_alpha_sweep_backtest` | Matches |
| Delta-convexity diagnostic (paper Figs. 5/8/11) | `backtester/plotting.py::plot_delta_convexity` | Matches, and was the tool that caught the RNN/LSTM failure |
| Option premium P₀ in the wealth objective | *(not implemented)* | Open — wealth excludes premium collected, per this project's original `math_spec.md` |
| GAN fidelity validation | `generator/validate.py` | **Beyond** the paper — not something Kim (2021) does |

## Part I: frictionless replication

Paper's exact setup: S₀ = K = 100 (at-the-money), r = 0, vol = 0.15, T = 1/12
(one month), 30 time steps, batch size 1000, Adam, **zero transaction costs**.
`RecurrentHedgingAgent` uses `hidden_dim=128` with a single linear readout
(see [below](#the-rnnlstm-training-failure) for why not the deeper head).
500 training epochs per (architecture, α) pair; 5000-path out-of-sample test.

CVaR of terminal PnL, lower is better (reproduce with
`python src/backtester/replicate_part1.py --epochs 500`):

| α | Black-Scholes | MLP | Basic RNN | LSTM | GRU |
|---|---|---|---|---|---|
| 0.50 | 1.93 | 2.12 | 2.77 | 2.76 | **2.06** |
| 0.75 | 2.07 | 2.32 | 3.58 | 3.54 | **2.39** |
| 0.99 | 2.63 | 3.04 | 6.25 | 6.28 | 4.77 |

**GRU and MLP genuinely work.** GRU's learned delta closely tracks
Black-Scholes' analytic S-curve (see
`results/part1_replication/alpha_0_99/delta_convexity.png`), and its CVaR is
competitive with — sometimes better than — the closed-form baseline. MLP
learns a real, softer S-curve and improves substantially over an earlier,
unnormalized-input version (CVaR₉₉ dropped from 12.2 to 3.04 once the price
input was rescaled to moneyness — see below).

**Basic RNN and LSTM do not.** Both converge to a policy that is nearly
input-insensitive — δ ≈ 0.5 regardless of spot or time — and their CVaR is
close to what an unhedged/statically-hedged position would show. This is the
paper's central comparison, and it doesn't replicate for these two cell
types in this codebase, at this scale.

### The RNN/LSTM training failure

This was diagnosed, not just observed. In order:

1. **Input scale.** Neither policy network originally normalized price by
   strike. Every earlier experiment in this project used `strike=1.0`
   (normalization a no-op), so this never surfaced until Part I's literal
   S≈100 scale exposed it — raw S~100 saturates RNN/LSTM gate
   nonlinearities. **Fixing this helped MLP enormously** (CVaR₉₉ 12.2 → 3.04)
   but did not fix RNN/LSTM.
2. **Gradient explosion?** Measured directly — gradients are small (0.03–0.3),
   not exploding. Added gradient clipping to `PolicyTrainer` anyway (now
   available via `grad_clip_norm`); it isn't the fix here.
3. **Insufficient training?** Ruled out — 2000 epochs (4x) produced *zero*
   change in input-sensitivity (delta diff stayed at exactly 0.0000 the
   entire time). Not slow learning; a genuine dead-end.
4. **Learning rate?** Tried 5x and 10x higher — made it *worse* (full sigmoid
   saturation to a constant 0, a classic saturation lockup).
5. **Orthogonal recurrent-weight initialization** (a standard RNN
   stabilization trick, per-gate) at three learning rates (1e-3, 3e-3,
   1e-2): no improvement over the untrained baseline.
6. **The output head.** The paper's stated "128, 64, 64, 1" node counts were
   initially read as `RNN(128) → FC(64) → FC(64) → 1`. Direct hidden-state
   inspection showed the RNN's hidden state *does* carry a little genuine,
   input-dependent signal — but the deep ReLU head afterward killed it (dead
   units). Dropping to a single linear readout gave a small, real
   improvement (input-sensitivity went from exactly 0 to non-zero) — but
   confirmed via the same flat-prefix-then-jump test the plots use (not a
   naive "constant price for the whole path" test, which is unrealistic and
   overstates sensitivity), the improvement is roughly 1–3% of the
   sensitivity Black-Scholes/GRU/MLP show. Real, but not remotely sufficient
   to change the qualitative outcome.

Net: five bounded, principled interventions, one genuine partial fix (input
normalization, which fixed MLP), and Basic RNN/LSTM still don't learn
dynamic hedging in this setup. Whatever GRU's gating does differently here
remains an open question — plausibly connected to the paper's own suggested
future work (actor-critic variance reduction, entropy-bonus exploration),
which was not attempted.

## The GAN fidelity story

`generator/validate.py` compares real vs. synthetic terminal-return
statistics on four independent signals: diversity ratio (mode collapse),
mean bias in real-σ units (wrong distribution location), and skew/kurtosis
mismatches (wrong tail shape). It runs automatically after every
`train_gan.py` run.

**What it caught, in order:**

1. **Mode collapse — not actually present.** First hypothesis when the
   real-data-trained generator's downstream policies performed terribly was
   mode collapse. Directly measured: diversity ratio was 66.5%, well above
   the 30% warning threshold. Wrong diagnosis, caught before acting on it.
2. **Mean bias — real, and fixed.** The generator had learned a synthetic
   distribution centered **13.1 standard deviations** from the real mean (a
   fabricated permanent ~67% decline, vs. real data's actual +1% mean
   30-day return). Fixed by retraining with more epochs (200→1500) and a
   higher learning rate (1e-4→3e-4); mean bias dropped to **-1.1σ**,
   confirmed by the same tool.
3. **Tail shape — found, not yet fixed.** Real 30-day log-returns: skew
   -0.90, excess kurtosis +4.20 (real markets' crash risk). Synthetic: skew
   +0.08, kurtosis +0.08 — essentially symmetric and thin-tailed. This is
   the single biggest reason the stress-test backtest (below) shows all
   four trained policies underperforming Black-Scholes: they were trained
   against a generator that doesn't know crashes happen.

Current verdict (`results/gan_fidelity_summary.json`):
```
WARNING: skewness off by +0.99 (synthetic +0.08 vs. real -0.90) -- generator
isn't capturing real tail asymmetry; excess kurtosis off by -4.12 (synthetic
+0.08 vs. real +4.20) -- generator isn't capturing real fat-tail risk.
```

## Stress-test backtest

Regime-switching volatility (15%/60%, 10% per-step switch probability), 30
bps transaction friction, policies trained against the real-data WGAN-GP
generator described above (reproduce with `python src/backtester/evaluate.py`):

| Strategy | Mean wealth | CVaR 95% | CVaR 99% | Skew | Excess kurtosis | Total tx. cost |
|---|---|---|---|---|---|---|
| Black-Scholes | -0.695 | 1.82 | 2.34 | -1.97 | 5.34 | 12.90 |
| MLP | -0.511 | 5.76 | 19.96 | -15.79 | 291.3 | 3.88 |
| Basic RNN | -0.524 | 6.06 | 18.87 | -13.89 | 233.3 | 9.97 |
| LSTM | -0.521 | 6.00 | 19.86 | -15.64 | 286.3 | 5.49 |
| GRU | -0.535 | 5.71 | 18.41 | -15.41 | 271.8 | 4.87 |

All four trained policies show catastrophic excess kurtosis (230–290, vs.
Black-Scholes' 5.3) — a handful of extreme tail losses dominating an
otherwise tightly-clustered PnL distribution. This is the direct, predictable
consequence of the tail-shape gap above: the generator never showed these
policies a real crash during training, so none of them learned to defend
against one. This is not a separate bug — it's the same root cause reported
twice, once by the fidelity checker (on the generator directly) and once by
the backtest (on the policies it produced).

## Known limitations

Roughly in priority order:

1. **Generator tail-risk fidelity** (see above) — the most consequential
   open item; it's the reason the stress-test numbers look bad.
2. **Basic RNN / LSTM non-convergence** in the frictionless setting — root
   cause not fully identified (GRU-specific gating advantage, unconfirmed).
3. **Single-feature WGAN-GP, not TimeGAN** — the paper's Part II generator
   is a genuine multi-variate (6-feature) model with an
   embedder/recovery/supervisor architecture; this repo's generator is a
   simpler GRU/LSTM WGAN-GP on one feature (price).
4. **No option premium (P₀)** in the wealth formula — `Wealth_T` excludes
   the premium collected for writing the option, so mean wealth is
   persistently negative across every experiment in this repo. Consistent
   throughout, but a real deviation from the paper's formal objective.
5. **Scale** — networks and training budgets throughout are toy-sized
   relative to the paper (500k Monte Carlo scenarios, larger networks).
6. Real-data ticker (`^GSPC`) is a pure index with no dividend/split
   adjustments (`Adj Close == Close` always) — if the paper's authors used a
   security where those differ, this isn't an exact data match.

## Ideas for future work

- Extend `validate.py`'s tail-shape fix into the generator itself (e.g. a
  reconstruction loss term that explicitly penalizes skew/kurtosis
  mismatch, or a TimeGAN-style supervised stepwise loss).
- Implement TimeGAN properly (embedder + recovery + supervisor +
  discriminator, multi-variate OHLCV) — the biggest remaining structural gap.
- Actor-critic variance reduction or an entropy bonus for RNN/LSTM training,
  per the paper's own future-work section.
- Add the P₀ premium term to the wealth objective.
- Scale up: more Monte Carlo scenarios, larger networks, longer training,
  matching the paper's actual computational budget.
