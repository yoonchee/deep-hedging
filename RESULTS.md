# Results and Findings

This documents what was implemented against Kim (2021), "Deep Hedging,
Generative Adversarial Networks, and Beyond," what the actual experiments
found, and — deliberately — what didn't work and why, not just what did.

## Contents

- [What's implemented vs. the paper](#whats-implemented-vs-the-paper)
- [Part I: frictionless replication](#part-i-frictionless-replication)
- [The GAN fidelity story](#the-gan-fidelity-story)
- [Stress-test backtest](#stress-test-backtest)
- [TimeGAN: the paper's actual Part II generator](#timegan-the-papers-actual-part-ii-generator)
- [Known limitations](#known-limitations)
- [Ideas for future work](#ideas-for-future-work)

## What's implemented vs. the paper

| Paper component | This repo | Status |
|---|---|---|
| CVaR-minimizing direct policy search | `loss/cvar.py`, `policy/train_policy.py` | Matches |
| Basic RNN / LSTM / GRU comparison vs. Black-Scholes | `policy/hedging_agent.py` (`RecurrentHedgingAgent`) | Matches architecturally; RNN/LSTM don't converge in practice (see below) |
| Frictionless Part I (GBM, no transaction costs) | `backtester/replicate_part1.py` | Matches paper's exact Table 1 params (S₀=K=100, vol=0.15, T=1/12, 30 steps) |
| Part II: GAN-driven nonparametric scenarios | `generator/market_gan.py` (WGAN-GP) + `generator/timegan.py` (TimeGAN) | Both implemented. TimeGAN matches the paper's actual architecture, but WGAN-GP+moment-loss produces better downstream hedging policies (see TimeGAN section) |
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
3. **Tail shape — found, then substantially fixed.** Real 30-day
   log-returns: skew ≈ -0.91, excess kurtosis ≈ +4.0-5.5 (real markets'
   crash risk; see the note on estimator noise below). The
   adversarially-trained generator had learned skew +0.08, kurtosis +0.08 —
   essentially symmetric and thin-tailed. The critic alone doesn't penalize
   this: matching the first two moments (mean, variance) is enough to fool
   it, so nothing in the WGAN-GP objective pushed the generator toward real
   markets' asymmetric crash risk.

   **The fix**: `train_gan.py` now adds an explicit moment-matching term to
   the generator loss — `(skew_fake - skew_real)² + (kurtosis_fake -
   kurtosis_real)²`, computed each step directly on the generated batch (see
   `common/stats.py::skewness_tensor` / `excess_kurtosis_tensor`, and
   `WGANGPTrainer.train_generator_step` in `train_gan.py`). The real-data
   targets are fixed once, up front, from a large (5000-path) sample
   (`--moment-target-batch-size`), not recomputed per minibatch — the WGAN's
   own 64-path training minibatch is far too small to estimate a 3rd/4th
   moment without enormous noise. Retraining the real-data generator with
   this term (same 1500 epochs, lr 3e-4) moved synthetic skew from +0.08 to
   **≈ -1.65** and kurtosis from +0.08 to **≈ 3.1** — the sign is now right
   and the magnitude is in the real ballpark, though skew overshoots
   (real's -0.91 vs. synthetic's -1.65) and kurtosis still runs a bit low.

   **A bug found along the way**: the automatic post-training fidelity check
   was comparing real vs. synthetic using only `args.batch_size` (64) real
   paths — nowhere near enough to estimate kurtosis reliably (a
   rare-event-dominated statistic). This made the auto-check's verdict
   noisy independent of any actual generator quality change. Fixed by
   reusing the larger `moment_target_batch_size` sample for the check
   instead. Even at N=5000, kurtosis estimates on ^GSPC vary
   seed-to-seed (real: 4.0 to 5.5 across three re-samples) because 70 years
   of daily data contains only a handful of independent crash episodes
   (1987, 2000-02, 2008-09, 2020) — a real data-scarcity limit on how
   precisely "real kurtosis" can even be pinned down, not a code bug.

Representative verdict (`results/gan_fidelity_summary.json`, seed 0, N=5000):
```
WARNING: skewness off by -0.71 (synthetic -1.65 vs. real -0.94) -- generator
isn't capturing real tail asymmetry.
```
(Kurtosis no longer triggers the warning threshold on this seed; skew now
overshoots in the correct direction rather than missing it entirely.)

## Stress-test backtest

Regime-switching volatility (15%/60%, 10% per-step switch probability), 30
bps transaction friction, policies retrained against the moment-matched
real-data WGAN-GP generator described above (reproduce with `python
src/backtester/evaluate.py`, after retraining each policy so it's trained
against the new generator checkpoint):

| Strategy | Mean wealth | CVaR 95% | CVaR 99% | Skew | Excess kurtosis | Total tx. cost |
|---|---|---|---|---|---|---|
| Black-Scholes | -0.695 | 1.82 | 2.34 | -1.97 | 5.34 | 12.90 |
| MLP | -0.667 | 4.58 | 7.26 | -4.00 | 23.8 | 10.00 |
| Basic RNN | -0.536 | 6.17 | 19.61 | -14.83 | 257.3 | 1.53 |
| LSTM | -0.548 | 5.77 | 18.37 | -14.90 | 259.6 | 2.77 |
| GRU | -0.712 | 3.14 | **4.52** | -2.19 | 6.7 | 15.49 |

Compare against the pre-fix table (MLP CVaR₉₉ 19.96 → **7.26**; GRU CVaR₉₉
18.41 → **4.52**, kurtosis 271.8 → **6.7** — essentially Black-Scholes-level
tail behavior now). **This is the same split found in Part I, closing the
loop**: MLP and GRU actually condition their hedge on the market state, so
giving them a generator with real crash risk taught them to defend against
one. Basic RNN and LSTM barely moved (CVaR₉₉ 18.9 → 19.6 and 19.9 → 18.4 —
noise-level, not improvement) because — per the Part I diagnosis — they
never learned to read the market state in the first place; a better
generator can't teach a policy something it structurally isn't sensing. The
tail-shape gap and the RNN/LSTM non-convergence turn out to be two
independent problems that only *looked* like one shared explanation when
both were present simultaneously.

## TimeGAN: the paper's actual Part II generator

The WGAN-GP above is a reasonable placeholder, but Kim (2021)'s actual Part
II generator is TimeGAN (Yoon et al. 2019): a 5-network
embedder/recovery/generator/supervisor/discriminator architecture over
multi-variate (OHLCV) data, not a single-feature model. `generator/timegan.py`
implements it (GRU-based Embedder/Recovery/Generator/Supervisor, LSTM-based
per-timestep Discriminator, all bounded to a shared [0,1] latent space via
sigmoid — see `math_spec.md` section 5); `generator/train_timegan.py`
implements the paper's 3-phase training procedure (autoencoder pretraining
→ supervised pretraining → joint adversarial training). One deliberate
deviation: the discriminator uses this repo's WGAN-GP loss (gradient
penalty) rather than the original paper's binary cross-entropy, for
consistency with the existing generator and because WGAN-GP's stability has
already mattered in this project (see the GAN fidelity story above).
Features: Open, High, Low, Close, Volume (5, not the paper's 6 — Adj Close
is dropped since it equals Close for `^GSPC`, a pure index; see known
limitations). The moment-matching loss (`math_spec.md` section 4.1) is
reused unchanged, applied to TimeGAN's recovered price channel.

Trained on the same real `^GSPC` data (500+500+1500 epochs, ~9 seconds
total — TimeGAN trains far faster per epoch than the WGAN-GP here, since
its default `hidden_dim=24` is much smaller than the WGAN-GP's 64).
Fidelity check on the extracted price channel, range across 4 seeds:

| Metric | Real | Synthetic | Verdict |
|---|---|---|---|
| Diversity ratio | -- | 30.7-33.0% | Borderline — right at the 30% mode-collapse threshold |
| Mean bias | -- | -1.5 to -1.6σ | Under the 2.0σ threshold, but a consistent, non-trivial bias |
| Skewness | ≈ -1.01 | ≈ -1.08 | Excellent — diff -0.14 to +0.07 across seeds |
| Excess kurtosis | ≈ 4.50 | ≈ 5.99 | Good — diff +0.80 to +1.59, always slightly high |

**TimeGAN's tail-shape fidelity (skew/kurtosis) is better than the
WGAN-GP+moment-loss generator's** — every seed passes cleanly ("OK", no
warnings), where the WGAN-GP still triggers a skew warning on some seeds.
This is a genuinely more paper-faithful result on the exact axis the
moment-loss fix targeted.

**But the stress-test backtest tells a different story.** Retraining all
four policies against TimeGAN (identical architecture, identical training
budget) and rerunning the same regime-switching stress test (reproduce with
`python src/backtester/evaluate.py`, after `train_policy.py --generator-type
timegan --architecture {mlp,rnn,lstm,gru}`):

| Strategy | Mean wealth | CVaR 95% | CVaR 99% | Skew | Excess kurtosis |
|---|---|---|---|---|---|
| MLP (WGAN-GP) | -0.667 | 4.58 | 7.26 | -4.00 | 23.8 |
| MLP (TimeGAN) | -0.482 | 8.26 | **26.31** | -14.64 | 252.2 |
| GRU (WGAN-GP) | -0.712 | 3.14 | **4.52** | -2.19 | 6.7 |
| GRU (TimeGAN) | -0.519 | 6.80 | **22.33** | -11.55 | 199.8 |

TimeGAN-trained MLP and GRU are dramatically *worse* than their
WGAN-GP+moment-loss counterparts — worse, in fact, than the original
pre-moment-loss WGAN-GP policies (CVaR₉₉ 19.96 and 18.41 respectively). The
likely cause is directly visible in the fidelity table above: **diversity,
not tail shape, is the bottleneck here.** TimeGAN's synthetic price paths
have only ~31% of real data's standard deviation — right at the
mode-collapse warning line — because its architecture composes *four*
sigmoid-bounded transformations in sequence (Embedder → Generator →
Supervisor → Recovery), each squashing toward the middle of [0,1],
compounding into a narrower overall distribution than the WGAN-GP's single
tanh-bounded-log-return parameterization. Policies trained on that narrower
distribution rarely see large moves during training — the near-zero
transaction cost for TimeGAN-trained MLP (0.0000375 vs. WGAN-GP's 0.005) is
the tell, it learned an almost fully static hedge — and get caught
flat-footed by the stress test's genuinely volatile regime-switching
scenario. Excellent tail-*shape* fidelity didn't compensate for a
diversity/scale shortfall the fidelity checker had already flagged, before
either policy was ever trained.

This is the headline finding of the TimeGAN work: **a more paper-faithful
generator architecture does not automatically produce better downstream
hedging policies.** Here, the simpler WGAN-GP, once explicitly corrected
for tail shape, generalizes better to a genuinely adversarial stress
scenario than the more complex, more paper-faithful TimeGAN — because of a
diversity shortfall visible in the fidelity checker's output well before
either policy was ever trained. This isn't a failure of the implementation
so much as an argument for keeping the fidelity checker in the loop for
*every* generator, not just the first one built.

## Known limitations

Roughly in priority order:

1. **Generator tail-risk fidelity — improved, not perfect.** The
   moment-matching loss fixed the sign and rough magnitude of both skew and
   kurtosis, and this measurably improved MLP/GRU's stress-test tail risk.
   But synthetic skew now overshoots real's (-1.65 vs. -0.91) and kurtosis
   still runs a bit low (~3.1 vs. ~4.0-5.5, itself a noisy target — see
   above). A learned, per-batch-adaptive weighting (vs. the current fixed
   `--lambda-moment`) could plausibly tighten this further.
2. **Basic RNN / LSTM non-convergence** in the frictionless setting — root
   cause not fully identified (GRU-specific gating advantage, unconfirmed).
   Confirmed independent of the generator fidelity issue: retraining RNN/LSTM
   against the fixed generator left their stress-test tail risk unchanged
   (they don't condition on market state at all, so a better generator has
   nothing to teach them).
3. **TimeGAN's diversity shortfall** — its price-channel standard deviation
   is only ~31% of real data's (right at the mode-collapse warning
   threshold), despite excellent skew/kurtosis fidelity. Traced to
   composing four sigmoid-bounded transformations (Embedder→Generator→
   Supervisor→Recovery); see the TimeGAN section above. This is why
   WGAN-GP+moment-loss, not TimeGAN, is the better choice for policy
   training today, despite being the less paper-faithful architecture.
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

- Tighten the moment-matching loss further (adaptive `lambda_moment`
  schedule, or matching higher moments / a full quantile loss instead of
  just skew+kurtosis) to close the remaining tail-shape gap.
- Loosen TimeGAN's [0,1] latent-space bound on Generator/Supervisor (only
  Embedder/Recovery strictly need it, to guarantee Recovery's reconstructed
  price channel is well-defined) — the compounding sigmoid squash is the
  likely cause of its diversity shortfall, and fixing it could let TimeGAN's
  superior tail-shape fidelity actually translate into better stress-test
  results.
- Actor-critic variance reduction or an entropy bonus for RNN/LSTM training,
  per the paper's own future-work section.
- Add the P₀ premium term to the wealth objective.
- Scale up: more Monte Carlo scenarios, larger networks, longer training,
  matching the paper's actual computational budget.
