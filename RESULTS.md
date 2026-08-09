# Results and Findings

This documents what was implemented against Kim (2021), "Deep Hedging,
Generative Adversarial Networks, and Beyond," what the actual experiments
found, and — deliberately — what didn't work and why, not just what did.

## Contents

- [What's implemented vs. the paper](#whats-implemented-vs-the-paper)
- [Terminal wealth and the P₀ (premium) term](#terminal-wealth-and-the-p₀-premium-term)
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
| Basic RNN / LSTM / GRU comparison vs. Black-Scholes | `policy/hedging_agent.py` (`RecurrentHedgingAgent`) | Matches in Part I (all three cell types replicate Black-Scholes-level CVaR after a standardized-log-moneyness input fix); in the harder stress-test setting Basic RNN was seed-sensitive, addressed by a CVaR control-variate baseline — the current checkpoint's stress-test CVaR₉₉ (2.58) is now the best of the three recurrent architectures, ahead of LSTM (3.49) and GRU (3.81) — see below |
| Frictionless Part I (GBM, no transaction costs) | `backtester/replicate_part1.py` | Matches paper's exact Table 1 params, including scale: 500,000-scenario train/test sets, 25,000 gradient steps (the paper's "50 epochs" over that dataset size at batch=1000, in this codebase's per-step convention) — see below |
| Part II: GAN-driven nonparametric scenarios | `generator/market_gan.py` (WGAN-GP) + `generator/timegan.py` (TimeGAN) | Both implemented, both now scaled to paper's Table 2 (TimeGAN) and paper-scale evaluation. TimeGAN's generator fidelity is now close (87.3% diversity), but its trained policies show catastrophic tail risk at the paper's own 500,000-path test scale — see TimeGAN section and Known limitations |
| Multi-alpha risk-return sweep | `train_policy.py --alpha-sweep`, `evaluate.py::run_alpha_sweep_backtest` | Matches, now extended to the paper's own Part II grid {0.5, 0.75, 0.99, 0.995, 0.997} |
| Delta-convexity diagnostic (paper Figs. 5/8/11) | `backtester/plotting.py::plot_delta_convexity` | Matches, and was the tool that caught the RNN/LSTM failure |
| Option premium P₀ in the wealth objective | `MarketEnvironment(premium=...)`, `common/black_scholes.py::black_scholes_call_price`, `environment/market_env.py::estimate_premium_monte_carlo` | Implemented everywhere: closed-form and exact in Part I, Monte Carlo-estimated (500,000 paths, chunked) in the stress test and every GAN-driven setting, which have no closed-form price — see [below](#terminal-wealth-and-the-p₀-premium-term) |
| GAN fidelity validation | `generator/validate.py` | **Beyond** the paper — not something Kim (2021) does |

## Terminal wealth and the P₀ (premium) term

Every mean-wealth/mean-PnL number in this document used to be negative —
Part I's Black-Scholes baseline showed -1.729 at every α, and the
stress-test's Black-Scholes baseline showed -0.695. Both numbers turned out
to be explained by the same missing term, not a bug and not an unexplained
finding:

`Wealth_T = -Payoff(S_T) + Σ δ_t(S_{t+1}-S_t) - Costs` (the original form,
`math_spec.md` §1.1) never added back the premium P₀ that would be
collected for writing the option in the first place. For a hedge that
(near-)perfectly replicates the payoff, the classical replication argument
is that the accumulated hedging P&L, financed at the risk-free rate, nets
out to `Payoff(S_T) - C₀` — so without collecting a premium, `Wealth_T ≈
-C₀`: a constant offset equal to the option's own fair value, not zero.

This was directly verified in both settings:

- **Part I** (S₀=K=100, vol=0.15, T=1/12, r=0): the closed-form
  Black-Scholes call price at these exact parameters is C₀ ≈ 1.727. The
  measured Black-Scholes mean PnL (without P₀) was **-1.729** — a match to
  two decimal places.
- **Stress test** (regime-switching vol 15%/60%, switch_prob=0.10, S₀=K=1,
  30 steps, dt=1): there's no closed-form price for this process, so the
  fair value was estimated directly as `E[Payoff(S_T)]` via Monte Carlo
  (200,000 paths, same simulator used for the backtest): **0.690**. The
  measured Black-Scholes mean wealth was **-0.695** — again a match to
  within simulation noise.

**P₀ is now implemented — for Part I only.** `MarketEnvironment` gained a
`premium` parameter (default `0.0`, so nothing changes unless a caller
opts in — `math_spec.md` §1.1) and `black_scholes_call_price` was added to
`common/black_scholes.py`. `replicate_part1.py` computes the closed-form
C₀ at its own exact parameters and passes it in, since constant vol and
r=0 make that price exact; a constant additive shift to wealth doesn't
change the CVaR-minimizing optimal policy (same argmin, same gradient),
only the reported scale, so this required no retraining-logic changes,
only rerunning. The stress test and every GAN-driven setting
(`evaluate.py`, `train_policy.py`) have no closed-form price to use
instead and were **deliberately left at `premium=0.0`** rather than
substituting a Monte Carlo estimate, whose own error would then need
characterizing — see Ideas for future work.

The result, in Part I: Black-Scholes' mean PnL is now **-0.0017**,
matching the paper's own reported ≈0.0005 to three decimal places, and —
more significantly — this repo's absolute CVaR numbers now land within
2-9% of the paper's own absolute CVaR figures for three of four
architectures (see the [Part I section](#part-i-frictionless-replication)
for the full comparison table). This is a materially stronger replication
claim than "same qualitative shape," and it retires the earlier, more
cautious framing of this section (which had verified only the mean/P₀
portion of the gap, not the CVaR-sign-convention question — that question
is now answered: the two conventions are negatives of each other once P₀
is included).

**This is also almost certainly why the paper's own Part II results show
large positive mean PnL (≈18-35 on their S₀=100 scale) while this repo's
stress-test numbers are still negative** (P₀ isn't wired in there yet). The
paper's own Part I table reports Black-Scholes mean PnL ≈ 0.0005 —
essentially zero — which is only reachable if its objective *does* include
a correctly-priced P₀ ≈ C₀ that cancels the expected payout, exactly as
verified above for this repo once P₀ was added. This is not evidence of
any issue with the paper's own generator, a hypothesis briefly considered
while first investigating this gap and dropped once the reconciliation
held up.

## Part I: frictionless replication

Paper's exact setup: S₀ = K = 100 (at-the-money), r = 0, vol = 0.15, T = 1/12
(one month), 30 time steps, batch size 1000, Adam, **zero transaction costs**.
`RecurrentHedgingAgent` uses `hidden_dim=128` with a single linear readout
(see [below](#the-rnnlstm-training-failure-and-its-real-fix) for why not the
deeper head). Table 1 also specifies **500,000 pre-generated training
scenarios and a separate 500,000-scenario test set** — re-read directly
from the paper (not worked from a "500k Monte Carlo scenarios" summary) to
scope a compute-budget push once the user offered 24+ hours of local
compute. Standard ML epoch semantics (one epoch = one pass over the
dataset) make the paper's "50 training epochs" at batch_size=1000 over that
fixed 500,000-scenario set into 50 × (500,000 / 1,000) = **25,000 gradient
steps** — not 50 in this codebase's own convention, where each "epoch" is
one gradient step on a freshly-sampled batch (GBM data is cheap to
generate on the fly, so there's no fixed pool to epoch over). This
supersedes the project's earlier "500 epochs, 10x the paper's 50"
framing, which was comparing mismatched units the whole time: an earlier,
smaller push to 500 steps was actually only ~2% of the paper's real
budget, not 10x it. Current defaults: `train_epochs=25_000`,
`test_batch_size=500_000` (the same scale as the paper's own test set;
evaluated in chunks of 50,000 to avoid a severe, non-linear CPU slowdown
RNN/LSTM/GRU policies hit at very large single-pass batch sizes — see
`MarketEnvironment.simulate`'s `chunk_size` docstring).

**Since [P₀ was added](#terminal-wealth-and-the-p₀-premium-term)**,
this is also the one experiment in the repo where the wealth objective
includes the option premium (closed-form Black-Scholes price at these exact
params, C₀ ≈ 1.727) — constant vol and r=0 make it exact here, unlike the
regime-switching/GAN-driven settings elsewhere. Black-Scholes' own mean PnL
is **0.0001** at the paper's actual training/test scale — matching the
paper's own reported ≈0.0005 even more closely than the earlier
smaller-scale run did (-0.0017). Every architecture's mean PnL is now within
0.001-0.0013 of zero (MLP 0.0000 to GRU 0.0012 across the three α values),
tightening toward the paper's near-zero convention as the training budget
grew — a clean, unambiguous improvement, unlike the CVaR picture below.

CVaR of terminal PnL, lower is better, at the paper's exact training/test
scale (reproduce with `python src/backtester/replicate_part1.py`):

| α | Black-Scholes | MLP | Basic RNN | LSTM | GRU |
|---|---|---|---|---|---|
| 0.50 | 0.207 | 0.212 | 0.325 | **0.193** | 0.268 |
| 0.75 | 0.343 | 0.347 | 0.466 | **0.312** | 0.311 |
| 0.99 | 0.947 | 0.843 | 1.237 | **0.697** | 0.699 |

**This is a genuinely mixed result, not a clean confirmation that more
compute closes the remaining gaps.** LSTM/GRU still beat Black-Scholes at
every α (as at the smaller scale), and MLP now also beats it at α=0.99 —
the paper's core qualitative claim holds. But comparing against the
paper's own absolute CVaR figures (same sign-convention reconciliation as
before: paper reports CVaR of *PnL*, negative = loss; this repo reports
CVaR of *losses*, positive) tells a less tidy story than the smaller-scale
run did:

| α | Architecture | Paper | This repo | Difference |
|---|---|---|---|---|
| 0.50 | Black-Scholes | 0.2028 | 0.207 | 2.1% |
| 0.50 | Basic RNN | 0.2040 | 0.325 | 59.4% |
| 0.50 | LSTM | 0.2197 | 0.193 | 12.2% |
| 0.50 | GRU | 0.2000 | 0.268 | 34.1% |
| 0.75 | Black-Scholes | 0.3353 | 0.343 | 2.2% |
| 0.75 | Basic RNN | 0.3257 | 0.466 | 43.0% |
| 0.75 | LSTM | 0.3132 | 0.312 | **0.4%** |
| 0.75 | GRU | 0.3119 | 0.311 | **0.4%** |
| 0.99 | Black-Scholes | 0.9203 | 0.947 | 2.9% |
| 0.99 | Basic RNN | 0.7810 | 1.237 | 58.3% |
| 0.99 | LSTM | 0.8974 | 0.697 | 22.4% |
| 0.99 | GRU | 0.7270 | 0.699 | 3.8% |

Three things worth stating plainly:

- **Black-Scholes matches the paper to 2-3% at every α, consistently** —
  expected, since it's an analytic policy, not a learned one; the residual
  is pure Monte Carlo sampling noise between this repo's 500,000-path test
  set and the paper's own. This is the stable anchor the rest of the table
  should be read against.
- **LSTM/GRU are mixed, not uniformly improved by the larger budget**:
  a striking 0.4% match at α=0.75 for both, but 12-34% at α=0.5 and
  3.8-22.4% at α=0.99. Scaling up training did not monotonically tighten
  the match to the paper's numbers architecture-by-architecture — it
  moved some cells closer and others farther, which is itself informative:
  it suggests the residual gap isn't simply "not enough training," or the
  gap would have shrunk everywhere, not just at one α.
- **Basic RNN's mismatch got *worse* at the correct scale, not better** —
  59% and 58% off at α=0.5/0.99 (vs. 32% at the old, far smaller
  500-step/seed-0 run reported previously), and 43% at α=0.75. This is the
  same default seed=0 as every other Basic RNN result in this project, so
  it doesn't distinguish "genuinely worse at this scale" from "still the
  same seed-sensitivity already documented below and in the stress-test
  section, just expressed differently at a different step count." A
  multi-seed sweep at 25,000 steps (not done here) would be needed to tell
  the two apart — this scale-up adds a data point consistent with Basic
  RNN's instability being real and scale-independent, not evidence that
  more training resolves it.

### The RNN/LSTM training failure, and its real fix

This took two rounds of diagnosis, and the first round's conclusion was
**wrong** — worth stating plainly rather than only keeping the corrected
version, since the wrong diagnosis was published in this file for a while
and the process of finding the error is itself informative.

**Round 1 (five interventions, all either ineffective or only partially
effective):**

1. **Input scale — believed fixed, actually wasn't.** Dividing price by
   strike (`S/K`) was applied and did help MLP enormously (CVaR₉₉ 12.2 →
   3.04), which looked like confirmation. It didn't help RNN/LSTM at all.
2. **Gradient explosion?** Measured directly — gradients are small
   (0.03–0.3), not exploding. Gradient clipping added to `PolicyTrainer`
   regardless (`grad_clip_norm`); not the fix.
3. **Insufficient training?** Ruled out — 2000 epochs (4x) produced *zero*
   change in input-sensitivity.
4. **Learning rate?** Tried 5x and 10x higher — made it *worse* (full
   sigmoid saturation to a constant 0).
5. **Orthogonal recurrent-weight initialization**, three learning rates: no
   improvement.
6. **The output head.** Reading the paper's "128, 64, 64, 1" node counts as
   `RNN(128) → FC(64) → FC(64) → 1` turned out to kill a little genuine
   signal via dead ReLU units; dropping to a single linear readout gave a
   *small, real* improvement (input-sensitivity went from exactly 0 to
   non-zero) — but only ~1-3% of the sensitivity Black-Scholes/GRU/MLP show.

Five interventions, one small real effect, RNN/LSTM still fundamentally
stuck. The conclusion at the time: "whatever GRU's gating does differently
here remains an open question."

**Round 2: the actual root cause.** `RecurrentHedgingAgent` feeds the RNN
a single channel — `prices[:, :-1, :] / strike`. Measured directly on Part
I's own data: this channel has **mean ≈0.998, std ≈0.030**. The informative
signal is only ~3% of the input's magnitude, buried under a near-constant
offset of 1.0. Dividing by strike doesn't fix this at all — it's
scale-invariant with respect to the signal-to-DC ratio; it just moves the
offset from ~100 down to ~1.0 and leaves the *ratio* of noise to constant
untouched. This is the actual reason every round-1 intervention either
failed or barely helped: there was never enough usable signal reaching the
network to shape an input-sensitive policy. The tiny round-1 improvement
from the shallower output head is the fingerprint of a real but
~30-100x-too-small signal, not an absent one. And it explains why MLP and
GRU escaped this: MLP gets `delta_prev`, `T-t`, and `implied_vol` as
explicit, well-scaled side features (never just raw price alone); GRU's
specific gating apparently extracts enough of the attenuated signal to
still work, where RNN/LSTM's don't.

**The fix**: replace the raw `S/K` channel with standardized log-moneyness,
`log(S_t/K) / (implied_vol · √T)` (`hedging_agent.py::RecurrentHedgingAgent`,
now takes `implied_vol`/`time_to_maturity` as constructor hyperparameters
used only to fix this scaling constant, not fed to the network per-step).
This is pure preprocessing — no new information the network didn't already
have access to, still "the RNN sees only the price path" — but it turns a
signal that's 97% swamped by a constant into an O(1)-scaled quantity.
Verified directly before touching CVaR: on Part I's exact params, the old
channel's std was ≈0.03; the new one lands in the 0.1–10 range by
construction (`tests/test_policy.py::test_recurrent_agent_moneyness_input_is_order_one_scale_for_realistic_params`).
Delta **span** across the spot grid (the metric that directly measures
input-sensitivity, unlike CVaR which can move for other reasons) went from
LSTM ≈0.085 / Basic RNN ≈0 to **LSTM ≈0.96, Basic RNN ≈0.97** at 200
epochs — matching GRU's ≈1.0 essentially exactly. The CVaR table above (500
epochs) confirms this translates into the paper's actual claimed result.

The lesson: the round-1 diagnosis mistook "did the input-normalization
category of fix help *at all*" (yes, for MLP) for "is input normalization
now correctly solved" (no — the specific normalization applied didn't
address the actual failure mode). Measuring the raw input tensor's own
statistics directly, rather than only measuring downstream training
behavior, is what actually found it.

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
against the new generator checkpoint).

**This table was retrained three times** at the project's earlier,
2,000-path test scale, and each retrain corrected a mistaken conclusion
from the previous one — see the RNN/LSTM sequence below for the full
trail (its numbers predate the paper-scale rerun and are preserved as
history, not current results). **Current numbers, at the paper's own
500,000-path test scale** (Table 1/3), **including P₀** (reproduce with
`python src/backtester/evaluate.py`):

| Strategy | Mean wealth | CVaR 95% | CVaR 99% | Skew | Excess kurtosis | Total tx. cost |
|---|---|---|---|---|---|---|
| Black-Scholes | -0.033 | 1.20 | 1.85 | -2.24 | 7.6 | 3233.0 |
| MLP | -0.037 | 2.38 | 3.69 | -2.08 | 6.9 | 4446.5 |
| Basic RNN | -0.036 | 1.64 | 2.58 | -2.09 | 7.5 | 3863.7 |
| LSTM | -0.042 | 2.17 | 3.49 | -2.19 | 8.5 | 6843.6 |
| GRU | -0.041 | 2.14 | 3.81 | **-24.3** | **3,078.3** | 6576.6 |

**GRU's skew/kurtosis columns were three orders of magnitude off; they're
now about one order of magnitude off.** This row was regenerated after
`grad_clip_norm=1.0`'s full-scale fix attempt was promoted to the
production checkpoint (skew was **-115.4**, kurtosis **24,490** before; see
[the fix attempt](#fix-attempt-grad_clip_norm-substantially-improves-gru-wgan-gp-does-not-fully-close-it)).
Every other cell in this table looks like a normal fat-tailed P&L
distribution (skew around -2, kurtosis 7-8.5); GRU's mean wealth (-0.041)
and CVaR₉₅ (2.14) still look completely ordinary, and its skew/kurtosis,
while much improved, are still measurably elevated — the fix is
substantial, not complete. See [Catastrophic tail risk, invisible below
~500,000 test paths](#catastrophic-tail-risk-invisible-below-500000-test-paths)
below for the full investigation, scope across other checkpoints, and the
control that rules out a test-set artifact.

**P₀ is now included here too** ([Terminal wealth and the P₀ (premium)
term](#terminal-wealth-and-the-p₀-premium-term) covered Part I only when
first written; `evaluate.py` now estimates it via Monte Carlo —
`environment/market_env.py::estimate_premium_monte_carlo`, 500,000 paths
through the same regime-switching simulator used for the backtest itself,
chunked to keep memory bounded). No retraining was needed to get the table
above: a constant additive shift to wealth doesn't change the
CVaR-minimizing optimal policy, only the reported scale, so the *existing*
checkpoints (trained before this extension) were simply re-evaluated
through the premium-aware environment. This estimate run's premium was
0.663 — within the expected ~1% sampling noise of the 0.690 value verified
earlier, not a different number. Every mean wealth above is now near zero,
matching Part I's convention, and directly comparable to it for the first
time. The Basic RNN CVaR₉₉ control-variate numbers in the next subsection
predate this extension and are offset by this same constant (documented
there rather than regenerated, since they're presented as a historical
record of a since-superseded sweep, not current canonical numbers).

**Round 1 (moment-matching loss only)**: produced the table originally
here, claiming "MLP and GRU condition on market state, Basic RNN/LSTM
structurally don't."

**Round 2 (the RNN/LSTM DC-dominance fix)**: every `RecurrentHedgingAgent`
needed retraining since the input transform changed for all three cell
types. LSTM's CVaR₉₉ dropped from 18.37 to 5.02 — a complete turnaround,
matching GRU, and direct evidence the DC-dominance fix (not something
GRU-gating-specific) was the real lever all along. **Basic RNN did not
improve** (19.61 → 19.26) even with 5x more training, and the conclusion at
the time was "vanilla RNN's own well-known architectural limitation."

**That conclusion was also wrong**, and the error is instructive: every
single training run in this project, for every architecture and every
experiment, used the same default `--seed 0`. Testing Basic RNN's
stress-test training across 8 seeds (`--seed 0` through `7`, otherwise
identical settings) reveals its convergence is not architecturally blocked
at all — it's **highly seed-sensitive and bimodal**: some seeds converge to
a GRU/LSTM-competitive policy (delta span up to 0.999, CVaR₉₉ as low as
4.25), others get fully stuck (span exactly 0.0000, CVaR₉₉ ≈ 19-20). Seed 0
— the one every prior experiment in this repo happened to use — is one of
the stuck ones. A single-seed test cannot distinguish "architecturally
impossible" from "unlucky initialization," and round 2 mistook the latter
for the former.

### Fixing the seed-sensitivity: a CVaR control-variate baseline

Kim (2021)'s own suggested future work is an actor-critic (A2C/A3C)
variance-reduction baseline — but that assumes a stochastic-policy,
REINFORCE-style training setup this codebase doesn't use (see
`math_spec.md` section 6 for the full adaptation). The version implemented
here (`PolicyTrainer`'s `use_bs_baseline`) trains on
`CVaR_alpha(policy_wealth - black_scholes_wealth)`, both wealths computed on
the identical sampled price path, instead of raw `CVaR_alpha(policy_wealth)`
— using the closed-form Black-Scholes hedge as a zero-approximation-error
baseline (not a learned critic, since the exact value function is available
here) to cancel shared market-driven noise and reduce variance in which
paths CVaR's sparse gradient selects as "worst," batch to batch.

Same 8-seed sweep, with vs. without this flag (this sweep predates the
stress test's P₀ extension above; every number below carries the old
`premium=0.0` convention rather than the current one, a known constant
-0.663 offset from today's canonical numbers, and is preserved as the
historical record of this comparison rather than regenerated):

| | No baseline | `--use-bs-baseline` |
|---|---|---|
| CVaR₉₉ mean | 8.57 | **5.18** |
| CVaR₉₉ std across seeds | 6.89 | **1.20** (5.7x lower) |
| CVaR₉₉ worst seed | 20.14 | **7.27** |
| CVaR₉₉ best seed | 4.25 | 3.91 |
| Fully-stuck seeds (span < 0.1) | 2/8 (25%) | 1/8 (12.5%, and less severe) |

The technique doesn't uniformly outperform every individual seed (a couple
of already-lucky seeds do marginally better without it), but it
substantially and consistently **reduces the variance of the outcome**:
mean CVaR₉₉ drops 40%, cross-seed standard deviation drops 5.7x, and —
critically — the worst-case seed's CVaR₉₉ improves from catastrophic
(20.14, an essentially unhedged position) to merely mediocre (7.27 at the
time, ~~6.61 under the current P₀-inclusive convention~~ — see the
correction below). Retraining the project's canonical seed-0 checkpoint with
this flag: CVaR₉₉ 19.26 → 7.27. It turns Basic RNN from a coin-flip
between "works" and "completely broken" into a consistently
mediocre-but-functional policy — a genuine, measured win for the technique
at the time — though see below for why "the number in the table above" no
longer matches this document's own current main table.

**Correction: the "6.61" figure above is stale, and now the reason is
established.** Directly measuring the checkpoint currently at
`checkpoints/hedging_agent_rnn.pt` (same file used to generate the current
[Stress-test table](#stress-test-backtest)) gives CVaR₉₉ **2.575** —
matching that table's Basic RNN row (2.58) exactly, not the "6.61" claimed
here. `git log -S"7.27"` (the pre-P₀ number "6.61" was derived from) dates
this 8-seed sweep to commit `c810358` (Aug 6 23:30) — *before* this
project's paper-scale rescaling in `5928efb`, which changed training budget
(200 → 25,000 epochs, `batch_size` 64 → 1,000) and the stress test's own
evaluation batch (2,000 → 500,000 paths) together in the same commit,
confirmed by reading its diff directly. The checkpoint on disk now (mtime
Aug 7 20:10, `epochs: 25000` in its saved args) postdates that rescaling —
it was retrained at the paper's full training budget and re-evaluated at
the paper's full test-set scale sometime after this sweep ran, the same
kind of change that improved several other checkpoints in this project.
The "→ 6.61 today" annotation was simple arithmetic (7.27 − 0.663 for the
P₀ convention change) applied to the *pre-rescaling* sweep's number, never
re-verified against the checkpoint that was later retrained and
re-evaluated at paper scale — the same class of staleness this document
already flags for the RNN/LSTM table earlier ("predates the paper-scale
rerun"), just not
caught here until now. **The reliable number is the current main table's
2.58** (verified via direct re-measurement, seed=42, 500,000 paths), and
that's the baseline used below.

This is the third correction of this kind in this project's RNN/LSTM
investigation (see the moneyness-fix self-correction in Part I, and the
TimeGAN GRU-attribution retraction below) — worth stating plainly rather
than smoothing over: single-seed conclusions about *why* a specific
architecture fails are unreliable in this codebase's training regime, and
should be treated as provisional until checked across multiple seeds.

#### Follow-up: `use_bs_baseline` + `orthogonal_init`, the deferred 8-seed experiment — a negative result

[Known limitations](#known-limitations) item 4 explicitly deferred testing
whether `orthogonal_init` on top of `use_bs_baseline` would close Basic
RNN's remaining gap further, citing compute cost. Run to completion (8
seeds, full 25,000-step scale, `--use-bs-baseline --orthogonal-init`,
otherwise identical to the canonical checkpoint's hyperparameters):

| | Current incumbent (`use_bs_baseline` alone, seed=0) | `use_bs_baseline` + `orthogonal_init` (8 seeds) |
|---|---|---|
| CVaR₉₉ | 2.575 | mean 3.616, min 2.727, max 4.598 |
| CVaR₉₅ | 1.641 | mean 2.229 |
| Catastrophic paths (< -50) | 0/500,000 | 0/500,000 for 7 seeds, **2/500,000** for seed 0 |
| Worst loss | -7.34 | ranges -9.05 to -74.78 |

**Every one of the 8 new seeds does worse than the current incumbent on
CVaR₉₉, including the best one** (2.727 vs. 2.575) — that direction is
well-evidenced (8-for-8). The *size* of the effect is less certain than it
looks: the incumbent is a single seed-0 draw, not a distribution, and this
sweep's own cross-seed std (0.625) shows meaningful seed-to-seed spread —
so the mean-to-mean gap (3.616 vs. 2.575) could overstate the true effect
if the incumbent happens to be a favorable draw. `orthogonal_init` does not
help here; whether it actively hurts or merely fails to help is less
certain than "every seed lost" alone establishes. This closes the question
item 4 left open, with a negative answer rather than the hoped-for further
variance reduction. No checkpoint was promoted, since none beat the
incumbent — `checkpoints/hedging_agent_rnn.pt` is unchanged.

**Not comparable to this section's own historical 6.89/1.20 cross-seed std
figures.** `orthogonal_init` consumes additional random draws during weight
initialization (`nn.init.orthogonal_` calls), shifting the RNG stream for
every subsequent training step — so "seed 0 with `orthogonal_init`" isn't a
controlled single-variable ablation of "seed 0 without it," any more than
two different seeds are directly comparable. The old table also predates
changes this document already flags as unregenerated (see its own note
above). This new sweep's cross-seed std (0.625) is reported as a fresh,
independent measurement of Basic RNN's seed sensitivity under the *current*
codebase — the first such measurement this document has, since the
original 8-seed sweep is explicitly historical — not as something to
difference against the old figure.

**Practical implication**: the incumbent `hedging_agent_rnn.pt` (seed=0,
`use_bs_baseline` alone, no `orthogonal_init`) remains the best available
Basic RNN checkpoint under this stress test — and per the corrected numbers
above, there may no longer be a gap to close at all. The current main
table's CVaR₉₉ is 2.58 for Basic RNN, actually *better* than both LSTM
(3.49) and GRU (3.81), the reverse of this document's stale "doesn't close
the gap to LSTM/GRU's ~4.4" framing. The incumbent's paper-scale retrain
(see the correction above) already resolved the practical priority this
deferred item was chasing, independent of this `orthogonal_init`
experiment.

### Catastrophic tail risk, invisible below ~500,000 test paths

The stress test above uses the paper-matched 500,000-path test batch for
the first time (previously 2,000, chunked to stay CPU-feasible — see
`MarketEnvironment.simulate`'s `chunk_size` docstring). Rerunning at this
scale surfaced something the smaller batch never could: GRU's skew (-115.4
at the time) and excess kurtosis (24,490 at the time) were three orders of
magnitude out of line with every other architecture (skew ≈ -2.1, kurtosis
6.9-8.5), while its mean wealth and CVaR₉₅ looked completely ordinary. (GRU
(WGAN-GP) was since diagnosed and partially fixed — see the table below and
[the fix attempt](#fix-attempt-grad_clip_norm-substantially-improves-gru-wgan-gp-does-not-fully-close-it)
— but this paragraph is left describing the discovery as it happened.)

**Not a fluke of this one checkpoint.** Scanning every trained checkpoint
against the same seed-42, 500,000-path stress-test batch and counting
paths with wealth below -50 (a loss > 25x the option premium) turns up the
same pattern in several other places the summary tables above don't show
on their own, since the main table and the alpha-sweep/TimeGAN tables are
never printed side by side:

| Checkpoint(s) | Paths < -50 | Paths < -10 | Worst loss | mean tx. cost |
|---|---|---|---|---|
| MLP, Basic RNN, LSTM (WGAN-GP) | 0 / 500,000 | 0-4 | ≤ -12.8 | normal |
| ~~**GRU (WGAN-GP)**~~ **GRU (WGAN-GP) — substantially improved, not fully fixed** | ~~34~~ **4** / 500,000 (~~0.0068%~~ 0.0008%) | ~~327~~ **97** | ~~-417.5~~ **-137.5** | normal (0.0132) |
| α ∈ {0.5, 0.75, 0.9, 0.95, 0.995} (MLP) | 0 / 500,000 | 3-4 | ≤ -12.8 | normal |
| ~~**α = 0.99 (MLP)**~~ **α = 0.99 (MLP) — fixed, see below** | 0 / 500,000 | ~~**5,452**~~ 0 | ~~-48.7~~ -9.7 | ~~normal, but ~half the rest (0.0044)~~ normal (0.0100) |
| ~~**α = 0.997 (MLP)**~~ **α = 0.997 (MLP) — fixed, see below** | ~~814 / 500,000 (0.16%)~~ 0 / 500,000 | ~~5,257~~ 0 | ~~-6202.5~~ -9.7 | ~~0.0000~~ normal (0.0115) |
| MLP (TimeGAN) | 0 / 500,000 | 0 | ≤ -8.8 | normal |
| **Basic RNN (TimeGAN)** | 238 / 500,000 (0.048%) | 1,849 | -2564.7 | normal (0.0018) |
| **LSTM (TimeGAN)** | 793 / 500,000 (0.16%) | 4,754 | **-6202.4** | normal (0.0093) |
| **GRU (TimeGAN)** | 578 / 500,000 (0.12%) | 3,608 | -6033.3 | normal (0.0111) |

α=0.99's "0 / 500,000" at the -50 threshold looks clean at a glance —
its 5,452-path count at the -10 threshold (vs. 3-5 for every clean
checkpoint) is what actually flags it; it does not belong in the same row
as the genuinely unaffected checkpoints above it.

**Two distinct mechanisms, confirmed rather than guessed:**

1. **CVaR-α training starves at extreme α.** `PolicyTrainer`'s CVaR loss
   only backpropagates through the worst `(1-α)` fraction of each training
   batch (`batch_size=1000`, the paper's own Table 1/3 spec). At α=0.997
   that's **3 paths per gradient step** — too sparse and noisy a signal to
   learn anything beyond "don't bother." `mean_transaction_cost` isn't just
   small, it's *exactly* zero (`9.7e-15`, floating-point noise), confirming
   this checkpoint learned a fully degenerate never-hedge policy: its worst
   loss (-6202.48) matches the closed-form **unhedged** loss on the single
   most extreme path in the test set — `premium - (S_T - K)` =
   `0.663 - (6204.15 - 1)` = **-6202.48**, to 5 significant figures.
   α=0.99 (10 tail samples/step) is a related but *not identical* failure:
   its worst-case loss stays bounded (-48.7, nowhere near -6202) and it
   isn't a confirmed never-hedge policy — but transaction cost is already
   roughly halved and 5,452 paths sit below -10 (vs. 3-5 for every other
   α), consistent with under-hedging across a wide swath of the tail
   rather than α=0.997's complete collapse on the very worst few paths. Both
   point at the same sparse-gradient mechanism, but α=0.99's signature is a
   thickened shoulder, not a degenerate spike — the two shouldn't be
   conflated as "the same thing at different severity" without more
   evidence (e.g. checking whether α=0.99's `mean_transaction_cost` is
   uniformly lower across the whole batch or concentrated in the same
   handful of paths). This isn't an artifact of this codebase's own
   conventions — the paper's own Table 3 trains at α up to 0.997 with this
   same batch size, so a faithful reproduction of the paper's own setup
   hits the same sparse-gradient wall.
2. **TimeGAN-trained recurrent policies generalize badly to the stress
   test's price extremes.** Basic RNN and LSTM are clean under WGAN-GP
   training but catastrophic under TimeGAN (GRU is mildly affected under
   both generators). All four TimeGAN checkpoints were confirmed trained at
   the same α=0.95 as their WGAN-GP counterparts (`policy_args["cvar_alpha"]`
   read directly from each checkpoint) — so this is not mechanism 1 in
   disguise; it's a distinct effect specific to the generator. Unlike
   α=0.997, these checkpoints have *normal* transaction costs — they
   aren't refusing to hedge, they're hedging *badly* on paths well outside
   anything TimeGAN's training distribution produced. LSTM's worst loss
   lands within $0.07 of the fully-unhedged number (-6202.41 vs. -6202.48)
   apparently by coincidence — on the single most extreme test path, its
   accumulated hedge P&L happened to net to ≈0, not because it wasn't
   hedging (mean tx. cost 0.0093, normal for this architecture) but
   because whatever it did there didn't help. TimeGAN-MLP, which only ever
   sees the instantaneous price ratio and never sequence history, is
   unaffected — consistent with a
   recurrence-specific extrapolation failure, not a shared data problem.
   (**Less coincidental than it looked at the time**: a later diagnosis
   found LSTM's delta on this exact path collapses from 0.9998 to 0.0001 in
   a single step, at log-moneyness just 0.071 — still comfortably inside
   TimeGAN's own training range — and stays near zero for the remaining 27
   steps of the rally. Given that, netting to ≈0 hedge P&L on an almost
   entirely unhedged path is closer to expected than coincidental; see the
   [follow-up
   diagnosis](#follow-up-diagnosis-mechanism-b-is-a-sharp-cliff-at-timegans-training-distribution-boundary-not-a-gradual-generalization-failure)
   below for the precise mechanism and threshold.)

**Verified against a control that rules out a test-set artifact.**
Black-Scholes' closed-form delta hedge on the identical seed-42 batch
(same 500,000 price paths, several reaching S_T > 6,000 under the stress
test's 60%-vol regime) shows **zero** paths below -50 and a worst loss of
just **-4.35**. If the extreme test paths themselves were the problem,
Black-Scholes — an analytic hedge with no training or generalization
dependency — would show it too. It doesn't, which rules out "the
500,000-path test set just contains absurd prices" and confirms this is
genuine policy behavior: the affected checkpoints fail to hedge well on
paths the smaller, 2,000-path test batch was simply too small to ever
sample (expected frequency here ranges from roughly 1-in-650 to
1-in-15,000 depending on the checkpoint).

This was not caught by any test in this repo, nor by any summary statistic
reported elsewhere in this document until now — mean wealth and CVaR₉₅
both look unremarkable for every affected checkpoint; only the tail count
and skew/kurtosis columns show it. The diagnostic scripts used for the
checkpoint scan above were originally run ad hoc via the shell, not
committed as tests — **now fixed**: `evaluate.py::tail_risk_summary` and
`evaluate.py::scan_checkpoint_tail_risk` make the scan itself reproducible
from committed code (callable directly — deliberately *not* wired into
`evaluate.py`'s own `__main__`, since it would re-simulate paths and
re-estimate the premium per checkpoint group on top of what
`run_backtest`/`run_alpha_sweep_backtest` already compute there, roughly
tripling that script's already-500,000-path cost), and
`tests/test_tail_risk.py` turns it into a real regression suite: a
threshold-counting unit test that needs no trained checkpoints, a guard
over every known-good checkpoint (MLP/Basic RNN/LSTM WGAN-GP, MLP TimeGAN)
asserting zero paths below -50, and a canary over every known-bad one (GRU
WGAN-GP; Basic RNN/LSTM/GRU TimeGAN; the α=0.997 degenerate never-hedge
checkpoint) asserting the documented failure is still present — so a
future fix shows up as an intentional, informative test failure instead of
silently going unnoticed. Trained checkpoints are gitignored, so these
skip cleanly (not fail) on a fresh clone or in CI where none exist yet.
The regression suite itself runs at a reduced 50,000-path scale for speed
(vs. the 500,000 used above and in RESULTS.md's own scan) — checked
directly to still reproduce every known-bad checkpoint's nonzero tail
count at that scale, but note this is a sensitivity limit: a *newly*
regressed checkpoint with a low enough catastrophic-path rate (e.g. the
~1-in-15,000 end of the range measured above) could still read as clean
at 50,000 paths and pass.

### Mechanism (a), root-caused and fixed: sigmoid output saturation, not sparse gradients

The α=0.997 degenerate never-hedge policy above was originally attributed
to CVaR training's sparse gradient at extreme α (3 tail paths/step at
batch=1000). This section's own "Ideas for future work" offered two fixes:
importance sampling toward the tail, or simply a larger batch size at high
α. The larger-batch fix was tried first, as the cheaper one — and it made
things *worse*, which is itself the finding that led to the real root
cause.

**The batch-size hypothesis, falsified.** `HedgingAgent`'s own delta-span
metric (max − min hedge ratio across a spot grid, the same diagnostic used
throughout this document) was instrumented directly into the training loop
and logged every 100 steps, rather than relying on a single post-hoc
number. At α=0.997, batch=1,000 (seed 0, 3,000 steps): delta span
*oscillates* continuously between fully collapsed (0.0000) and healthy
(up to 0.99) — 7 collapse windows out of 30 logged. This reframed the
question: the checkpoint isn't converging to a stable degenerate optimum,
it's unstable and happens to land badly wherever training stops. Batch
10,000, same seed, same step budget: **the collapse became permanent**,
not rarer — 0 collapse windows in the first 11, then a solid run of 8
consecutive collapsed windows (800 steps) with the gradient norm reading
*exactly* `0.000000` (both mean and max over each 100-step window) through
every one of them. A second seed reproduced the same qualitative pattern
in both arms. Fewer, less-noisy gradient updates made the failure *more*
stable, the opposite of the variance-reduction intuition that motivated
trying a larger batch in the first place.

**The actual mechanism, confirmed by inspecting the real checkpoint
directly**: `HedgingAgent`'s output is `sigmoid(raw_output)`
(`hedging_agent.py`). Loading the committed, degenerate
`hedging_agent_mlp_alpha0_997.pt` and evaluating its pre-activation
logits across the same spot grid gave **raw logits around −250 to −260**
— so deeply saturated that `sigmoid'(x)` underflows to *exactly* `0.0` in
float32 (confirmed directly: `torch.sigmoid(x)*(1-torch.sigmoid(x))`
evaluates to `0.0` at these logits, not merely small). This is a genuine
numerical dead end, not a noisy-but-nonzero signal: once weights push the
logit this negative, **no gradient can reach them again**, regardless of
batch size, `h`'s value, or how many paths violate the CVaR threshold —
because the chain rule multiplies by a factor that has literally rounded
to zero. CVaR's `1/(1-α)` loss amplification (×333 at α=0.997) combined
with Adam's `lr=1e-2` is enough to jump a weight update this far in a
single oversized step (`grad_norm` spikes into the hundreds were observed
immediately before every collapse in the logs). A larger batch makes this
*worse*, not better, because it doesn't change how far a single outsized
step can push the logit — it just means there are more, not fewer,
identically-saturated samples all producing the same zero gradient once
the collapse happens, which is consistent with the observed longer,
harder-to-escape collapse.

**The fix already existed in this codebase and was never wired up.**
`PolicyTrainer.__init__`'s `grad_clip_norm` parameter has a docstring that
names this exact failure mode almost verbatim — "the policy gets stuck
outputting a near-constant value from the first few steps onward, with
zero further movement no matter how many epochs follow" — written for the
RNN/LSTM DC-dominance investigation earlier in this document. But
`train_policy.py`'s CLI never exposed it: `_train_and_save` always called
`PolicyTrainer(...)` without passing it, so **every checkpoint in this
repo has always trained with gradient clipping disabled**, including the
plain feed-forward `HedgingAgent`, which the docstring's own framing
(recurrent gates specifically) didn't anticipate would need it too. Added
a `--grad-clip-norm` CLI flag (`train_policy.py`) that wires straight
through to the existing parameter — no new mechanism, just connecting one
that was already there.

**Verified before committing to a full retrain.** Same instrumented setup,
α=0.997, batch=1,000 (the paper's own Table 3 batch size — no deviation
needed once this was the actual fix), 3,000 steps, two seeds:
`grad_clip_norm ∈ {1.0, 5.0}` produced **zero** collapse windows across 90
logged checkpoints (2 seeds × 2 clip values × ~22-30 windows each), versus
7-10 collapses per 30 windows unclipped at each seed. Both clipped runs
ended with delta span 0.92-0.97 and steadily positive, growing mean
wealth, instead of oscillating. This generalization check (clipping
prevents collapse, not just "this one seed's final checkpoint happened to
land clean") rests on 2 seeds × 3,000 steps — narrower than this
document's own standard elsewhere for architecture-level claims (the
Basic RNN seed-sensitivity investigation used 8 seeds); treat "clipping
reliably prevents this collapse" as well-supported but not exhaustively
checked.

**Production retrain, paper scale**: `hedging_agent_mlp_alpha0_997.pt`
retrained with `--grad-clip-norm 1.0`, otherwise identical settings
(batch=1,000, 25,000 steps, seed=0) to every other checkpoint in this
document. Unlike the reduced-budget probes above, the production CLI run
doesn't log delta span directly — but the collapsed state's signature
(`cvar_loss` flat in the 0.20-0.23 dead-zone band, `mean_wealth` pinned
within 0.002 of zero) is visible in `cvar_loss`/`mean_wealth` alone, and
neither appears anywhere in this run's 252 logged points across all 25,000
steps (minimum logged `mean_wealth` was -0.0196, at epoch 1 before
training ramps up; every later point is positive and none sit near zero).
So this is a direct, checked claim about the actual production run, not
an inference from the shorter probes: **this specific training run never
entered the collapsed state at all**, rather than entering and reliably
escaping it. Verified against the same seed-42, 500,000-path stress test
used for the checkpoint scan above:

| | Before (unclipped) | After (`grad_clip_norm=1.0`) |
|---|---|---|
| Paths < -50 | 814 / 500,000 (0.16%) | **0 / 500,000** |
| Worst loss | -6202.5 | **-9.7** |
| mean transaction cost | 0.0000 (exactly) | **0.0115** (normal) |
| CVaR₉₅ / CVaR₉₉ | (degenerate; not comparable) | 2.22 / 3.46 |
| delta span | 0.0000 | **0.9988** |

Every number now lands in the same range as the other clean α=0.5-0.995
checkpoints (see the updated alpha-sweep table below) — this isn't a
partial improvement, the checkpoint is fully repaired.
`tests/test_tail_risk.py` was updated accordingly: `MLP (alpha=0.997)`
moved from the known-bad canary list to the known-good guard list, and the
old degenerate-policy assertion now asserts the opposite (see that file's
own comments for what a future regression here should look like).

**A mistake made and caught during this fix, worth recording plainly
rather than smoothing over**: the first production retrain command omitted
`--alpha-sweep`, so `train_policy.py`'s default single-run checkpoint path
(`checkpoints/hedging_agent.pt` for `architecture=mlp`) was used instead of
the intended `hedging_agent_mlp_alpha0_997.pt` — silently overwriting the
main comparison table's MLP checkpoint (α=0.95) with the α=0.997 run,
without a backup taken first. Recovered by (1) moving the wrongly-placed
but correctly-trained file to its intended destination — which is exactly
the fix this section describes, so no work was lost there — and (2)
regenerating `hedging_agent.pt` from its known, recorded settings (α=0.95,
same seed=0, same untouched `market_gan.pt` generator checkpoint, read
directly off sibling checkpoints' saved `args`). The regenerated file was
verified against this document's own main stress-test table
(`mean_wealth`, CVaR₉₅/₉₉, skew, kurtosis, and total transaction cost all
matched to the last reported digit) before treating the recovery as
complete.

**Scope at the time this fix first landed: confirmed on one checkpoint
only.** GRU (WGAN-GP)'s milder tail issue and α=0.99's thickened-tail
warning sign (see the checkpoint scan table above) were initially left
untouched, since neither had been checked. Both were subsequently checked
directly (not guessed) — see [Extending the
fix](#extending-the-fix-alpha099-confirmed-same-mechanism-gruwgan-gp-and-basic-rnntimegan-confirmed-different-ones)
in the multi-alpha sweep section below for the full results: α=0.99 turned
out to be the same mechanism (now also fixed), GRU (WGAN-GP) confirmed
*not* to be. Mechanism (b) (TimeGAN-trained recurrent policies generalizing
badly to price extremes) remains a wholly separate failure mode, untouched
by this fix — see [Known limitations](#known-limitations) item 5 and [Ideas for future
work](#ideas-for-future-work).

### Multi-alpha risk-return sweep, extended to the paper's own Part II grid

The paper's Part II tests risk aversion at α ∈ {0.5, 0.75, 0.99, 0.995,
0.997} — noticeably more extreme than this repo's original sweep of {0.5,
0.75, 0.9, 0.95, 0.99}, which never exercised the two most extreme levels
at all. Extended by training two more MLP checkpoints
(`train_policy.py --architecture mlp --alpha-sweep 0.995,0.997`) and
rerunning the sweep backtest under the same, now paper-matched,
500,000-path stress-test conditions as the main table above (reproduce with
`python src/backtester/evaluate.py`):

Includes P₀ (see [above](#terminal-wealth-and-the-p₀-premium-term)):

**Regenerated after both the α=0.997 and α=0.99 fixes** (below); α=0.5-0.995
excluding 0.99 are unchanged checkpoints, re-evaluated on the same seed-42
batch and matching the previous table to the last reported digit — included
here as confirmation the fixes had no side effects on the others, not as new
results:

| α | Mean wealth | CVaR 95% | CVaR 99% | Skew | Excess kurtosis |
|---|---|---|---|---|---|
| 0.50 | -0.037 | 2.32 | 3.61 | -2.07 | 6.9 |
| 0.75 | -0.039 | 2.25 | 3.50 | -2.21 | 7.7 |
| 0.90 | -0.038 | 2.25 | 3.50 | -2.17 | 7.4 |
| 0.95 | -0.037 | 2.38 | 3.69 | -2.08 | 6.9 |
| ~~0.99~~ **0.99 (fixed)** | ~~-0.031~~ -0.038 | ~~7.99~~ 2.25 | ~~14.33~~ 3.50 | ~~-5.00~~ -2.13 | ~~35.7~~ 7.2 |
| 0.995 | -0.038 | 2.28 | 3.54 | -2.13 | 7.2 |
| ~~0.997~~ **0.997 (fixed)** | ~~-0.031~~ -0.039 | ~~**11.76**~~ 2.22 | ~~**43.15**~~ 3.46 | ~~**-248.9**~~ -2.21 | ~~**80,781**~~ 7.7 |

**Every α in this table is now clean** — the entire risk-aversion sweep
lands in the same tight range regardless of α, with no outliers. α=0.99
turned out to be the same mechanism as α=0.997 (confirmed, not assumed —
see [below](#extending-the-fix-alpha099-confirmed-same-mechanism-gruwgan-gp-and-basic-rnntimegan-confirmed-different-ones)),
just less severe (a raw pre-sigmoid logit range of roughly [-718, -43]
depending on time step, vs. α=0.997's ≈-250 — deeper saturation than
α=0.997 had, somewhat surprisingly, despite the milder stress-test damage).
Retrained the same way (`grad_clip_norm=1.0`, same paper-scale batch=1,000,
25,000 steps) and verified at the full 500,000-path scale: 0 catastrophic
paths (was 5,452 below -10), worst loss -9.7 (was -48.7). As with α=0.997,
the production run's own log was checked directly for the collapse
signature (`mean_wealth` pinned near zero, `cvar_loss` flat in the
dead-zone band) across all 252 logged points spanning the full 25,000
steps — absent throughout (`mean_wealth` ranged -0.0196 to 0.0530, always
outside the collapse band), so this run never entered the collapsed state,
matching the α=0.997 evidence standard rather than inferring cleanliness
from the final checkpoint alone. Each checkpoint
in this table is still a single training run at a single seed — this
project has already found (three times, in the RNN/LSTM investigation
above) that single-seed training outcomes here are noisy enough to matter,
so "every α is clean" describes this specific seed, not a proven general
property of `grad_clip_norm=1.0` at every α. Nor was this tested against
the paper's own numbers at these α levels (the paper's Part II uses
TimeGAN and Basic RNN specifically, not this repo's WGAN-GP+MLP
combination), so no match/mismatch verdict is claimed either — only that
the sweep now covers the risk-aversion range the paper cares about, at the
paper's own test scale, instead of stopping short of it.

### Extending the fix: α=0.99 confirmed same mechanism, GRU(WGAN-GP) and Basic RNN(TimeGAN) confirmed *different* ones

Rather than guessing which of the other flagged checkpoints (GRU (WGAN-GP),
α=0.99, and the three TimeGAN recurrent checkpoints) shared α=0.997's
sigmoid-saturation mechanism, each was checked directly with the same
training-free diagnostic that found and fixed it: load the checkpoint,
sweep a spot grid across several time steps, and read off the raw
pre-activation logit (for `HedgingAgent`) or hidden state (for
`RecurrentHedgingAgent`) alongside delta span. This took seconds per
checkpoint and reordered the whole remaining list before any retraining:

- **α=0.99 (MLP): confirmed saturated**, even more severely than α=0.997 —
  logits as extreme as [-718, -684] at t=0, span exactly 0.0000 at every
  timestep checked. Retrained with the same fix; verified clean above.
- **GRU (WGAN-GP): confirmed *not* saturated** by the spot-grid diagnostic
  (delta span 1.0000, logits in a moderate [-16, 21] range) — but that only
  ruled out mechanism (a); the actual mechanism was left "still
  uncharacterized" at the time. A follow-up diagnosis pass (below) has since
  characterized it directly, still without attempting a fix.

#### Follow-up diagnosis: GRU (WGAN-GP) is a GRU-specific hidden-state recovery lag, not saturation

**This entire subsection describes the *pre-fix* checkpoint** (preserved as
`checkpoints/hedging_agent_gru.pt.bak-pre-recovery-lag-fix` after the fix
below was promoted) — `checkpoints/hedging_agent_gru.pt` itself now refers
to the improved, `grad_clip_norm=1.0`-trained checkpoint. The diagnosis
here is what motivated and explains the fix, so it's left in present tense
throughout rather than rewritten past-tense; just don't expect to reproduce
these exact numbers by loading the current `hedging_agent_gru.pt`.

Rather than leave GRU (WGAN-GP)'s 34/500,000 catastrophic paths (worst loss
-417.5) as an unexplained residual, its own worst-loss paths from the
500,000-path scan (seed=42) were pulled out directly and inspected, then the
pattern found was tested with controlled synthetic probes:

- **Every one of the 10 worst paths shares the same shape**: an early
  downward move (log-moneyness dipping to roughly -0.4 to -2.0 within the
  first ~10 steps) followed by a large rally (final log-moneyness +4.0 to
  +5.3 — the call finishes deep in the money).
- **This is not simply "extreme log-moneyness confuses the network."** A
  smooth, monotonic ramp straight up to the same extreme levels (no
  preceding down-move) reaches delta=1.0 correctly every time, including
  under heavy added noise (tested up to amplitude 2.0 — more than double the
  down-move that breaks the down-then-rally case). The down-move
  specifically triggers the failure, not the eventual price level.
- **A down-depth sweep** (down-move to a given log-moneyness, then a smooth
  ramp to the same +4.87 target used by the actual worst path) shows a sharp
  threshold: recovers to delta≈1.0 for a dip to -0.4, degrades to 0.83 at
  -0.6, and collapses to 0.04 at -0.71 (the actual worst path's dip) and
  below 0.001 for dips of -1.0 or deeper. Measured against 200,000 fresh
  samples from the same WGAN-GP generator this checkpoint trained against,
  -0.6 sits at that generator's own 0.77th percentile and -0.71 at its
  0.35th percentile — pooled across all (path, timestep) observations, not
  per-path minima, but still genuinely rare for this generator's own output,
  not an artifact of the synthetic probe. (This threshold is specific to the
  smooth-ramp construction used to measure it, not a general property of any
  path reaching that depth.)
- **This is architecture-specific, not shared by LSTM.** `hedging_agent_lstm.pt`
  was checked against `hedging_agent_gru.pt`'s saved training args directly:
  identical in every hyperparameter (seed=0, 25,000 steps, `rnn_hidden_dim`=64,
  `rnn_num_layers`=2, `use_bs_baseline`=False, same generator checkpoint) except
  `architecture` itself and the Monte Carlo premium estimate (0.06580 vs.
  0.06578, noise-level difference) — and confirmed clean at 0 catastrophic
  paths on this exact 500,000-path stress test. Run through the identical
  down-depth sweep and the identical worst-path reproduction, LSTM recovers
  to delta≈1.0 within 1-2 steps at every tested depth, including the -0.71
  shock that collapses GRU to 0.04 for the remaining ~26 steps of the same
  window, and even at -1.5 (deeper than any dip in the actual worst-path
  scan). Same generator, same shock, same extremity relative to the shared
  training distribution — only GRU fails to recover on this timescale. Note
  LSTM still needs *some* downward move to see any effect at all (delta=1.0
  immediately at down_depth=0.0), so this isn't "GRU's gating is simply
  broken" — it's specifically the interaction between an out-of-distribution
  downward shock and GRU's gating dynamics that LSTM's gating doesn't share.
  (`hedging_agent_gru_timegan.pt` was not checked for the same signature —
  if this is a general GRU gating property, it should show it too, but that
  wasn't tested here.)
- **The defect is recovery *lag*, not permanent pinning** — and not always
  "delta stays near 0" at the end, either. 2 of the top-10 worst paths (idx
  288684, 492971) show delta eventually recovering to 0.95-0.97 by the end
  of the path, yet still lose -237.3 and -126.7. Because the price path is
  exponential, the largest absolute price increments happen early in the
  rally — exactly while delta is still catching up — so even a late,
  near-complete recovery misses most of the hedge P&L. Extending the horizon
  well past the paper's 30-step convention (60-90+ steps, an untrained
  regime for this checkpoint) does eventually let delta climb back toward
  1.0, confirming the hidden state isn't permanently stuck: the paper's
  fixed 30-step horizon is simply too short for GRU's own recovery rate
  after this kind of shock.
- ~~**No fix was attempted**~~ — **since attempted and promoted**:
  `grad_clip_norm=1.0` substantially, though not completely, fixes this at
  full training scale, and is now the production `hedging_agent_gru.pt`.
  See [Fix attempt](#fix-attempt-grad_clip_norm-substantially-improves-gru-wgan-gp-does-not-fully-close-it)
  below. The GRU (WGAN-GP) row in the checkpoint scan table above and the
  Stress-test table's GRU row reflect the promoted, improved checkpoint.

#### Fix attempt: `grad_clip_norm` substantially improves GRU (WGAN-GP), does not fully close it

The diagnosis above found the (then-production, now pre-fix, superseded
below) checkpoint's recurrent weight norms were 2.3-3.5x larger than a
3,000-step checkpoint of the same run (e.g. `weight_ih_l1` 73.3 → 169.9),
and that its layer-0 update gate settles into a persistently high value
(~0.93-0.94) right after a downward shock — barely updating the hidden
state each step. That read as circumstantial evidence for an
unclipped-gradient weight-growth story similar to mechanism (a)'s, so
`grad_clip_norm=1.0` was tried as a targeted fix, trained at the same full
scale (25,000 steps, same seed, same generator) as the checkpoint it would
go on to replace, initially into a scratch checkpoint path.

**Reduced-scale (3,000-step) probes were tried first and found
uninformative**: the down-then-rally defect that afflicted the pre-fix
checkpoint does not reproduce at all at 3,000 steps regardless of
`grad_clip_norm`/`orthogonal_init` (the baseline probe recovers immediately
even at a -0.71 dip), unlike Basic RNN (TimeGAN)'s saturation, which *did*
reproduce at reduced scale. That result is recorded here as a methodology
finding, not swept aside: the reduced-scale probe recipe that worked for one
mechanism does not transfer to this one, and any future diagnosis attempt
here needs the full 25,000-step schedule to say anything meaningful.

**The fix works, but not for the reason that motivated trying it.** Final
recurrent weight norms after clipping are barely different from the
unclipped pre-fix checkpoint's (`weight_ih_l0` 50.6 → 46.9,
`weight_hh_l0` 117.5 → 114.9, `weight_ih_l1` 169.9 → 164.7, `weight_hh_l1`
111.9 → 113.1 — one *larger*, not smaller). The weight-growth hypothesis
that motivated this experiment is not what changed; clipping altered the
training *trajectory* and landed on a qualitatively different solution with
essentially the same final weight magnitudes, not a smaller-norm one. This
is recorded as a falsified hypothesis, not a confirmed one -- consistent
with this document's recurring pattern of a plausible-sounding mechanism
turning out not to be the operative one on direct measurement.

**Results, same seed=42 500,000-path scenario used throughout this
document:**

| | Worst loss | Paths < -50 | Paths < -10 | std(wealth) | CVaR₉₅ | CVaR₉₉ |
|---|---|---|---|---|---|---|
| Pre-fix (unclipped) | -417.5 | 34 (0.0068%) | 327 (0.065%) | 1.427 | 2.418 | 5.353 |
| `grad_clip_norm=1.0` — **now production** | **-137.5** | **4** (0.0008%) | **97** (0.019%) | **0.773** | **2.139** | **3.814** |

Every metric improves substantially (worst loss down 67%, catastrophic path
count down 88%, CVaR₉₉ down 29%), and the down-depth sweep diagnostic
confirms why: at every dip depth up to -0.71 (the actual worst path's
depth), the clipped checkpoint's delta stays at or above 0.69 throughout —
no collapse at all, where the unclipped checkpoint fell to 0.0001-0.04. Only
at dips of -1.0 or deeper does the clipped checkpoint still show a *brief*
full collapse (delta hits exactly 0.0), but it now recovers within 1-2
steps instead of never recovering within the 30-step horizon.

**The residual failures are (mostly) the same mechanism, at reduced
severity, not a new one.** The 10 worst paths under the clipped checkpoint
were pulled out directly: 9 of 10 show the familiar down-then-rally shape
(early dip to log-moneyness -0.49 to -2.08) and a `min_delta_after_dip`
of exactly 0.0 — the brief collapse the down-depth sweep predicted for deep
dips — even though most of these paths' delta *does* recover close to 1.0
by the end (final delta 0.68-1.00 in 8 of 10 cases). This is the same
lag-during-the-largest-price-increments story as the unclipped checkpoint,
just shorter and less severe. One of the 10 (idx 59141) does not fit this
pattern at all — log-moneyness rises monotonically from 0 with no preceding
dip, yet still ends with a low final delta (0.02) and a real loss (-33.9) —
a reminder that clipping's partial fix doesn't fully explain every
remaining tail loss, and this residual case wasn't investigated further.

**Promoted to the production checkpoint, tables regenerated, not
hand-edited.** The improved checkpoint is strictly better on every metric
measured here, so it was promoted: the pre-fix checkpoint was preserved as
`checkpoints/hedging_agent_gru.pt.bak-pre-recovery-lag-fix` (same convention
as the mechanism (a) fix's `.bak-pre-gradclip-fix` files) before the
clipped checkpoint replaced `checkpoints/hedging_agent_gru.pt`. Every
affected number in this document (the Stress-test table above, the
checkpoint-scan table above) was then regenerated by actually running
`python src/backtester/evaluate.py` and the tail-risk scan again, not
hand-edited to the numbers already measured during the fix attempt itself —
those two sources agree (CVaR₉₅/₉₉ match to 3 decimal places), which is
itself a useful cross-check that nothing was copied wrong.

**This promoted checkpoint is not "clean."** It's a genuine, substantial
improvement, not a full fix — 4/500,000 catastrophic paths remain (down
from 34), and the residual-failures analysis above found most of them share
the same, now-shorter-but-not-eliminated recovery lag. `checkpoints/` is
gitignored, so nothing about this promotion is visible in `git diff` beyond
this document and the test file below — anyone re-running
`train_policy.py --architecture gru` from scratch without `--grad-clip-norm
1.0` would reproduce the *old*, more catastrophic checkpoint, not this one.
`tests/test_tail_risk.py`'s `_KNOWN_BAD_CHECKPOINTS` entry for GRU no
longer fits (below_-50_count is now 0 at the test suite's 50,000-path scale
— see why in the test file itself) and has been replaced with a dedicated
test at a scale large enough to still catch a regression, rather than being
moved to the known-good list, since 4/500,000 is not actually clean.
- **LSTM/GRU (TimeGAN): confirmed *not* saturated** (span 0.99-1.0, moderate
  logits) — this is the expected signature for mechanism (b) (generalizing
  badly to price extremes the training distribution never produced, not an
  output-layer pathology), consistent with these checkpoints' documented
  *normal* transaction costs. ~~No fix attempted; still open~~ — **since
  characterized precisely**: this spot-grid diagnostic evidently didn't
  probe far enough to find the actual cliff. See [the follow-up
  diagnosis](#follow-up-diagnosis-mechanism-b-is-a-sharp-cliff-at-timegans-training-distribution-boundary-not-a-gradual-generalization-failure)
  below. Still no fix attempted; still open, still a training-distribution
  problem, not a `grad_clip_norm` problem.
- **Basic RNN (TimeGAN): confirmed saturated, but through a third, distinct
  mechanism** — not `HedgingAgent`'s sigmoid output, but
  `RecurrentHedgingAgent`'s recurrent *hidden state*. Its raw logit was
  *exactly* constant (`0.349268`, to the same six decimal places) across
  every spot price tested, even sweeping the input across an absurdly wide
  range (moneyness 0.02-50). Direct inspection found why: the vanilla RNN's
  hidden state was pinned at `tanh`'s boundary, exactly ±1.0, regardless of
  input — a well-known vanilla-RNN pathology (the same reason LSTM/GRU exist),
  distinct from mechanism (a)'s single-step sigmoid saturation.
  `grad_clip_norm=1.0` was tested at reduced scale (3,000 steps, matching
  the methodology that validated mechanism (a)'s fix) and made **no
  difference** — hidden state still pinned at ±1.0, delta span still ≈0
  throughout. `RecurrentHedgingAgent`'s own `orthogonal_init` parameter
  (whose docstring names this exact failure mode — "can otherwise leave the
  network stuck at an input-insensitive constant output no matter how long
  it trains" — but was, like `grad_clip_norm`, never wired to
  `train_policy.py`'s CLI) was also tested, alone and combined with
  clipping: **also no improvement** in either case. Unlike mechanism (a),
  where a single fix cleanly resolved a confirmed cause, this is a
  confirmed cause (recurrent hidden-state saturation) with two plausible,
  purpose-built fixes both empirically ruled out — genuinely open, not
  merely unattempted. `Basic RNN (TimeGAN)` stays in the known-bad list.
- **A production-scale retrain of Basic RNN (TimeGAN) was started and
  killed after 50 minutes at only 14,000/25,000 steps** — far slower than
  the ~27 minutes MLP retrains take at the same step count, apparently
  because `use_bs_baseline` (used for every Basic RNN checkpoint in this
  project) computes a second forward pass through the closed-form
  Black-Scholes policy every step, compounding with the recurrent
  architecture's own higher per-step cost. Killed cleanly before any
  partial checkpoint was written (`torch.save` only happens at the very
  end), so nothing was lost, but this is a real operational constraint
  worth recording: this specific checkpoint doesn't fit into an
  hour-scale compute budget the way MLP retrains do.

**A process note on training these 7 checkpoints (plus the 8 architecture
checkpoints above) at `batch_size=1000, 25,000 steps`**: the first attempt
combined all 7 alphas into one long-running `--alpha-sweep` process, which
was killed twice with no error, traceback, or OOM evidence in either case
(`ps aux`/`vm_stat`/`memory_pressure` all checked, nothing abnormal found).
Switching to one standalone process per alpha worked far better — but
individual jobs still intermittently died, consistently around step
~1,400-1,430 regardless of which alpha was training. A retry-once policy
(retry the same job once; only escalate if the *same* job fails twice in a
row) was applied across the full 15-job queue (8 architectures × 2
generators + 7 alphas): 4 alpha checkpoints succeeded on the first attempt,
4 needed exactly one retry, and every retry succeeded — the escalation
branch was never triggered. The root cause of these kills was never
determined (no crash logs checked, no thermal/energy-state investigation)
and is recorded here as an open, unexplained anomaly, not a resolved one.

**Follow-up: not reproducible on a later attempt, cause still unknown.**
Revisited directly, hours after the original 15-job queue finished (the
original checkpoints' mtimes are 23:16-00:41; this check ran at ~07:00 the
same night), on the same machine. Three probe runs, chosen to cover both
originally-affected failure modes and both a light and a heavy
architecture, every one run well past the historical ~1,400-1,430 death
window:

| Probe | Mode | Architecture | Epochs | Result |
|---|---|---|---|---|
| 1 | single process per alpha | MLP | 5,000 | completed normally |
| 2 | single process per alpha | LSTM | 1,700 | completed normally |
| 3 | combined multi-alpha in one process (the mode with the original 2-for-2 kill rate) | MLP, 4 alphas | 1,600/alpha (6,400 total steps) | completed normally, all 4 checkpoints saved |

None of the ~13,700 total gradient steps across these three runs died.
`macOS`'s unified log was also checked directly for the exact window the
original 15-job queue ran in (still available, since it was only hours
earlier) via `log show --predicate 'eventMessage contains "jetsam" OR
... OR processImagePath contains "Python"'` — no OOM/jetsam/SIGKILL event
tied to any training process turned up, only routine unrelated noise
(Spotlight's `mdworker` cycling, `runningboardd` chatter for other apps).
Memory pressure, thermal state, and disk space were all normal both before
and after.

**This is evidence the anomaly isn't currently reproducing, not evidence
it's fixed.** The original per-job failure rate was roughly 50% (4/8 retried
alpha jobs), so three consecutive clean runs has a non-trivial chance
(~1-in-8) of happening even if the underlying cause is still present at the
same rate — a small sample, not a resolution. No root cause was found
either directly (log inspection) or indirectly (the conditions that
originally seemed most likely to trigger it, re-tried and surviving). The
practical implication: keep treating any long unattended training run here
as needing a retry-once policy by default, rather than assuming this was a
one-off environmental fluke that won't recur.

**A near-miss worth recording**: reproducing the combined-alpha-sweep mode
(probe 3 above) initially used `--alpha-sweep` with a scratch checkpoint
directory, forgetting that `train_policy.py`'s `--alpha-sweep` path ignores
`--checkpoint` entirely and always writes to the fixed
`checkpoints/hedging_agent_<architecture>_alpha<alpha>.pt` path (relative
to the process's cwd) — the same path the real, already-trained checkpoints
live at. Caught and killed before the first alpha finished (verified via
the real checkpoints' unchanged mtimes), then re-run with the process's cwd
pointed at an isolated scratch directory instead. Not a repo bug — `evaluate.py`'s
own `_load_all_policies`/`load_alpha_sweep_checkpoints` rely on this same
fixed-path convention to find checkpoints automatically — but worth noting
for the next person: `--alpha-sweep` has no output-path override, so any
throwaway/exploratory run using it needs an isolated cwd, not just a
different `--checkpoint` flag.

#### Follow-up diagnosis: mechanism (b) is a sharp cliff at TimeGAN's training-distribution boundary, not a gradual generalization failure

The spot-grid diagnostic that found "healthy delta span 0.99-1.0, moderate
logits" for LSTM/GRU (TimeGAN) evidently didn't probe far enough to find
where these checkpoints actually break. Pulling the checkpoints' own
worst-loss paths from the 500,000-path scan (seed=42) and testing with
controlled synthetic ramps, using the same training-free methodology
applied to GRU (WGAN-GP) above, finds a precise mechanism:

- **TimeGAN's own training distribution is far narrower than either
  generator used elsewhere in this project.** Sampling 20,000 fresh paths
  from the trained TimeGAN checkpoint (`checkpoints/timegan.pt`) and
  converting to the same log-moneyness scale the policy sees: log-moneyness
  spans only **-0.173 to +0.133** — versus WGAN-GP's roughly -1.43 to +0.27
  (the range measured for GRU (WGAN-GP) above), and versus the actual
  500,000-path stress test's most extreme paths, which reach log-moneyness
  **+8.34** (a shared worst-case path both LSTM and GRU (TimeGAN) fail on —
  expected, not evidence of anything, since it's the same seed-42 price
  tensor for every policy) — the stress test's most extreme paths reach
  about **60x** further out than TimeGAN's own training distribution's
  positive tail ever produced.
- **On the actual worst path, LSTM's delta collapses from 0.9998 to 0.0001
  in a single step**, at a log-moneyness of just **0.071** — comfortably
  inside the training range's positive tail (max 0.133), not even past it
  yet. A finer, controlled sweep (a smooth 29-step ramp from log-moneyness 0
  to a target, isolating the effect of level from the noisy actual path)
  pins down the real threshold: delta stays correctly high (>0.98) through
  log-moneyness ≈0.088-0.093, then falls to <0.02 by ≈0.098-0.107 over a
  handful of steps (steps 18-22 of the ramp, so step-count and level are
  confounded here — this doesn't isolate a pure step-count-driven
  transition, only the level range). **The stronger, more precise finding
  than "at the boundary": these checkpoints fail *before* reaching the
  edge of their own training distribution**, not right at it — the cliff
  (≈0.09-0.11) sits measurably inside the measured positive-tail boundary
  (0.133), and the worst path's actual collapse (0.071) sits further
  inside still. GRU (TimeGAN) shows the same cliff at a nearly identical
  threshold (≈0.10-0.11).
  **Both architectures failing at essentially the same threshold, despite
  different cell types, is the discriminating evidence that this is a
  training-distribution problem, not an architecture-specific one** — the
  opposite of GRU (WGAN-GP)'s mechanism (c) above, where LSTM (WGAN-GP)
  didn't share the failure under an identical shock from the same
  generator.
- **Control: MLP (TimeGAN) shows no cliff at all, out to log-moneyness
  ±8.0.** `hedging_agent_timegan.pt` was swept across the same log-moneyness
  range at several `delta_prev` values (its state includes `delta_prev`,
  `T-t`, and `implied_vol` explicitly, unlike the recurrent policies). Delta
  rises smoothly and monotonically from 0.43 at log-moneyness 0 to
  1.00000 by log-moneyness 0.5, and *stays* at 1.00000 all the way to 8.0 —
  correctly saturated at a deep-ITM call's natural bound, not collapsing.
  The selloff direction is equally well-behaved, falling smoothly toward 0
  as log-moneyness drops to -1.0. This is the missing control this
  diagnosis lacked before: mechanism (b) is specific to the *recurrent*
  policies, not a property of TimeGAN's training data that every
  architecture trained against it would show. A feed-forward network with
  no compounding hidden state extrapolates a learned monotonic relationship
  cleanly; the recurrent policies' hidden-state dynamics do not.
- **An "inverted delta" hypothesis was considered and ruled out.** An
  earlier, cruder probe (jumping straight to a target level rather than
  ramping smoothly) suggested delta might be tracking the *wrong sign* —
  collapsing toward 0 for rallies but climbing toward 1 for selloffs, the
  opposite of a call's correct ∂delta/∂S > 0 relationship. That probe
  confounded path *shape* with path *endpoint* (a jump has a different
  per-step trajectory than a ramp to the same target), so it couldn't
  actually support that claim. Redone as a proper controlled comparison —
  identical 29-step ramp construction, sign of the target flipped, full
  per-step delta path compared rather than just the final value — the
  rally direction still shows the sharp cliff described above, but the
  selloff direction shows *no* comparable collapse: delta stays at or above
  0.98 throughout a ramp to log-moneyness -0.13, and stays near 1.0 all the
  way out to -0.4 in the coarser sweep that first surfaced this asymmetry.
  This isn't evidence of a learned sign inversion — it's an asymmetric
  failure specific to the rally direction, with no explanation attempted
  here for *why* the selloff direction doesn't show it (the CVaR/fee
  structure may make "stay hedged during a selloff" a cheaper mistake than
  "stop hedging during a rally," but that's a hypothesis, not tested).
- **This is not mechanism (a)'s dead-gradient saturation, reconciling
  rather than contradicting the original "moderate logits" finding.** The
  raw pre-sigmoid logits at the collapsed steps sit around -16 to -21, not
  mechanism (a)'s ≈-250 — `sigmoid(-21) ≈ 7.6×10⁻¹⁰` is a very small but
  numerically real float32 value, not an underflowed dead zone with exactly
  zero gradient. In practice this is still effectively a dead end for
  learning, just not for the same underflow reason as mechanism (a):
  `sigmoid'(-21) ≈ sigmoid(-21) ≈ 7.6×10⁻¹⁰` is representable but far too
  small for a gradient step to move the weights in any meaningful number of
  steps, so training stalls in practice even though it isn't mathematically
  zero. The original diagnostic's "moderate logits" finding was correct for
  whatever range it actually tested; it simply didn't extend far enough to
  reach this specific cliff, which — per the point above — sits inside
  TimeGAN's own training boundary, not past it.
- **Validated against real stress-test paths, not just the smooth-ramp
  construction used above.** A smooth 29-step ramp isolates level from
  path shape, but it's a synthetic construction — worth checking it isn't
  itself an artifact before trusting it. Sampling 2,000 real paths from the
  500,000-path stress scan (seed=42) that reach final log-moneyness > 0.5
  (well past the training boundary) and reading each policy's actual final
  delta: **LSTM collapses (delta < 0.1) on 97.0% of them** — the ramp's
  "both architectures fail past the boundary" claim holds essentially
  unconditionally for LSTM. **GRU collapses on only 64.6%** — real, and
  still a majority, but well short of LSTM's near-certainty. The other
  35.4% include paths GRU hedges correctly even out past log-moneyness 6-8,
  which single-path anecdotes (below) can make look like the mechanism
  doesn't apply to GRU at all; the aggregate rate says otherwise. This
  matches the asymmetry already on this document's record for GRU (WGAN-GP,
  mechanism (c) above): GRU's collapse behavior here is also less of a
  deterministic threshold and more history/path-shape-dependent than
  LSTM's.
- **Fix attempted: widening TimeGAN's own training distribution.** See the
  new subsection immediately below — a mixed result, not promoted.

#### Fix attempt: widening TimeGAN's own training distribution (`output_scale`) — mixed, not promoted

The candidate fix already on this document's list — training-time exposure
to more extreme price excursions than TimeGAN's real-data-bounded training
distribution can produce — was implemented and tried at full stress-test
scale.

- **Design.** `TimeGANPriceGenerator` gained an `output_scale` parameter
  that widens the generated distribution. The first thing tried — scaling
  the input noise `z` by up to 20x — barely moved the output distribution
  (std stayed ≈0.031-0.033), because three stacked tanh layers absorb
  large-amplitude input noise. Diagnosed via a direct sweep before writing
  any training code, then pivoted to scaling the *recovered* signal
  (post-tanh, pre-inverse-transform) instead, which does widen the output
  roughly proportionally to the scale factor. `MinMaxScaler.inverse_transform`
  has no positivity floor, so at large scale factors (empirically, ≥8.0)
  prices went negative and produced `nan` downstream from `log` of a
  negative number; fixed with an explicit `.clamp(min=1e-3)`.
- **Retrained LSTM and GRU (TimeGAN) at `output_scale=3.0`, full scale,
  single seed each**, otherwise identical hyperparameters to the checkpoints
  diagnosed above — with one caveat: the Monte Carlo premium estimated over
  the *widened* generator jumped from 0.0106 to 0.0927 (≈8.7x), so the
  augmented checkpoints were trained against a materially different wealth
  objective (`P0` in the CVaR loss), not purely wider paths on an otherwise
  unchanged objective. This confound has a known, checkable direction:
  `MarketEnvironment.simulate_with_costs` adds `premium` as a flat additive
  term to every single path's terminal wealth (`wealth = wealth + premium -
  payoff`, `market_env.py`), so the +0.0821 premium increase uniformly
  shifts every wealth-level metric (worst_loss, below_-50 count, CVaR95,
  CVaR99) in the *favorable* direction for both augmented checkpoints,
  regardless of any change in policy quality — a free +0.0821 improvement
  baked into the comparison before the policy does anything differently.
  Skewness and excess kurtosis are location-invariant and unaffected. This
  matters for reading the table below: it makes LSTM's apparent gain
  partly (not wholly — the shift is only ≈0.08, small next to the observed
  changes) attributable to the confound rather than the policy, and it means
  GRU's regression on CVaR95/CVaR99/below_-50 happened *despite* this
  favorable tailwind, so the real, confound-adjusted regression is slightly
  larger than the raw numbers show, not smaller.
- **The smooth-ramp cliff moved outward as designed, for both
  architectures.** Pre-fix, both LSTM and GRU collapse between ramp targets
  +0.10 and +0.13. Post-`output_scale=3.0`, LSTM's cliff moves to between
  +0.30 and +0.40 (delta 0.997 → 0.005); GRU's moves to the same range
  (0.742 → 0.000). Roughly a 3x shift, tracking the scale factor — the
  intervention does what it was designed to do on the diagnostic that
  motivated it.
- **At full 500,000-path stress-test scale, the result is mixed and does
  not clearly justify promoting either checkpoint:**

  | Metric | LSTM pre-fix | LSTM scale=3.0 | GRU pre-fix | GRU scale=3.0 |
  |---|---|---|---|---|
  | worst_loss | -6202.41 | **-5909.96** | -6033.30 | **-5548.27** |
  | below −50 (count) | 793 | **783** | 578 | **744** (worse) |
  | CVaR95 | 10.860 | **10.344** | 8.212 | **10.061** (worse) |
  | CVaR99 | 42.133 | **40.982** | 31.974 | **38.556** (worse) |
  | skewness | -249.4 | -247.7 | -307.8 | **-259.7** |
  | excess kurtosis | 81,035 | 80,420 | 145,523 | **95,133** |

  LSTM improves modestly and consistently across every metric (~3-5%), but
  per the premium-direction note above, some of that (≈0.08 of the
  ≈0.5-1.2 point CVaR moves) is the favorable premium shift rather than the
  policy — most of the gain still looks real, but it's smaller and less
  certain than the raw table suggests, on a single seed. GRU is worse on
  `below_-50`, CVaR95, and CVaR99 — the three metrics this document has
  otherwise treated as most decision-relevant — *despite* the same
  favorable premium tailwind working against that regression, which makes
  it more likely to be a genuine policy-level effect, not less. GRU also
  improves on worst_loss, skew, and kurtosis. Net, not a fix for GRU;
  arguably a regression on the metrics that matter most for a tail-risk
  analysis.
- **The GRU regression was investigated on individual paths and remains
  only partially explained.** The augmented GRU's single worst path
  (idx=93230, wealth=-5548.3) shows a clean collapse: delta starts near
  1.0, holds through log-moneyness ≈0.96, then crashes to ≈0.03 on the very
  next step (log-moneyness ≈1.6, a ≈1.0 single-step jump rather than a
  gradual ramp) and never recovers. The *pre-fix* GRU, run on the identical
  price path, does not collapse — delta dips to 0.146 then climbs back to
  0.93-1.0 and stays there through log-moneyness 7+. Taken alone, this
  looked like evidence that widening the training distribution had traded
  a graceful-recovery behavior for a sharper, more brittle one. It doesn't
  hold up as a general explanation, though: per the real-path validation
  above, pre-fix GRU only recovers on ~35% of real paths past the training
  boundary in the first place, so idx=93230 landing in that minority is not
  unusual — it's exactly the kind of single-anecdote result the aggregate
  scan above was built to guard against. Whether the augmented GRU's
  recovery rate on the *same* real-path population is lower than pre-fix
  GRU's 64.6%, and if so why, was not measured at the time — **now measured,
  see the follow-up section immediately below: it is lower, confirming the
  regression rather than leaving it ambiguous.**
- **Not promoted.** LSTM's gain is small and single-seed; GRU regressed on
  the primary tail metrics, and the follow-up below confirms this
  regression directly on real paths rather than leaving it as an aggregate-
  metric inference. Both augmented checkpoints remain out-of-tree
  (`timegan_augment/{lstm,gru}_timegan_scale3.pt`, not copied into
  `checkpoints/`).

#### Follow-up: measuring the augmented GRU's real-path collapse rate directly — confirms the regression, and explains its mechanism

The question left open above — does the augmented GRU actually recover less
often on real paths, not just score worse on aggregate CVaR — was measured
directly, on the same 500,000-path population (seed=42) and the same
"final log-moneyness > 0.5" subsample construction used for the original
64.6%/97.0% real-path figures:

- **The augmented GRU collapses more often, not less.** Among 2,000 sampled
  real paths ending past log-moneyness 0.5: pre-fix GRU's end-of-path
  collapse rate (delta_final < 0.1) is **66.3%** (consistent with the
  64.6% reported above; the ~2-point difference is subsample-draw noise,
  not a discrepancy — this run fixed a subsample seed the original script
  didn't). The `output_scale=3.0` checkpoint's rate is **86.8%** — worse by
  20.5 points, not better. The fraction of paths that *ever* dip below
  delta 0.1 at some point rises too (80.6% → 96.9%), and the fraction that
  recover to delta > 0.5 given they dipped *falls* (6.5% → 4.1%). This
  directly settles the open question: widening the generator's training
  distribution made GRU's real-path behavior measurably worse, not just its
  aggregate CVaR table.
- **A finer ramp sweep (targets 0.05 → 8.0, the real stress test's actual
  extreme range) explains why.** Pre-fix GRU collapses sharply at its
  cliff (~0.13) but then settles into a **stable partial-hedge basin** for
  everything beyond it — delta ≈0.17-0.39 across targets 3.0-8.0, not
  correct, but not fully dead either, and this basin is what keeps its
  real-path collapse rate at "only" 66% rather than higher. The
  `output_scale=3.0` checkpoint pushes its cliff outward to ~0.30-0.40 as
  designed (delta 0.85-1.00 through target 0.20, matching the 3x-shift
  claim already on record above), but **loses the recovery basin instead
  of moving it**: from target 0.40 to 8.0 delta is pinned near 0.00 almost
  everywhere, punctuated by two narrow, unstable spikes (0.606 and 0.915 at
  targets exactly 4.0 and 5.0) with near-total collapse on either side —
  not a basin, closer to two accidental non-monotonic bumps in an otherwise
  dead extrapolated function. Widening the training distribution improved
  the *interpolated* region (correctly extended high delta out to ~0.2-0.3)
  but made the *extrapolated* region — where the real stress test's worst
  paths actually live — behave worse, not better. This reframes the
  earlier "non-monotonic post-cliff behavior... remains open and
  unexplained" note: it isn't unexplained anymore, and it isn't specific to
  this one scale factor — the mechanism (a bounded generator produces a
  policy with no principled behavior once its input leaves *any* bounded
  training range, however wide) predicts that a larger `output_scale` would
  just relocate the same problem further out, not remove it. This is why
  the fix attempted next targets the input transform directly instead of
  widening training data again.

#### Fix attempt: clipping the RNN's log-moneyness input at the training boundary — free (no retraining), works for GRU, does not work for LSTM

The recovery-basin finding above suggests a specific alternative to
widening training data: if the RNN only behaves correctly on inputs it saw
during training, and any *finite* widening just relocates the same edge
further out, then instead of trying to cover an unbounded input range with
bounded training data, clip the input itself so out-of-range prices always
present an in-range value. `RecurrentHedgingAgent.moneyness_scale`'s log-
moneyness transform already reduces price to a single scalar per step
(`math_spec.md`); clamping that scalar to `[-0.15, 0.10]` — just inside
TimeGAN's own measured training boundary (-0.173/+0.133), tightened to stay
inside the range independently confirmed clean (delta > 0.98 up to
≈0.093) — before it reaches the recurrent cell means an arbitrarily extreme
real price looks identical, at the input layer, to a value the network
already handles correctly. This needs no retraining: it's a forward-pass
wrapper around the existing production checkpoint's weights.

Tested on both `hedging_agent_lstm_timegan.pt` and
`hedging_agent_gru_timegan.pt` (production checkpoints, unmodified), single
seed, no code changes to the checkpoints themselves — the clip was applied
in a wrapper `forward()` that clamps `log(S_t/K) / moneyness_scale` before
the `self.rnn(...)` call, otherwise identical:

| Metric | GRU unclipped | GRU clipped (-0.15, 0.10) | LSTM unclipped | LSTM clipped |
|---|---|---|---|---|
| worst_loss | -6033.30 | **-4481.16** (+25.7%) | -6202.41 | -6202.31 (~0%) |
| below −50 (count) | 578 | **456** (+21.1%) | 793 | 789 (~0%) |
| CVaR95 | 8.212 | 8.201 (~flat) | 10.860 | 10.725 (~1%) |
| CVaR99 | 31.979 | **28.422** (+11.1%) | 42.139 | 41.945 (~0.5%) |
| mean wealth | -0.0299 | -0.0484 (worse) | -0.0398 | -0.0410 (worse) |
| real-path collapse rate (final lm > 0.5) | 66.3% | **43.4%** | 97.5% | 96.8% |

- **GRU improves substantially and for free**: worst_loss, below_-50, and
  CVaR99 all improve 11-26% with zero retraining cost, and the real-path
  collapse rate among extreme-moneyness paths drops from 66.3% to 43.4%
  (mean delta on those paths 0.19 → 0.44). The ramp sweep shows why: clipped
  delta now stays 0.98-1.00 through target 2.0 (fully correct, vs. collapsed
  unclipped), directly confirming the interpolation-vs-extrapolation
  reframing above. Mean wealth is slightly worse (-0.0299 → -0.0484, likely
  a transaction-cost effect from the clipped policy still adjusting its
  hedge on price moves it can no longer distinguish once clipped — not
  investigated further here), a small cost against a much larger CVaR gain.
- **LSTM is essentially unaffected** (all metrics within ~1% either way).
  Consistent with this document's existing finding that LSTM collapses
  near-deterministically (97.5%) versus GRU's more history/path-dependent
  66.3% — LSTM's failure isn't primarily an input-level extrapolation
  problem the way GRU's partially is, so clamping the input doesn't reach
  it. What *does* explain LSTM's failure is still open.
- **The clip itself isn't complete even for GRU**: the same ramp sweep that
  shows the fix working through target 2.0 also shows a *new* degradation
  further out — clipped delta falls back to ≈0.067 at targets 4.0-8.0,
  actually below the unclipped checkpoint's 0.17-0.21 there. Holding the
  input pinned at exactly the clip bound for many consecutive steps (as the
  ramp construction does) is itself a path shape the checkpoint never saw
  in training — TimeGAN's real training paths are noisy, not flat — so this
  isn't evidence the clipping idea is wrong, but it does mean the current
  clip bound trades one out-of-distribution failure mode for a narrower
  one, rather than eliminating the category. Not investigated further here.
- **Not yet tried**: training a checkpoint with the clip active from the
  start (rather than wrapping an already-trained checkpoint at inference
  time only), which would let the network's own weights adapt to the
  clipped-and-possibly-repeated input distribution instead of encountering
  it only at test time; a systematic sweep over the clip bound (only
  `(-0.15, 0.10)` was tried, chosen from the already-measured training
  boundary rather than from any search); and multi-seed validation, since
  this is a single checkpoint pair, same caveat as everywhere else in this
  document. **Not promoted to production** — this is a promising, free
  result for GRU specifically, but "promoted" in this document has meant
  copying a checkpoint into `checkpoints/`, regenerating every affected
  table, and updating `tests/test_tail_risk.py`; this finding hasn't gone
  through that protocol yet, pending a decision on whether to build the
  clip into `RecurrentHedgingAgent` properly (a CLI-exposed parameter,
  applied during training too) rather than ship it as an inference-only
  wrapper.

## TimeGAN: the paper's actual Part II generator

The WGAN-GP above is a reasonable placeholder, but Kim (2021)'s actual Part
II generator is TimeGAN (Yoon et al. 2019): a 5-network
embedder/recovery/generator/supervisor/discriminator architecture over
multi-variate (OHLCV) data, not a single-feature model. `generator/timegan.py`
implements it (GRU-based Embedder/Recovery/Generator/Supervisor, LSTM-based
per-timestep Discriminator, all bounded to a shared [-1,1] latent space via
tanh — see `math_spec.md` section 5). `hidden_dim=31`, `num_layers=3`,
training `seq_len=31` all match the paper's Table 2 exactly (updated from
this repo's earlier 24/2/30 — see attempt 3 below). `generator/train_timegan.py`
implements the paper's 3-phase training procedure (autoencoder pretraining
→ supervised pretraining → joint adversarial training). The discriminator
now uses the paper's own binary cross-entropy loss by default
(`--discriminator-loss bce`); this repo's earlier WGAN-GP deviation is
kept available behind `--discriminator-loss wgan-gp` rather than deleted
(see attempt 3). Features: Open, High, Low, Close, Volume (5, not the
paper's 6 — Adj Close is dropped since it equals Close for `^GSPC`, a pure
index; see known limitations). The moment-matching loss (`math_spec.md`
section 4.1) and diversity-matching loss (section 4.2) are applied to
TimeGAN's recovered price channel.

Trained on the same real `^GSPC` data (500+500+1500 epochs).

### Attempt 1: sigmoid, [0,1] latent space (the paper's literal convention)

Fidelity check on the extracted price channel, range across 4 seeds:

| Metric | Real | Synthetic | Verdict |
|---|---|---|---|
| Diversity ratio | -- | 30.7-33.0% | Borderline — right at the 30% mode-collapse threshold |
| Mean bias | -- | -1.5 to -1.6σ | Under the 2.0σ threshold, but a consistent, non-trivial bias |
| Skewness | ≈ -1.01 | ≈ -1.08 | Excellent — diff -0.14 to +0.07 across seeds |
| Excess kurtosis | ≈ 4.50 | ≈ 5.99 | Good — diff +0.80 to +1.59, always slightly high |

Tail-shape fidelity here was *better* than the WGAN-GP+moment-loss
generator's (every seed passed cleanly; WGAN-GP still triggers a skew
warning on some seeds). But the stress-test backtest told a different
story: retraining all four policies against this generator and rerunning
the regime-switching stress test produced dramatically *worse* results than
WGAN-GP-trained policies — MLP's CVaR₉₉ 7.26 → 26.31, GRU's 4.52 → 22.33,
both worse than even the original pre-moment-loss WGAN-GP numbers. The
likely cause: TimeGAN's synthetic price paths had only ~31% of real data's
standard deviation, because its architecture composes *four*
sigmoid-bounded transformations in series (Embedder → Generator →
Supervisor → Recovery), each squashing toward the middle of [0,1] and
compounding into a narrower overall distribution than the WGAN-GP's single
tanh-bounded-log-return parameterization. Policies trained on that narrow
distribution rarely saw large moves during training and got caught
flat-footed by the stress test.

### Attempt 2: tanh, [-1,1] latent space (widen the compressed range)

The fix: switch Embedder/Recovery/Generator/Supervisor from sigmoid to
tanh, and `data.py::MinMaxScaler` from `[0,1]` to `[-1,1]` to match —
doubling the linear span per layer (width 1 → width 2) without giving up
boundedness. Retrained from scratch, same budget. Fidelity check, 4 seeds:

| Metric | Real | Synthetic | Verdict |
|---|---|---|---|
| Diversity ratio | -- | **214-224%** | Overshot, badly — the opposite failure mode |
| Mean bias | -- | +1.4σ | Under threshold, sign flipped from attempt 1 |
| Skewness | ≈ -1.01 | -1.51 to -1.55 | Borderline (diff -0.36 to **-0.59**, fails on one seed) |
| Excess kurtosis | ≈ 4.50 | 3.74-4.50 | Good (diff -0.76 to -1.53) |

The fix worked in *direction* but overshot in *magnitude* — synthetic
diversity went from 31% (badly mode-collapsed-adjacent) to 214-224% of
real (badly over-dispersed), consistently across seeds, not noise. This is
also a real gap in `validate.py`'s fidelity checker: `DIVERSITY_WARNING_THRESHOLD`
only fires on *low* diversity (mode collapse); nothing currently flags a
diversity ratio this far *above* 100%, so this run's checker output still
prints "OK" even though the distribution is clearly miscalibrated in the
other direction.

**And yet the stress-test backtest is a genuinely different story here** —
and it changed again after the RNN/LSTM moneyness fix above, in a way that
overturns this section's original conclusion. Retraining all four policies
against the tanh-fixed TimeGAN, first attempt (before the moneyness fix):

| Strategy | Mean wealth | CVaR 95% | CVaR 99% | Skew | Excess kurtosis | Total tx. cost |
|---|---|---|---|---|---|---|
| MLP (TimeGAN) | -0.511 | 5.50 | 19.05 | -16.47 | 315.2 | 5.98 |
| Basic RNN (TimeGAN) | -0.556 | 5.77 | 17.96 | -14.44 | 246.8 | 11.50 |
| LSTM (TimeGAN) | -0.566 | 5.70 | 18.34 | -13.45 | 226.4 | 12.60 |
| GRU (TimeGAN) | **-0.697** | **1.00** | **1.00** | **+1.01** | **-0.58** | 6.00 |

At the time, GRU's result looked like the headline finding: mean/std
essentially matching Black-Scholes exactly, CVaR beating it outright, the
opposite skew/kurtosis signature from every crash-exposed policy — and the
write-up here attributed it to *GRU's gating specifically* interacting well
with TimeGAN's over-dispersed training distribution, tying it to the "GRU
genuinely conditions on market state" finding from Part I.

**That explanation is now known to be wrong.** Once the RNN/LSTM
DC-dominance bug was fixed (see the Stress-test backtest section above),
every `RecurrentHedgingAgent` needed retraining against TimeGAN too —
including GRU, since the input transform changed for all three cell types.
Retrained numbers, same tanh-fixed TimeGAN checkpoint, moneyness-fixed
policies:

| Strategy | Mean wealth | CVaR 95% | CVaR 99% | Skew | Excess kurtosis | Total tx. cost |
|---|---|---|---|---|---|---|
| MLP (TimeGAN) | -0.511 | 5.50 | 19.05 | -16.47 | 315.2 | 5.98 |
| Basic RNN (TimeGAN) | **-0.697** | **1.00** | **1.00** | **+1.01** | **-0.58** | 6.00 |
| LSTM (TimeGAN) | -0.571 | 4.91 | 15.65 | -15.00 | 262.0 | 4.46 |
| GRU (TimeGAN) | -0.507 | 6.53 | 22.56 | -13.85 | 231.9 | 14.22 |

**Basic RNN now shows the exact numbers GRU used to show** — mean wealth
-0.6968 vs. GRU's old -0.6967, CVaR₉₅/₉₉ at 1.0025/1.0029 vs. GRU's old
1.0022/1.0025, transaction cost 6.00 vs. 6.00, matching to 3-4 significant
figures. **GRU's own performance under the identical generator got *worse*
with its input scaling fixed** — CVaR₉₉ 1.00 → 22.56. LSTM improved
moderately (18.34 → 15.65) but nowhere near Basic RNN's jump; MLP is
unaffected (not a `RecurrentHedgingAgent`).

This means the original "GRU-specific gating advantage" explanation was
never real — it was an artifact of the *old, buggy* input scaling
happening to route GRU specifically into a particular low-sensitivity
policy that, by luck, generalized well against this exact stress scenario.
Once every cell type sees correctly-scaled input, that same lucky
attractor moved to a *different* architecture (Basic RNN this time), not
consistently to "whichever architecture conditions best on market state"
as the earlier write-up assumed. The genuinely interesting, still-open
question is *why* this particular attractor exists at all under TimeGAN's
badly over-dispersed (214-224% of real diversity) training data, and why
it lands on a different architecture each time the input encoding changes
— not which architecture's gating is "better." Whether it reflects
something structural about training against an over-dispersed generator,
or is closer to coincidence, is not established.

This was the headline finding of the TimeGAN work through two revisions:
**neither generator is straightforwardly "better," and claims about
*which architecture* benefits from a given generator should be treated as
provisional until checked against every subsequent bug fix** — this
project found out the hard way that an input-encoding bug can masquerade
as an architecture-level finding. Attempt 3 below adds a new data point to
the still-open question of *why* that attractor existed at all.

### Attempt 3: paper hyperparameters, the paper's own BCE loss, and an explicit diversity-matching loss

Prompted by a broader push (this session's `/goal`: implement the paper's
models faithfully and correctly) to close every known gap between this
repo and the paper rather than just the diversity-calibration one, three
changes landed together and were retrained as one combined run rather than
three separate ones (each would have invalidated the others' numbers):

- **Hyperparameters now match the paper's Table 2** exactly: `hidden_dim`
  24 → **31**, `num_layers` 2 → **3**, training `seq_len` 30 → **31**
  (generation at inference time is unaffected — `sample_noise(batch_size,
  seq_len, device)` takes `seq_len` at call time and every network is
  GRU/LSTM-based, hence length-agnostic; only the training window
  changed).
- **The discriminator now uses the paper's own loss** — binary
  cross-entropy on a per-step realism logit (`BCEWithLogitsLoss`), not
  this repo's WGAN-GP deviation — as the default
  (`--discriminator-loss bce`), with `wgan-gp` kept available behind a
  flag rather than deleted (`math_spec.md` section 5).
- **An explicit diversity-matching loss** (`math_spec.md` section 4.2)
  targets the synthetic/real terminal-return std *ratio* directly (target
  1.0), rather than the previous approach of picking a bounded latent
  activation and re-measuring after the fact.

Same training budget (500+500+1500 epochs), same real `^GSPC` data.
Fidelity check (single seed — the historical 4-seed ranges above predate
this attempt and weren't rerun at this scale):

| Metric | Real | Synthetic | Verdict |
|---|---|---|---|
| Diversity ratio | -- | **130.2%** | Still overshooting, but much closer to 100% than attempt 2's 214-224% |
| Mean bias | -- | +0.4σ | Well under threshold, smallest bias of the three attempts |
| Skewness | -1.01 | -1.46 | Diff -0.45, just inside the 0.5 threshold |
| Excess kurtosis | 4.14 | 2.88 | Diff -1.27, comfortably under threshold |

**Verified via ablation, not assumed**: since hyperparameters, discriminator
loss, and the diversity loss all changed together, the 214-224% → 130.2%
improvement could have come from any of the three. A same-budget run with
identical hyperparameters and BCE loss but `--disable-diversity-loss` gives
diversity **218.1%** — essentially unchanged from attempt 2's 214-224%,
confirming the diversity-matching loss specifically (not the BCE switch or
the larger network) is what drove the improvement, not a confound. It
still didn't land exactly on 100%, and no claim is made here about why it
stops short (untried: a higher `--lambda-diversity`, more phase-3 epochs).
One side note from the ablation, not chased further: without the diversity
loss, the discriminator loss collapsed to exactly 0.0000 and the
generator's adversarial loss climbed to 17.6 by epoch 1500 (vs. 0.02-0.2
and ~7-8 respectively with it) — a much more severe D-dominated imbalance,
raising the possibility that the diversity loss has a secondary stabilizing
effect on the adversarial dynamics themselves, not just the final
diversity number. Speculative, not established.

Retraining all four policies against this generator and rerunning the
regime-switching stress test:

| Strategy | Mean wealth | CVaR 95% | CVaR 99% | Skew | Excess kurtosis | Total tx. cost |
|---|---|---|---|---|---|---|
| Black-Scholes | -0.032 | 1.15 | 1.68 | -1.97 | 5.3 | 12.90 |
| MLP (TimeGAN) | 0.176 | 7.07 | 25.00 | -15.11 | 265.4 | 2.64 |
| Basic RNN (TimeGAN) | 0.093 | 4.24 | 15.00 | -15.00 | 262.6 | 6.24 |
| LSTM (TimeGAN) | 0.170 | 6.99 | 24.73 | -14.38 | 249.3 | 20.12 |
| GRU (TimeGAN) | 0.179 | 6.91 | 24.66 | -14.58 | 254.6 | 22.91 |

**The attractor weakened but did not disappear.** In attempt 2, one
architecture landed *at* Black-Scholes (CVaR₉₉ ≈1.0, essentially matching
the closed-form baseline). Here, no architecture gets remotely close to
Black-Scholes' 1.68 — but Basic RNN (CVaR₉₉ 15.00) is still clearly
separated from the other three (24.7-25.0), by the same signature it
showed in attempt 2: lowest transaction cost (6.24 vs. 20-23), lowest std
(1.98 vs. ~3.3). Calling this "no outlier" would overstate what changed;
the honest read is that attempt 3's less extreme diversity overshoot
(130.2% vs. 214-224%) produced a *smaller* version of the same
architecture-specific effect, not its absence. This is still a genuinely
informative data point on the open question from attempt 2 — *is the
attractor's strength tied to how extreme the diversity overshoot is?* —
and it's directionally consistent with that (weaker overshoot, weaker
effect), but it doesn't establish causation: three things changed at once
(hyperparameters, discriminator loss, diversity loss), and no run isolates
diversity as the one that matters for *this* effect specifically.

**TimeGAN-trained policies now uniformly underperform WGAN-GP-trained ones
on this stress test** (compare CVaR₉₉ 15-25 here against 3.9-6.6 in the
main stress-test table above) — a less confounded comparison than attempts
1-2 produced, since no architecture here comes close to matching
Black-Scholes the way one did each time before. Whether this reflects
TimeGAN genuinely being a worse fit for this stress scenario, or the
remaining 130.2% diversity overshoot still meaningfully mismatching the
training and test distributions, is not established — the same
honest-uncertainty posture as the rest of this section.

### Attempt 4: paper scale (batch=178, 10,000 iterations) and the paper's own temporal train/test split

Same architecture and losses as attempt 3, scaled to the paper's exact
Table 2 training budget (`--batch-size 178`, phases 2000/2000/6000 epochs
≈ 10,000 total iterations) and — for the first time — trained and
fidelity-checked across the paper's own temporal split (train
1/3/1950-1/25/2010, test 1/3/1950-1/25/2021; the fidelity check now runs
against the *held-out* 2010-2021 window instead of resampling the training
period, made possible by the `HistoricalPriceLoader` cache-hit bug fix
described [above](#part-ii-implement-the-papers-temporal-traintest-split)).
Fidelity check, held-out data:

| Metric | Real (held-out) | Synthetic | Verdict |
|---|---|---|---|
| Diversity ratio | -- | **87.3%** | First *undershoot* across all four attempts (31% → 214-224% → 130.2% → 87.3%) — landed on the other side of 100% instead of asymptotically approaching it |
| Mean bias | -- | -0.43σ | Comfortably under threshold |
| Skewness | -0.977 | -0.984 | Diff **-0.007** — the tightest skew match of any attempt |
| Excess kurtosis | 4.08 | 3.09 | Diff -0.99, comfortably under threshold |

The full-text verdict from `validate.py`: *"OK: diversity is 87.3% of
real, mean bias is -0.4 real std devs, skew diff -0.01, kurtosis diff
-0.99."* This is the best fidelity match of the four attempts on every
tracked metric — closest diversity ratio to 100%, tightest skew — though
it approached from the other side (87.3% is 12.7 points low; 130.2% was
30.2 points high) rather than converging monotonically, so "closest yet"
shouldn't be read as "still closing the same gap in the same direction."

Retraining all four policies against this generator and rerunning the
paper-scale, 500,000-path regime-switching stress test:

| Strategy | Mean wealth | CVaR 95% | CVaR 99% | Skew | Excess kurtosis | mean tx. cost |
|---|---|---|---|---|---|---|
| Black-Scholes | -0.033 | 1.20 | 1.85 | -2.24 | 7.6 | 0.0065 |
| MLP (TimeGAN) | -0.039 | 1.98 | 3.10 | -2.35 | 8.5 | 0.0113 |
| Basic RNN (TimeGAN) | -0.031 | 4.47 | **17.46** | **-250.1** | **81,339** | 0.0018 |
| LSTM (TimeGAN) | -0.040 | 10.86 | **42.13** | **-249.4** | **81,035** | 0.0093 |
| GRU (TimeGAN) | -0.030 | 8.21 | **31.97** | **-307.8** | **145,523** | 0.0111 |

**The best-fidelity generator to date produced the worst-behaved policies
to date, and the "attractor" framing from attempts 1-3 is superseded by a
larger finding.** MLP is clean (CVaR₉₉ 3.10 — not directly comparable to
attempt 3's 25.00; see the scale/convention caveat below) and the other
three architectures show the same catastrophic-tail signature
documented in [Catastrophic tail risk](#catastrophic-tail-risk-invisible-below-500000-test-paths)
above (skew/kurtosis three-to-four orders of magnitude out of line, driven
by a small fraction of paths — 238 to 793 out of 500,000 — where the
policy's hedge fails on price excursions the smaller, 2,000-path test
batches used in attempts 1-3 never sampled). **This means every prior
attempt's "attractor" investigation — which architecture beats
Black-Scholes and why — was conducted at a test scale too small to see
this.** Basic RNN's signature across attempts 2-4 has consistently been
"lowest transaction cost, most separated from the pack" (attempt 2: near
Black-Scholes; attempt 3: still separated, though not matching; attempt
4: lowest CVaR₉₉ of the three catastrophic architectures, and the fewest
catastrophic paths, 238 vs. 578-793). Read against the tail-risk finding,
the retrospective interpretation is that "beats Black-Scholes" in attempts
1-2 was plausibly never a genuinely better hedge — it was a policy that
traded less and looked good on a test set too small to catch the rare
paths where trading less costs the most. This is not proven (attempts 1-2
weren't rerun at 500,000 paths, since their checkpoints were superseded
before this scale-up), but it is the more parsimonious explanation given
everything now known, and it means the multi-attempt "attractor" narrative
in this section should be read as **history that turned out to be
investigating the wrong signal**, not a resolved architectural finding.

**Caveat on the CVaR₉₉ comparison across attempts**: attempt 3's table
above used the 2,000-path test batch and `premium=0.0` convention (predates
the P₀ extension to this stress test); attempt 4's table uses 500,000 paths
and includes P₀. The two tables are not directly comparable cell-by-cell —
MLP's swing from CVaR₉₉ 25.00 to 3.10 is not evidence of a 4-attempt
improvement in generator quality, it's largely a scale-and-convention
change. The one comparison that *is* apples-to-apples is attempt 4's TimeGAN
table against the main WGAN-GP stress-test table above, both at 500,000
paths with P₀: TimeGAN-trained policies remain uniformly worse than
WGAN-GP-trained ones on this stress test (same conclusion as attempt 3,
now on a like-for-like basis), and three of four TimeGAN architectures now
additionally carry the catastrophic-tail failure mode that only one
WGAN-GP architecture (GRU, mildly) shows.

## Known limitations

Roughly in priority order:

1. **~~No option premium (P₀)~~ — resolved.** `MarketEnvironment` supports
   a `premium` term; Part I wires in the exact closed-form value, and the
   stress test / GAN-driven training now estimate it via Monte Carlo
   (`environment/market_env.py::estimate_premium_monte_carlo`, chunked to
   stay memory-safe through both the cheap analytic regime-switching
   simulator and slower RNN-based GAN forward passes). Part I's mean
   wealth now reads ≈0 like the paper's, with CVaR matching the paper's
   own absolute figures to within 2-9% for three of four architectures
   (see [above](#terminal-wealth-and-the-p₀-premium-term)); the stress
   test's mean wealth is now ≈0 too (see
   [above](#stress-test-backtest)). No retraining was needed for existing
   checkpoints — a constant additive wealth shift doesn't change the
   CVaR-minimizing optimal policy. No longer an open limitation; kept
   first, struck through, as a record of what implementing "faithfully"
   actually required fixing.
2. **~~Part I's training budget doesn't match the paper~~ — resolved,
   with a mixed result.** The paper's "50 epochs" is over a fixed
   500,000-scenario dataset at batch_size=1000, i.e. 25,000 gradient
   steps in this codebase's per-step convention — now the default. This
   is a genuine unit reconciliation, not just a bigger number: an earlier
   push to 500 steps (thought at the time to be "10x the paper's 50") was
   actually only ~2% of the paper's real budget. At the corrected scale,
   Black-Scholes matches the paper's absolute CVaR to 2-3% at every α
   (expected, it's analytic); LSTM/GRU are mixed (0.4% at α=0.75, up to
   34% at α=0.5); Basic RNN's mismatch got *worse*, not better (58-59% at
   α=0.5/0.99, up from 32% at the old, far smaller scale) — see
   [above](#part-i-frictionless-replication) for the full table. Scaling
   up did not uniformly close the remaining gaps, which is itself an
   informative (if less tidy) result.
3. **Generator tail-risk fidelity — improved, not perfect.** The
   moment-matching loss fixed the sign and rough magnitude of both skew and
   kurtosis, and this measurably improved MLP/GRU's stress-test tail risk.
   But synthetic skew now overshoots real's (-1.65 vs. -0.91) and kurtosis
   still runs a bit low (~3.1 vs. ~4.0-5.5, itself a noisy target — see
   above). A learned, per-batch-adaptive weighting (vs. the current fixed
   `--lambda-moment`) could plausibly tighten this further.
4. **~~Basic RNN's stress-test performance is improved but not fully closed
   to LSTM/GRU's level~~ — resolved (deferred experiment run, negative
   result; and the "gap" itself turned out to be stale).** The DC-dominance
   input fix (Part I) closed the gap completely in Part I's frictionless
   setting and for LSTM in the harder WGAN-GP stress-test setting (CVaR₉₉
   18.37 → 5.02). Basic RNN was initially believed to have a separate,
   deterministic architectural limitation in the stress-test setting — that
   diagnosis was wrong: it's actually extreme seed-sensitivity (bimodal
   outcomes across random seeds, confirmed via an 8-seed sweep), masked
   because every experiment in this project used the same default seed. A
   CVaR control-variate baseline (`PolicyTrainer`'s `use_bs_baseline`,
   math_spec.md section 6) substantially reduces this variance (cross-seed
   CVaR₉₉ std 6.89 → 1.20) and turns the canonical seed-0 checkpoint's
   CVaR₉₉ from 19.26 to 7.27 — a real improvement at the time, before this
   project's paper-scale rescaling. **The "still short of LSTM/GRU's ~5.0"
   framing turned out to be stale**: that 8-seed sweep (and its 7.27/6.61
   figures) predates the paper-scale retrain, and the checkpoint currently
   on disk (paper-scale, `epochs: 25000`) was never re-measured against it
   — directly re-measuring now gives CVaR₉₉ 2.575, matching the current
   main table's 2.58, not 7.27; see [the
   correction](#fixing-the-seed-sensitivity-a-cvar-control-variate-baseline)
   for the full dating. Against the *current, verified* numbers, Basic RNN
   (2.58) already beats both LSTM (3.49) and GRU (3.81) — there was no gap
   left to close.
   **The deferred `orthogonal_init` experiment was run anyway** (8 seeds,
   full 25,000-step scale) once the compute budget allowed, and gives a
   clean negative result: every one of the 8 new seeds does worse than the
   current incumbent (CVaR₉₉ mean 3.62, best seed 2.73, vs. the incumbent's
   2.58), and seed 0 newly shows 2/500,000 catastrophic paths where the
   incumbent shows 0. No checkpoint was promoted. See [the full
   writeup](#follow-up-use_bs_baseline--orthogonal_init-the-deferred-8-seed-experiment--a-negative-result).
5. **Catastrophic tail risk in several trained policies — newly discovered
   at paper scale; mechanism (a) root-caused and fixed, mechanism (b) still
   open.** Rerunning the stress test at the paper's own 500,000-path scale
   (see item 6 below) surfaced extreme, previously-invisible tail losses in
   GRU (WGAN-GP), the α=0.997 alpha-sweep checkpoint, and Basic RNN/LSTM/GRU
   under TimeGAN — full detail, checkpoint-by-checkpoint scope, and the
   Black-Scholes control that rules out a test-set artifact in
   [Catastrophic tail risk](#catastrophic-tail-risk-invisible-below-500000-test-paths).
   Two confirmed, distinct mechanisms:
   - **(a) ~~CVaR-α training's gradient touches only the worst `(1-α)`
     fraction of each batch~~ — resolved, but not the way it was first
     diagnosed.** The α=0.997 checkpoint's degenerate never-hedge policy
     turned out not to be a sparse-gradient starvation problem: it was
     `HedgingAgent`'s sigmoid output layer getting pushed into permanent
     numerical saturation (pre-activation logits ≈ −250, where
     `sigmoid'(x)` underflows to exactly `0.0` in float32) by an
     oversized, CVaR-amplified gradient step — a genuine dead end no
     amount of further training could escape. A larger batch size (the
     obvious first fix to try) made this measurably *worse*, not better —
     see [the full writeup](#mechanism-a-root-caused-and-fixed-sigmoid-output-saturation-not-sparse-gradients)
     for why. The actual fix was `PolicyTrainer`'s existing but
     never-wired-up `grad_clip_norm`, now exposed via `train_policy.py
     --grad-clip-norm`, at the paper's unmodified batch=1,000. Verified at
     the full 500,000-path scale: 0 catastrophic paths (was 814), worst
     loss -9.7 (was -6202.5), CVaR₉₅/₉₉ 2.22/3.46 — fully in line with
     every other clean checkpoint. **Extended, not just left open**: every
     remaining candidate was checked directly with the same diagnostic
     (raw pre-activation logit / hidden state across a spot grid, no
     training needed) rather than guessed at — see [Extending the
     fix](#extending-the-fix-alpha099-confirmed-same-mechanism-gruwgan-gp-and-basic-rnntimegan-confirmed-different-ones).
     α=0.99 turned out to be the same mechanism (even more severely
     saturated than α=0.997) and is now also fixed, verified clean at
     500,000 paths. GRU (WGAN-GP) and LSTM/GRU (TimeGAN) are confirmed
     **not** saturated (healthy delta span, moderate logits) — their
     failures are genuinely different mechanisms, not this one guessed
     wrong. Basic RNN (TimeGAN) *is* saturated, but through a third,
     distinct mechanism (the vanilla RNN's recurrent hidden state pinned
     at tanh's ±1.0 bound, not the sigmoid output layer) — confirmed via
     direct inspection, and confirmed **not fixed** by `grad_clip_norm`,
     `RecurrentHedgingAgent`'s own (also never-wired-up) `orthogonal_init`,
     or both together, all tested at reduced scale. Genuinely open, not
     merely unattempted.
   - **(b) TimeGAN-trained recurrent policies (not MLP) generalize badly
     to price extremes** outside their training distribution — since
     precisely characterized, not just confirmed distinct from (a) and
     Basic RNN (TimeGAN)'s hidden-state saturation. TimeGAN's own training
     distribution (bounded by real historical `^GSPC` daily moves) spans
     only log-moneyness -0.17 to +0.13 — versus the stress test's most
     extreme paths (+8.34, about 60x further out) or even WGAN-GP's own
     training range (-1.43 to +0.27). Both LSTM and GRU (TimeGAN) collapse
     from a healthy delta (>0.98) to near-zero over a handful of steps, at
     a threshold (≈0.09-0.11) that sits measurably *inside* the training
     distribution's own positive boundary (0.133), not at or past it — a
     sharp cliff, not a gradual degradation, and both architectures fail at
     essentially the same threshold despite different cell types, which is
     what marks this as training-data-driven rather than
     architecture-specific (unlike mechanism (c) below). An "inverted
     delta" hypothesis (collapsing for
     rallies, climbing for selloffs — the wrong sign) was considered and
     ruled out via a properly path-shape-controlled comparison; the failure
     is asymmetric (rally-side only, no comparable selloff-side collapse
     found), not sign-inverted. **Validated against 2,000 real stress-test
     paths, not just the synthetic ramp**: LSTM collapses on 97.0% of real
     paths past the training boundary (the ramp's claim holds essentially
     unconditionally); GRU collapses on 64.6% — real, still a majority, but
     markedly less deterministic than LSTM, consistent with the
     history/path-shape-dependence already seen in GRU's mechanism (c)
     below. See [the follow-up
     diagnosis](#follow-up-diagnosis-mechanism-b-is-a-sharp-cliff-at-timegans-training-distribution-boundary-not-a-gradual-generalization-failure).
     **Fix attempted** (`TimeGANPriceGenerator.output_scale`, widening the
     generator's own training distribution 3x) — moved the smooth-ramp
     cliff outward as designed for both architectures, but at full
     500,000-path stress-test scale gave only a small, single-seed LSTM
     improvement (~3-5% across all metrics) and a GRU **regression** on
     below_-50/CVaR95/CVaR99 despite improving worst_loss/skew/kurtosis. Not
     promoted. See [the fix-attempt
     writeup](#fix-attempt-widening-timegans-own-training-distribution-output_scale--mixed-not-promoted).
     **Follow-up: the GRU regression is confirmed directly on real paths, not
     just aggregate CVaR** (collapse rate among real extreme-moneyness paths
     66.3% pre-fix → 86.8% post-`output_scale`), and a ramp sweep shows why:
     widening training data extended pre-fix GRU's correct region but
     destroyed its accidental partial-hedge recovery basin at the far tail,
     trading a broader dead zone for a narrower one rather than shrinking it
     — see [the follow-up
     measurement](#follow-up-measuring-the-augmented-grus-real-path-collapse-rate-directly--confirms-the-regression-and-explains-its-mechanism).
     **A second, different fix attempt — clipping the RNN's log-moneyness
     input at the training boundary, no retraining required — works for
     GRU** (worst_loss/below_-50/CVaR99 improve 11-26%, real-path collapse
     rate 66.3% → 43.4%) **but not for LSTM** (~1% either way, consistent
     with LSTM's failure being closer to deterministic/history-independent
     than GRU's). Not yet promoted (single seed, clip bound not swept, and
     the clip itself still degrades at very extreme targets 4.0-8.0 where
     the checkpoint sees a path shape — many consecutive identical clipped
     readings — it never saw in training). See [the fix-attempt
     writeup](#fix-attempt-clipping-the-rnns-log-moneyness-input-at-the-training-boundary--free-no-retraining-works-for-gru-does-not-work-for-lstm).
     GRU is now a substantially narrower, better-characterized problem than
     LSTM within mechanism (b); LSTM's failure mechanism remains unexplained
     beyond "near-deterministic collapse past the training boundary."
   - **(c) GRU (WGAN-GP): a GRU-specific hidden-state recovery lag after a
     rare downward shock** — since diagnosed (not just ruled out as
     saturation) and since ~~no fix attempted~~ **substantially, though not
     completely, fixed and promoted**: every one of the pre-fix checkpoint's
     worst-loss paths was a rare early downturn (below roughly the 1st
     percentile of its own training generator's output) followed by a large
     rally, with the hedge ratio taking far longer than the paper's 30-step
     horizon to climb back up, missing most of the rally's P&L before
     catching up. Confirmed architecture-specific, not a property of the
     shock itself: `LSTM` (WGAN-GP) — same generator, same shock, confirmed
     clean at 0/500,000 catastrophic paths — recovers within a single step
     where GRU did not. `grad_clip_norm=1.0`, trained at full scale, cut the
     catastrophic-path rate from 34/500,000 to 4/500,000 (worst loss -417.5
     → -137.5) and is now the production checkpoint — genuinely improved,
     not genuinely clean. See [the full
     diagnosis](#follow-up-diagnosis-gru-wgan-gp-is-a-gru-specific-hidden-state-recovery-lag-not-saturation)
     and [the fix attempt](#fix-attempt-grad_clip_norm-substantially-improves-gru-wgan-gp-does-not-fully-close-it).
6. **~~Scale~~ — resolved.** Training and evaluation now run at the
   paper's own scale throughout: Part I's 500,000 train/test scenarios and
   25,000 gradient steps (item 2 above), TimeGAN's Table 2 batch size (178)
   and ~10,000 iterations (attempt 4 above), and every stress-test/
   alpha-sweep evaluation batch (2,000 → 500,000 paths). This is also what
   surfaced item 5 above — the smaller scale wasn't just "less precise,"
   it was blind to a real failure mode.
7. **TimeGAN's diversity is much closer to 100% but now undershoots
   instead of overshooting.** Sigmoid latents undershot real diversity
   (31%); tanh overshot it (214-224%); the diversity-matching loss at a
   smaller training budget landed at 130.2% (attempt 3); the same loss at
   the paper's full training scale (attempt 4, batch=178, ~10,000
   iterations, paper's own temporal split) landed at **87.3%** — the
   closest of all four attempts to 100%, but from the other side, so this
   is not simply "keep scaling and it converges." See the TimeGAN section
   above for the full four-attempt history. `validate.py`'s fidelity
   checker's `DIVERSITY_OVERSHOOT_WARNING_THRESHOLD` (1.7x) would have
   caught attempt 2's 214-224% without flagging either 130.2% or 87.3%.
   Attempt 4's downstream stress test showed the worst policy behavior of
   any attempt (item 5 above) despite the best fidelity numbers — the
   "which architecture beats Black-Scholes" attractor investigated across
   attempts 1-3 is now believed to have been chasing the wrong signal
   entirely (see attempt 4's writeup above for the full argument), so no
   further diversity-tuning work is recommended until item 5 is
   understood, since a "better-fidelity" generator most recently produced
   the worst downstream result yet.
8. Real-data ticker (`^GSPC`) is a pure index with no dividend/split
   adjustments (`Adj Close == Close` always) — if the paper's authors used a
   security where those differ, this isn't an exact data match.

## Ideas for future work

- ~~Extend P₀ to the stress test and GAN-driven settings~~ — **done**: both
  now estimate it via Monte Carlo
  (`environment/market_env.py::estimate_premium_monte_carlo`), chunked
  internally so a single call stays feasible whether the sampler is the
  cheap analytic regime-switching simulator or a slower RNN-based GAN
  forward pass (500k paths: ~0.3s analytic, ~10s through the WGAN-GP
  generator; an earlier un-chunked attempt at 500k through the generator
  didn't finish in 180s). 500,000 paths was chosen empirically — cross-seed
  std of the estimate is ~1% relative at that count, vs. 7-10% at 50k/100k
  (see [above](#terminal-wealth-and-the-p₀-premium-term)).
- Investigate Basic RNN's CVaR gap vs. the paper's own RNN figure at Part I
  (43-59% off across α=0.5/0.75/0.99 at the paper-matched 25,000-step
  scale, *worse* than the 32% at the earlier, far smaller 500-step scale)
  — is this the same seed-sensitivity found in the stress-test setting,
  now showing up in Part I too and possibly amplified by more training on
  the same unlucky seed, or something specific to Part I's setup? A
  multi-seed sweep at 25,000 steps (same methodology as the stress-test
  one) would tell seed-sensitivity apart from a genuine scale effect.
- Tighten the moment-matching loss further (adaptive `lambda_moment`
  schedule, or matching higher moments / a full quantile loss instead of
  just skew+kurtosis) to close the remaining tail-shape gap.
- ~~Calibrate TimeGAN's diversity properly instead of guessing a bound
  width~~ — **done, partially**: the diversity-matching loss (attempt 3,
  `math_spec.md` section 4.2) brought the ratio from 214-224% to 130.2%.
  Still not landing on 100% — try a higher `--lambda-diversity`, more
  phase-3 epochs, or investigate the discriminator-loss-trending-to-zero /
  generator-loss-climbing pattern observed late in attempt 3's training
  (a possibly-imbalanced BCE endgame that wasn't diagnosed).
- ~~Add an upper-bound check to `validate.py`'s diversity signal~~ —
  **done**: `DIVERSITY_OVERSHOOT_WARNING_THRESHOLD` (1.7x) flags severe
  over-dispersion the same way `DIVERSITY_WARNING_THRESHOLD` flags mode
  collapse. Would have caught attempt 2's 214-224% (which printed "OK" at
  the time); doesn't flag attempt 3's 130.2%, since that's real (if
  incomplete) progress, not a failure. The exact threshold (1.7x) is a
  heuristic positioned between those two data points, same spirit as the
  existing 0.3x floor — not derived from anything more principled.
- **Partially answered, one level down**: *why* did training against
  TimeGAN's over-dispersed data route exactly one architecture into a
  "beats Black-Scholes" attractor each time (GRU under the old input
  scaling, Basic RNN under the fixed scaling)? An ablation (paper
  hyperparameters + BCE loss, `--disable-diversity-loss`) confirmed the
  diversity loss specifically — not the BCE switch or the larger network —
  is what took the fidelity diversity ratio from 214-224% to 130.2% (the
  ablation alone reproduces 218.1%, matching attempt 2). What that
  ablation does *not* settle is whether the *stress-test attractor's*
  weakening tracks the diversity number specifically: the ablation checked
  fidelity only, not the downstream policy-training + stress-test
  pipeline. Running the ablation checkpoint through the same 4-policy
  retrain + stress test as attempt 3 would close this remaining gap.
- ~~Close the remaining gap between Basic RNN's `use_bs_baseline` result
  and LSTM/GRU's~~ — **done, negative result, and the gap itself was
  stale**: `orthogonal_init` combined with `use_bs_baseline`, tested at
  full 8-seed/25,000-step scale, regresses every seed (best of 8: CVaR₉₉
  2.73 vs. the incumbent's verified-current 2.58) — see [the
  writeup](#follow-up-use_bs_baseline--orthogonal_init-the-deferred-8-seed-experiment--a-negative-result).
  Separately, the "6.61 vs. LSTM/GRU's ~4.4" framing this bullet was
  chasing turned out to be stale: that number predates this project's
  paper-scale rescaling and was never re-verified against the checkpoint
  later retrained at full scale (verified 2.58, already better than LSTM's
  3.49 and GRU's 3.81) — see [the correction](#fixing-the-seed-sensitivity-a-cvar-control-variate-baseline).
  Still
  untried: averaging over multiple `use_bs_baseline`-alone seeds (no
  `orthogonal_init`) and picking the best rather than a single fixed one —
  not the same experiment as the negative result above, since every seed
  in that sweep also had `orthogonal_init` applied and none beat the
  current incumbent — or an entropy bonus per the paper's own future-work
  section (untried; this project only adapted the actor-critic-baseline
  half of the paper's two suggestions).
- Apply the same 8-seed-sweep methodology used to catch Basic RNN's
  seed-sensitivity to every other single-seed claim in this document —
  it's the third time in this project a single-seed conclusion turned out
  to be wrong (see the moneyness-fix self-correction and the TimeGAN
  GRU-attribution retraction), so other still-standing single-seed claims
  should be treated as provisional, not just this one.
- ~~Scale up: more Monte Carlo scenarios, larger networks, longer
  training, matching the paper's actual computational budget~~ — **done**:
  Part I (500k scenarios, 25k steps), TimeGAN (batch=178, ~10k iterations,
  paper's temporal split), and every stress-test/alpha-sweep evaluation
  (2,000 → 500,000 paths) all now run at paper scale. This is also what
  surfaced the tail-risk finding below — worth internalizing as the reason
  to keep chasing "scale" items in general: the smaller scale wasn't just
  imprecise, it was blind to a real failure mode.
- **Root-cause and fix the catastrophic tail risk** documented in
  [Catastrophic tail risk](#catastrophic-tail-risk-invisible-below-500000-test-paths)
  and [Known limitations](#known-limitations) item 5. Two confirmed
  mechanisms, one now fixed:
  - ~~For the α=0.997 degenerate never-hedge policy: a variance-reduction
    technique for the CVaR loss at extreme α...~~ — **done, but the
    diagnosis changed along the way**: it wasn't a sparse-gradient problem
    (importance sampling was never tried; a larger batch was tried first
    and made things *worse*), it was `HedgingAgent`'s sigmoid output
    saturating into a numerically dead zone. Fixed via `PolicyTrainer`'s
    existing `grad_clip_norm`, newly exposed as `train_policy.py
    --grad-clip-norm`, at the paper's unmodified batch=1,000 — see the
    [full writeup](#mechanism-a-root-caused-and-fixed-sigmoid-output-saturation-not-sparse-gradients).
    ~~**New, highest priority now**: apply the same fix to GRU (WGAN-GP)'s
    milder tail issue and α=0.99's thickened-tail warning sign~~ — **done,
    checked rather than assumed**: α=0.99 confirmed the same mechanism (even
    more severely saturated) and is now fixed the same way, verified clean
    at 500,000 paths. GRU (WGAN-GP) confirmed **not** the same mechanism
    (healthy delta span on the same diagnostic) — see
    [Extending the fix](#extending-the-fix-alpha099-confirmed-same-mechanism-gruwgan-gp-and-basic-rnntimegan-confirmed-different-ones).
    **GRU (WGAN-GP) is now characterized and partially fixed**: a
    GRU-specific hidden-state recovery lag after a rare (~1st-percentile)
    downward shock, confirmed not shared by LSTM under an identical shock —
    see the [full diagnosis](#follow-up-diagnosis-gru-wgan-gp-is-a-gru-specific-hidden-state-recovery-lag-not-saturation).
    ~~Candidates not yet tried: a lower peak learning rate or LR warmup for
    the recurrent weights (motivated by a weight-growth hypothesis)~~ —
    **`grad_clip_norm=1.0` tried at full scale, substantially improves but
    does not fully close it** (worst loss -417.5 → -137.5, catastrophic
    paths 34 → 4 at 500,000 paths) — see the
    [fix attempt](#fix-attempt-grad_clip_norm-substantially-improves-gru-wgan-gp-does-not-fully-close-it).
    The weight-growth hypothesis that motivated trying it turned out to be
    falsified (clipped and unclipped final weight norms are nearly
    identical), which also weakens the case for the LR-warmup variant, since
    both were motivated by the same now-falsified growth story. Not yet
    tried: training against price paths that include this kind of rare
    down-then-rally excursion directly (the WGAN-GP generator's own output
    only reaches this depth at its ~1st percentile — see the diagnosis for
    the measured figure), or investigating why clipping changes the training
    trajectory enough to fix most of the failure despite not changing final
    weight magnitudes. ~~The fix has not been promoted~~ — **promoted**:
    `checkpoints/hedging_agent_gru.pt` is now the `grad_clip_norm=1.0`
    checkpoint (pre-fix version preserved as
    `hedging_agent_gru.pt.bak-pre-recovery-lag-fix`); every affected table
    in this document was regenerated, not hand-edited, and
    `tests/test_tail_risk.py` was updated to match (see the fix-attempt
    writeup's promotion paragraph). Still not fully clean — 4/500,000
    catastrophic paths remain — so closing the residual gap stays open.
  - For TimeGAN-driven RNN/LSTM/GRU's generalization failure: training-time
    exposure to more extreme price excursions — either by widening the
    regime-switching stress scenario into the training distribution
    itself, or an adversarial/extreme-scenario data augmentation step —
    since the core issue is TimeGAN's training data (bounded by real
    `^GSPC` history) never produces the kind of extreme excursion the
    regime-switching stress test does. **Now precisely characterized**: a
    sharp cliff at log-moneyness ≈0.09-0.11, almost exactly TimeGAN's own
    measured positive-tail boundary (0.133), shared by both LSTM and GRU
    (TimeGAN) at nearly the same threshold — see [the follow-up
    diagnosis](#follow-up-diagnosis-mechanism-b-is-a-sharp-cliff-at-timegans-training-distribution-boundary-not-a-gradual-generalization-failure).
    **Fix attempted** (`TimeGANPriceGenerator.output_scale=3.0`, widening
    the generator's own recovered-signal distribution) — see the
    [fix-attempt
    writeup](#fix-attempt-widening-timegans-own-training-distribution-output_scale--mixed-not-promoted):
    moved the smooth-ramp cliff outward 3x for both architectures as
    designed, but at full stress-test scale only LSTM improved (modestly,
    single seed), while GRU regressed on below_-50/CVaR95/CVaR99. Not
    promoted. **The GRU regression was then confirmed directly on real
    paths** (collapse rate 66.3% → 86.8%) and root-caused via a ramp sweep:
    widening the generator relocates the training-distribution boundary but
    doesn't remove the underlying problem (no principled behavior once the
    input leaves *any* finite training range), so it traded pre-fix GRU's
    broad, shallow partial-hedge recovery basin at extreme moneyness for a
    narrower one with two unstable spikes instead of a basin — see the
    [follow-up
    measurement](#follow-up-measuring-the-augmented-grus-real-path-collapse-rate-directly--confirms-the-regression-and-explains-its-mechanism).
    **A different fix — clipping the RNN's log-moneyness input at the
    training boundary instead of widening the training data — was tried
    next and works for GRU without any retraining** (worst_loss/below_-50/
    CVaR99 improve 11-26%, real-path collapse rate 66.3% → 43.4%), **but not
    for LSTM** (~1% either way): see the [clipping fix-attempt
    writeup](#fix-attempt-clipping-the-rnns-log-moneyness-input-at-the-training-boundary--free-no-retraining-works-for-gru-does-not-work-for-lstm).
    Not yet promoted — single seed, clip bound not swept, and the clip
    itself degrades again at very extreme targets (4.0-8.0) where the
    checkpoint faces a many-consecutive-identical-readings path shape it
    never saw in training. Remaining unexplored ideas: training with the
    clip active from the start rather than only at inference time; a
    systematic clip-bound sweep; and LSTM's failure, which the clipping
    result suggests is a genuinely different (and still unexplained)
    mechanism from GRU's, not the same problem at a different severity. An
    adversarial/extreme-scenario data augmentation step targeted at GRU's
    non-monotonic post-cliff behavior specifically (rather than a uniform
    distributional widening) remains untried. **Basic RNN (TimeGAN)
    specifically is a narrower, better-characterized sub-problem**: its
    vanilla RNN's hidden state is saturated at tanh's ±1.0 boundary
    regardless of input — confirmed via direct inspection, and confirmed
    *not* fixed by `grad_clip_norm`, `orthogonal_init` (now both wired to
    the CLI), or the combination. The final checkpoint's recurrent weight
    norms are large (`weight_ih_l1` ≈ 14.6) and saturation was already
    present by epoch 100 in the reduced-scale probes — consistent with,
    but not confirmed as, runaway growth during training (the weight
    trajectory itself wasn't logged, so early-saturation-from-initialization
    hasn't been ruled out as a competing explanation). Candidates not yet
    tried: a learning-rate warmup or lower peak LR specifically for the
    recurrent weights (motivated by the growth hypothesis, if it holds up),
    or simply switching this checkpoint's cell type away from vanilla RNN
    (LSTM/GRU's gating exists precisely to avoid this failure mode, and
    LSTM/GRU under TimeGAN don't show it).
  - ~~Add a committed regression test asserting the fraction of paths with
    wealth below some threshold stays bounded for known-good checkpoints~~
    — **done**: `tests/test_tail_risk.py`, backed by
    `evaluate.py::tail_risk_summary` / `scan_checkpoint_tail_risk` (see
    [above](#catastrophic-tail-risk-invisible-below-500000-test-paths)).
    This catches *reintroduction* of the bug in currently-clean
    checkpoints; it doesn't fix the two mechanisms above, which remain
    open.
- Revisit whether the "beats Black-Scholes" attractor investigated across
  TimeGAN attempts 1-3 (`## TimeGAN` section above) was ever a real,
  distinct phenomenon, or whether it was the tail-risk finding the whole
  time, just below the old test scale's detection threshold. Attempts 1-2's
  checkpoints were superseded before the 500,000-path stress test existed
  and weren't preserved, so this can't be checked directly — but retraining
  fresh checkpoints at attempt 1/2's exact settings (sigmoid or tanh latent,
  no diversity loss, old hyperparameters) and rerunning them through the
  current 500,000-path stress test would settle it.
- Investigate the alpha-sweep's puzzling α=0.995 dip: sandwiched between
  α=0.99 (elevated CVaR₉₉, thickened tail) and α=0.997 (catastrophic,
  confirmed degenerate), α=0.995 trained cleanly (CVaR₉₉ 3.54,
  indistinguishable from α≤0.95) despite having only **5** tail
  samples/step at batch=1000 — *fewer* than α=0.99's 10, not more.
  Tail-sample count alone doesn't order these three outcomes. Whether this
  is the sparse-gradient mechanism being probabilistic rather than a hard
  threshold, or genuine seed luck (a single training run per α, same
  caveat as everywhere else in this document), is unresolved; a
  multi-seed sweep at α ∈ {0.99, 0.995, 0.997} would tell them apart.
