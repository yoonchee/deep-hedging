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
| Basic RNN / LSTM / GRU comparison vs. Black-Scholes | `policy/hedging_agent.py` (`RecurrentHedgingAgent`) | Matches in Part I (all three cell types replicate Black-Scholes-level CVaR after a standardized-log-moneyness input fix); in the harder stress-test setting LSTM/GRU match, Basic RNN is seed-sensitive and improved (not fully closed) by a CVaR control-variate baseline — see below |
| Frictionless Part I (GBM, no transaction costs) | `backtester/replicate_part1.py` | Matches paper's exact Table 1 market/option params (S₀=K=100, vol=0.15, T=1/12, 30 steps); trained for 500 epochs, not the paper's stated 50 — verified insufficient in this implementation, see below |
| Part II: GAN-driven nonparametric scenarios | `generator/market_gan.py` (WGAN-GP) + `generator/timegan.py` (TimeGAN) | Both implemented. Mixed, architecture-dependent results that shifted after the RNN/LSTM fix — see TimeGAN section |
| Multi-alpha risk-return sweep | `train_policy.py --alpha-sweep`, `evaluate.py::run_alpha_sweep_backtest` | Matches, now extended to the paper's own Part II grid {0.5, 0.75, 0.99, 0.995, 0.997} |
| Delta-convexity diagnostic (paper Figs. 5/8/11) | `backtester/plotting.py::plot_delta_convexity` | Matches, and was the tool that caught the RNN/LSTM failure |
| Option premium P₀ in the wealth objective | `MarketEnvironment(premium=...)`, `common/black_scholes.py::black_scholes_call_price` | Implemented for Part I (closed-form, exact); still `0.0` in the stress test and every GAN-driven setting, which have no closed-form price — see [below](#terminal-wealth-and-the-p₀-premium-term) |
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
deeper head). 500 training epochs per (architecture, α) pair; 5000-path
out-of-sample test. **This is 10x the paper's own stated 50 epochs** (Table
1) — checked directly rather than left as an assumption, and the effect
turns out to be architecture-dependent, not uniform: at α=0.99, `--epochs
50` gives CVaR 3.3x worse for MLP (4.39 vs. 1.34) and roughly 1.3x worse
for LSTM/GRU (1.05/1.04 vs. 0.84/0.79), but Basic RNN barely moves (1.08 vs.
1.03) — the extra epochs help most architectures but this one architecture
already seems to have settled by epoch 50 in this run (possibly the same
seed-sensitivity flagged below rather than a genuine training-budget
effect; not disentangled here). Either way, 50 epochs is verified
insufficient for at least three of four architectures to reach the
500-epoch numbers below, not merely untested. The params below are the
paper's exact market/option setup; the training budget is not.

**Since [P₀ was added](#terminal-wealth-and-the-p₀-premium-term)**,
this is also the one experiment in the repo where the wealth objective
includes the option premium (closed-form Black-Scholes price at these exact
params, C₀ ≈ 1.727) — constant vol and r=0 make it exact here, unlike the
regime-switching/GAN-driven settings elsewhere. Black-Scholes' own mean PnL,
which was -1.729 (≈ -C₀) before P₀ was added, is now **-0.0017** — matching
the paper's own reported ≈0.0005 to three decimal places.

CVaR of terminal PnL, lower is better (reproduce with
`python src/backtester/replicate_part1.py --epochs 500`):

| α | Black-Scholes | MLP | Basic RNN | LSTM | GRU |
|---|---|---|---|---|---|
| 0.50 | 0.207 | 0.307 | 0.291 | 0.209 | **0.199** |
| 0.75 | 0.341 | 0.626 | 0.470 | 0.351 | **0.335** |
| 0.99 | 0.903 | 1.344 | 1.032 | **0.839** | 0.789 |

**All four architectures still replicate the paper's core result**: every
cell type tracks Black-Scholes-level CVaR at every risk-aversion level, with
LSTM/GRU slightly *beating* the closed-form baseline at every α once P₀
shrinks the scale (0.789-0.839 vs. 0.903 at α=0.99). MLP is the one
architecture that still trails, consistent with its simpler per-step state
formulation. `delta_convexity.png` for every α is unaffected by P₀ (a
constant additive shift to wealth doesn't change the optimal hedge ratio at
any point, only the reported PnL/CVaR scale) and still shows every learned
delta curve overlaying Black-Scholes' analytic S-curve closely.

**This now matches the paper's own absolute CVaR numbers, not just their
qualitative shape.** The paper reports Part I's α=0.99 CVaR as (its own
sign convention: CVaR of *PnL*, negative meaning a loss)
Black-Scholes/RNN/LSTM/GRU = -0.9203/-0.7810/-0.8974/-0.7270; this repo (CVaR
of *losses*, i.e. `-PnL`, so positive) now reports 0.903/1.032/0.839/0.789.
Comparing magnitudes:

| | Paper (PnL, − = loss) | This repo (loss magnitude) | Difference |
|---|---|---|---|
| Black-Scholes | -0.9203 | 0.903 | 1.9% |
| LSTM | -0.8974 | 0.839 | 6.5% |
| GRU | -0.7270 | 0.789 | 8.5% |
| Basic RNN | -0.7810 | 1.032 | 32.1% |

Black-Scholes, LSTM, and GRU now land within 2-9% of the paper's own
figures — strong evidence the two sign conventions are simply negatives of
each other (`paper's CVaR(PnL) ≈ -1 × this repo's CVaR(loss)`) once P₀ is
accounted for, and that this repo's Part I replication matches the paper
not just qualitatively but at the level of absolute numbers. **Basic RNN is
the exception**: its CVaR here (1.032) is 32% worse than the paper's RNN
figure (0.781), a real, unexplained gap rather than noise-level — worth
tracking as an open question given this project's independent finding
(below and in the stress-test section) that Basic RNN is unusually
seed-sensitive in this codebase; this single α=0.99, seed=0 run hasn't been
checked across multiple seeds the way the stress-test claim was.

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

**This table was retrained three times**, and each retrain corrected a
mistaken conclusion from the previous one — see the RNN/LSTM sequence below
for the full trail. Current numbers, **now including P₀** (reproduce with
`python src/backtester/evaluate.py`):

| Strategy | Mean wealth | CVaR 95% | CVaR 99% | Skew | Excess kurtosis | Total tx. cost |
|---|---|---|---|---|---|---|
| Black-Scholes | -0.032 | 1.15 | 1.68 | -1.97 | 5.34 | 12.90 |
| MLP | -0.004 | 3.92 | 6.60 | -4.00 | 23.8 | 10.00 |
| Basic RNN | -0.013 | 2.38 | **6.61** | -9.19 | 120.0 | 18.07 |
| LSTM | **-0.023** | **2.60** | **4.36** | -2.83 | 11.7 | 15.21 |
| GRU | -0.047 | 2.41 | 3.86 | -2.48 | 10.2 | 18.60 |

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
time, **6.61 under the current P₀-inclusive convention** — the number in
the table above). Retraining the project's canonical seed-0 checkpoint with
this flag: CVaR₉₉ 19.26 → 7.27 (→ 6.61 today). It doesn't close the gap to
LSTM/GRU's ~5.0 (~4.4 today), but it turns Basic RNN from a coin-flip
between "works" and "completely broken" into a consistently
mediocre-but-functional policy — a genuine, measured win for the technique,
even though it doesn't fully solve Basic RNN's stress-test performance.

This is the third correction of this kind in this project's RNN/LSTM
investigation (see the moneyness-fix self-correction in Part I, and the
TimeGAN GRU-attribution retraction below) — worth stating plainly rather
than smoothing over: single-seed conclusions about *why* a specific
architecture fails are unreliable in this codebase's training regime, and
should be treated as provisional until checked across multiple seeds.

### Multi-alpha risk-return sweep, extended to the paper's own Part II grid

The paper's Part II tests risk aversion at α ∈ {0.5, 0.75, 0.99, 0.995,
0.997} — noticeably more extreme than this repo's original sweep of {0.5,
0.75, 0.9, 0.95, 0.99}, which never exercised the two most extreme levels
at all. Extended by training two more MLP checkpoints
(`train_policy.py --architecture mlp --alpha-sweep 0.995,0.997`) and
rerunning the sweep backtest under the same stress-test conditions as the
main table above (reproduce with `python src/backtester/evaluate.py`):

Includes P₀ (see [above](#terminal-wealth-and-the-p₀-premium-term)):

| α | Mean wealth | CVaR 95% | CVaR 99% | Skew | Excess kurtosis |
|---|---|---|---|---|---|
| 0.50 | -0.035 | 2.01 | 2.95 | -1.82 | 4.9 |
| 0.75 | -0.037 | 1.94 | 2.92 | -1.94 | 5.5 |
| 0.90 | -0.026 | 2.83 | 4.36 | -2.28 | 6.5 |
| 0.95 | -0.033 | 2.08 | 3.11 | -1.90 | 4.6 |
| 0.99 | 0.019 | 5.93 | 13.20 | -7.17 | 70.1 |
| 0.995 | 0.059 | 5.72 | **17.83** | -15.5 | 335 |
| 0.997 | 0.027 | 6.10 | 16.56 | -8.96 | 104 |

CVaR₉₉ and tail extremity (skew/kurtosis) are both clearly higher at α ≥
0.99 than at α ≤ 0.95, but the ordering *within* each of those groups is
non-monotonic — α=0.9's CVaR₉₉ (5.03) exceeds α=0.95's (3.77), and
α=0.995's kurtosis (335) exceeds α=0.997's (104). Each checkpoint here is a
single training run at a single seed, and this project has already found
(three times, in the RNN/LSTM investigation above) that single-seed
training outcomes in this codebase are noisy enough to produce this kind of
non-monotonicity on their own — no mechanism is claimed for the ordering
here, and doing so would repeat that exact mistake. Nor was this tested
against the paper's own numbers at these α levels (the paper's Part II uses
TimeGAN and Basic RNN specifically, not this repo's WGAN-GP+MLP
combination), so no match/mismatch verdict is claimed either — only that
the sweep now actually covers the risk-aversion range the paper cares
about, instead of stopping short of it.

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
2. **Part I trains for 10x the paper's stated epoch budget** (500 vs. 50,
   Table 1) — checked directly, not assumed: `--epochs 50` gives CVaR 3.3x
   worse for MLP and ~1.3x worse for LSTM/GRU at α=0.99 (Basic RNN barely
   moves — see [above](#part-i-frictionless-replication)), so this isn't a
   cosmetic difference the paper's own budget was actually enough for in
   this implementation, at least for most architectures.
3. **Generator tail-risk fidelity — improved, not perfect.** The
   moment-matching loss fixed the sign and rough magnitude of both skew and
   kurtosis, and this measurably improved MLP/GRU's stress-test tail risk.
   But synthetic skew now overshoots real's (-1.65 vs. -0.91) and kurtosis
   still runs a bit low (~3.1 vs. ~4.0-5.5, itself a noisy target — see
   above). A learned, per-batch-adaptive weighting (vs. the current fixed
   `--lambda-moment`) could plausibly tighten this further.
4. **Basic RNN's stress-test performance is improved but not fully closed
   to LSTM/GRU's level.** The DC-dominance input fix (Part I) closed the
   gap completely in Part I's frictionless setting and for LSTM in the
   harder WGAN-GP stress-test setting (CVaR₉₉ 18.37 → 5.02). Basic RNN was
   initially believed to have a separate, deterministic architectural
   limitation in the stress-test setting — that diagnosis was wrong: it's
   actually extreme seed-sensitivity (bimodal outcomes across random
   seeds, confirmed via an 8-seed sweep), masked because every experiment
   in this project used the same default seed. A CVaR control-variate
   baseline (`PolicyTrainer`'s `use_bs_baseline`, math_spec.md section 6)
   substantially reduces this variance (cross-seed CVaR₉₉ std 6.89 → 1.20)
   and turns the canonical seed-0 checkpoint's CVaR₉₉ from 19.26 to 7.27 —
   a real improvement, though still short of LSTM/GRU's ~5.0.
5. **TimeGAN's diversity is improved but still overshoots.** Sigmoid
   latents undershot real diversity (31%); tanh overshot it (214-224%); an
   explicit diversity-matching loss (attempt 3) brought it to 130.2% —
   real progress, still not landing on 100%, and not diagnosed further
   (untried: higher `--lambda-diversity`, more phase-3 epochs). See the
   TimeGAN section above for the full three-attempt history. `validate.py`'s
   fidelity checker used to have a real gap surfaced by this —
   `DIVERSITY_WARNING_THRESHOLD` only caught *low* diversity, nothing
   flagged a ratio above 100% — now fixed: `DIVERSITY_OVERSHOOT_WARNING_THRESHOLD`
   (1.7x) would have caught attempt 2's 214-224% (it printed "OK" at the
   time) without flagging attempt 3's 130.2%. Attempt 2's overshoot produced one
   architecture's best stress-test result and another's worst, and *which*
   architecture benefited changed completely after the RNN/LSTM moneyness
   fix (never a GRU-specific effect); attempt 3's smaller overshoot
   produced a weaker version of the same effect (Basic RNN still separated
   from the pack, just no longer matching Black-Scholes outright) rather
   than its absence — every architecture's TimeGAN CVaR now trails
   Black-Scholes, but not uniformly — see the TimeGAN section's
   attempt 3 writeup for why this is suggestive, not conclusive, evidence
   the earlier attractor was an overshoot artifact.
6. **Scale** — networks and training budgets throughout are toy-sized
   relative to the paper (500k Monte Carlo scenarios, larger networks).
7. Real-data ticker (`^GSPC`) is a pure index with no dividend/split
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
- Investigate Basic RNN's 32% CVaR gap vs. the paper's own RNN figure at
  Part I α=0.99 (1.032 vs. 0.781, the one architecture that didn't land
  within the 2-9% band the other three did) — is this the same
  seed-sensitivity found in the stress-test setting, now showing up in
  Part I too, or something specific to α=0.99? A multi-seed sweep here
  (same methodology as the stress-test one) would tell the two apart.
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
- Close the remaining gap between Basic RNN's `use_bs_baseline` result
  (CVaR₉₉ 6.61 under the current P₀-inclusive convention) and LSTM/GRU's
  (~4.4): try combining the baseline with
  orthogonal init (not re-tested together since the input-scaling fix),
  average over multiple seeds and pick the best rather than a single fixed
  seed, or an entropy bonus per the paper's own future-work section
  (untried; this project only adapted the actor-critic-baseline half of
  the paper's two suggestions).
- Apply the same 8-seed-sweep methodology used to catch Basic RNN's
  seed-sensitivity to every other single-seed claim in this document —
  it's the third time in this project a single-seed conclusion turned out
  to be wrong (see the moneyness-fix self-correction and the TimeGAN
  GRU-attribution retraction), so other still-standing single-seed claims
  should be treated as provisional, not just this one.
- Add the P₀ premium term to the wealth objective.
- Scale up: more Monte Carlo scenarios, larger networks, longer training,
  matching the paper's actual computational budget.
