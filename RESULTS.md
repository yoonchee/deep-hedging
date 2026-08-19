# Results and Findings

This documents what was implemented against Kim (2021), "Deep Hedging,
Generative Adversarial Networks, and Beyond," what the actual experiments
found, and — deliberately — what didn't work and why, not just what did.

## Summary — current state

This section states where things stand now, without the iteration history.
Everything below it in this document is the detailed record of how each
result was reached, including failed attempts and self-corrections — read
that if you need to trust a number, not just cite it.

Both parts of the paper are implemented and run at the paper's own scale:
Part I (frictionless GBM, 500,000 train/test scenarios, 25,000 gradient
steps) and Part II (GAN-driven scenarios with transaction costs, WGAN-GP
and TimeGAN generators, evaluated on 500,000-path stress-test batches).

**Part I: frictionless replication.** CVaR of terminal PnL, lower is
better (reproduce with `python src/backtester/replicate_part1.py`):

| α | Black-Scholes | MLP | Basic RNN | LSTM | GRU |
|---|---|---|---|---|---|
| 0.50 | 0.207 | 0.212 | 0.205 | 0.193 | 0.268 |
| 0.75 | 0.343 | 0.347 | 0.333 | 0.312 | 0.311 |
| 0.99 | 0.947 | 0.843 | 0.780 | 0.697 | 0.699 |

All four learned architectures beat Black-Scholes at every α — the
paper's core qualitative claim. Absolute CVaR matches the paper's own
figures to 2-3% for Black-Scholes (the Monte Carlo noise floor) and
0.1-2.3% for Basic RNN (`--rnn-lr 1e-3`, 4-seed validated); LSTM/GRU are
mixed, 0.4-34% depending on α.

**Stress test: regime-switching volatility + transaction costs, paper
scale (500,000 paths, reproduce with `python src/backtester/evaluate.py`):**

| Strategy | Mean wealth | CVaR 95% | CVaR 99% | Skew | Excess kurtosis |
|---|---|---|---|---|---|
| Black-Scholes | -0.033 | 1.20 | 1.85 | -2.24 | 7.6 |
| MLP | -0.037 | 2.38 | 3.69 | -2.08 | 6.9 |
| Basic RNN | -0.036 | 1.64 | 2.58 | -2.09 | 7.5 |
| LSTM | -0.042 | 2.17 | 3.49 | -2.19 | 8.5 |
| GRU | -0.041 | 2.14 | 3.81 | -24.3 | 3,078.3 |

Every architecture except GRU looks like an ordinary fat-tailed P&L
distribution. **GRU's row is the least trustworthy number in this table.**
It was measured on a single seed of a checkpoint that no longer exists, and a
later 5-seed rerun found baseline GRU (WGAN-GP)'s CVaR₉₉ spans 5.37-13.11
across seeds with no intervention at all — so this row reports one draw from
a wide distribution, not a property of the architecture. Its
`grad_clip_norm=1.0` fix (reported here as 4/500,000 catastrophic paths, down
from 34) does not survive multi-seed validation: it improves 2/5 seeds and is
inert at a third. See [the rebuild
section](#rebuilding-every-checkpoint-from-scratch-and-multi-seeding-the-fixes-that-shipped).

**TimeGAN-driven policies, paper scale:**

| Architecture | Status | Fix |
|---|---|---|
| MLP | **Not reproducible, but not broken either.** Attempt 4's generator was never preserved, so the original row cannot be re-derived — only re-anchored. Across 5 seeds against the surviving generator: **0/500,000 catastrophic paths at every seed** (worst loss anywhere -46.3), while paths below -10 span 0.012%-0.877% around the <0.1% known-good bound. The single-seed rebuild that read 0.88% sampled the worst of the five. | — (seed-dependent, no fix identified or needed) |
| Basic RNN | **Substantially fixed, promoted.** Mean CVaR₉₉ 20.65 → 3.77 and catastrophic paths 324.6 → 16.8 across 5 seeds (5/5 improved); 2/5 seeds fully clean. Not a full close — 3/5 seeds retain 15-50 catastrophic paths. | `--lr 1e-3 --moneyness-clip -0.15 0.10`. The clip was previously ruled out against the *saturated* checkpoint, where it could not have worked; `--lr 1e-3` de-saturates first. |
| LSTM | **Fixed, promoted — but the improvement is smaller than documented.** Re-anchored 5-seed paired: CVaR₉₉ 3.94 → 3.27 (4/5 seeds), mean catastrophic paths 1.8 → 0.2. The documented 42.13 → 3.24 does not reproduce: the post-fix figure does (3.27 ± 0.51), but untreated LSTM's worst seed here is 4.47 — the collapsed pre-fix policy that 42.13 describes doesn't occur on the surviving generator. One ramp seed retains a single -799 path. | `--slow-ramp-fraction 0.05` (training-time exposure to slow price ramps through the failure zone), 5-seed validated |
| GRU | **Fix retracted — it is harmful.** Across 5 paired seeds `--moneyness-clip` improves 1/5, doubles mean CVaR₉₉ (12.14 → 24.97) and nearly triples mean catastrophic paths (150 → 411.8). The untreated baseline mean is already better than the documented post-fix figure. | none — the promoted fix should not be used |

**Open items, priority order:**

1. **GRU, both generators, is dominated by seed variance rather than by any fix — now explained, though not removed.** Baseline CVaR₉₉ spans 5.37-13.11 (WGAN-GP) and 2.78-39.67 (TimeGAN) with no intervention; between-seed spread dwarfs every between-condition difference measured here. Both previously-promoted GRU fixes failed multi-seed validation — one inert, one harmful (retracted above) — and a threshold sweep confirmed gradient clipping is the wrong intervention for GRU (WGAN-GP) at any threshold tested. **The variance is one shared defect at seed-dependent severity, not seed-dependent luck**: every seed's catastrophic paths carry the same down-then-rally signature (100-160x enriched over its 0.53% population rate), the conditional failure-rate curve has the same shape for every seed and differs only in level (~5x), and the failing paths themselves barely overlap (2 shared across 5 seeds, of a 265-path union). Severity is measurable training-free in milliseconds by the recovery-lag probe, which detects the collapse mode reliably (lag > 5 ⇒ CVaR₉₉ 17-37, no overlap with the rest) while carrying no signal on architectures without the defect (+0.005 across 20 non-GRU checkpoints). It ranked seeds within every GRU arm at Spearman +0.70 to +1.00 on the arms it was measured on, but that does not hold out-of-sample — see the bound below. **Why a seed lands at a given severity is now answered, negatively**: a 3x3 initialization x data-draw factorial rules out both main effects by direct contradiction — the initialization behind the arm's worst checkpoint is clean under all three data draws, and an initialization that was clean turns severe under one — so severity is decided by the joint trajectory and no pre-training property predicts it. Multi-seed evaluation is therefore not optional for GRU. The probe screens for the collapse mode (lag > 5 ⇒ CVaR₉₉ 17-37, no overlap with the rest) but does not rank the runs that survive it. **When it is decided is now answered too**: at a run-specific step (18k-19k in one severe run, 5k-8k in another, never in two clean ones), invisible in the training loss at 1,000-step resolution, and not preceded by any usable warning — the probe trails the damage in one run and leads it by 6,000 steps in the other. Together these close off cheap fixes: nothing to tune, no step to stop at, no pre-training property to select on. Train several seeds, evaluate all at 500,000 paths, discard the broken ones. See [the forensics](#where-grus-seed-variance-comes-from), [the decomposition](#why-a-seed-lands-at-a-given-severity-neither-initialization-nor-data-draw), and [the trajectories](#when-severity-is-decided-at-a-run-specific-step-with-no-in-distribution-signal).
2. **~~The TimeGAN table rows cannot be reproduced from repo state~~ — re-anchored (they still cannot be *re-derived*).** All four rows are now measured at 5 seeds against the surviving generator, and two documented claims changed: MLP is seed-dependent around the known-good bound rather than "no longer clean" (0/500,000 catastrophic at every seed), and LSTM's `--slow-ramp-fraction` fix is worth ~17% on CVaR₉₉ here rather than the documented ~92%, because untreated LSTM does not collapse on this generator. Attempt 4's generator is still gone, so the original numbers remain permanently unverifiable. See [the re-anchoring](#re-anchoring-all-four-timegan-rows-to-the-surviving-generator).
3. **~~α=0.995 alpha-sweep checkpoint~~ — closed.** Retrained with `grad_clip_norm=1.0` and validated against the seed-1 draw that motivated it: 8,495 paths below -10 and 6 catastrophic → 0 and 0. The promoted checkpoint is now the clipped run.
4. **Basic RNN (TimeGAN) is improved but not closed** — 3/5 seeds retain 15-50 catastrophic paths, the clip bound was inherited from GRU rather than tuned, and all of it is against one generator.
5. A dedicated TimeGAN path-dynamics loss term (`--lambda-dynamics`) produced the first generator to pass all 7 fidelity checks — but a policy trained against it had dramatically *worse* tail risk, not better. Documented as an open, single-seed negative result; not pursued further.
6. Lower priority: WGAN-GP's moment-matching loss still slightly over/undershoots real skew/kurtosis; the `^GSPC` data has no dividend/split adjustments. Note the standing "several single-seed results haven't been multi-seed validated" caveat is now partly discharged — the two promoted GRU fixes were the main outstanding cases, and both failed.

## Contents

- [Summary — current state](#summary--current-state)
- [What's implemented vs. the paper](#whats-implemented-vs-the-paper)
- [Terminal wealth and the P₀ (premium) term](#terminal-wealth-and-the-p₀-premium-term)
- [Part I: frictionless replication](#part-i-frictionless-replication)
- [The GAN fidelity story](#the-gan-fidelity-story)
- [Stress-test backtest](#stress-test-backtest)
- [TimeGAN: the paper's actual Part II generator](#timegan-the-papers-actual-part-ii-generator)
- [Rebuilding every checkpoint from scratch, and multi-seeding the fixes that shipped](#rebuilding-every-checkpoint-from-scratch-and-multi-seeding-the-fixes-that-shipped)
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
| 0.50 | 0.207 | 0.212 | **0.205** | 0.193 | 0.268 |
| 0.75 | 0.343 | 0.347 | **0.333** | 0.312 | 0.311 |
| 0.99 | 0.947 | 0.843 | **0.780** | 0.697 | 0.699 |

(Basic RNN's column reflects a fix — `--rnn-lr 1e-3` instead of the shared
1e-2 default; see below the paper-comparison table for the full story.
MLP/LSTM/GRU are bit-for-bit reproductions of the previous table: each
architecture resets `torch.manual_seed(seed)` before its own training loop,
so Basic RNN's changed learning rate has no effect on the others' training
or on the shared post-loop test-price draw — confirmed directly, not just
by argument: the pre-fix `results/part1_replication/part1_summary.json`
and the post-fix regeneration report identical Black-Scholes `cvar_pnl` to
full float precision, e.g. `0.20698045194149017` at α=0.5, not merely
matching to the table's rounded 3 significant figures.)

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
| 0.50 | Basic RNN | 0.2040 | 0.205 | **0.6%** |
| 0.50 | LSTM | 0.2197 | 0.193 | 12.2% |
| 0.50 | GRU | 0.2000 | 0.268 | 34.1% |
| 0.75 | Black-Scholes | 0.3353 | 0.343 | 2.2% |
| 0.75 | Basic RNN | 0.3257 | 0.333 | 2.3% |
| 0.75 | LSTM | 0.3132 | 0.312 | **0.4%** |
| 0.75 | GRU | 0.3119 | 0.311 | **0.4%** |
| 0.99 | Black-Scholes | 0.9203 | 0.947 | 2.9% |
| 0.99 | Basic RNN | 0.7810 | 0.780 | **0.1%** |
| 0.99 | LSTM | 0.8974 | 0.697 | 22.4% |
| 0.99 | GRU | 0.7270 | 0.699 | 3.8% |

**Basic RNN went from the worst-matching architecture to the best**:
0.6%/2.3%/0.1% against the paper, tighter than LSTM's 12.2%/0.4%/22.4% and
GRU's 34.1%/0.4%/3.8%, and at α=0.5 and α=0.99 tighter even than
Black-Scholes's own 2.1%/2.9% analytic Monte-Carlo noise floor — this
turned out not to be a hard architectural limitation at all. Full story
below.

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
  500-step/seed-0 run reported previously), and 43% at α=0.75. **Followed
  up with a 4-seed sweep (seeds 0-3, all at the full 25,000-step scale,
  seed=0 rerun through the identical single-architecture code path used for
  seeds 1-3 rather than reused from the four-architecture run above, since
  the two draw different out-of-sample test sets) — confirmed as a real,
  systematic gap, not primarily the seed-sensitivity already documented
  below and in the stress-test section.** The right normalization is each
  run's own RNN/Black-Scholes CVaR *ratio*, not "% above the paper's raw
  number" — the paper's own ratio itself varies by α (0.849 at α=0.99 to
  1.006 at α=0.5), so comparing raw percentages conflates that with
  whatever this repo is doing differently. By ratio:

  | α | repo RNN/BS ratio (seeds 0,1,2,3) | mean | paper's ratio |
  |---|---|---|---|
  | 0.50 | 1.574, 1.396, 1.425, 1.572 | 1.492 | 1.006 |
  | 0.75 | 1.361, 1.173, 1.228, 1.267 | 1.257 | 0.971 |
  | 0.99 | 1.318, 1.165, 1.104, 1.103 | 1.173 | 0.849 |

  Every seed at every α sits well above the paper's ratio, and the repo's
  own ratio **decreases monotonically as α rises — true seed-by-seed, not
  just in the means** (seed 0: 1.574→1.361→1.318; seed 1: 1.396→1.173→1.165;
  seed 2: 1.425→1.228→1.104; seed 3: 1.572→1.267→1.103; four sequences, four
  monotone decreases). This is the opposite of a U-shape, and the same
  direction the paper's own ratio moves, just starting and staying higher
  throughout. (An earlier draft of this section claimed α=0.75 was the
  *smallest* gap and speculated about a tails-of-the-distribution quantile
  effect — that claim used "% above the paper's raw CVaR," which isn't
  directly comparable across α because it bakes in the paper's own
  changing ratio; restated in ratio terms, as above, there is no U-shape,
  just a monotone trend, so that speculation is retracted rather than
  repeated here.) **A free noise-floor control**: each run's own
  Black-Scholes CVaR — an analytic, non-learned policy — varies only
  0.3-0.9% across the four seeds/runs at a given α, while Basic RNN's CVaR
  varies 11.7-18.3% at the same α, **20-47x** the test-set-sampling noise
  floor (27x at α=0.5, 47x at α=0.75, 20x at α=0.99). That gap is
  training-driven, not measurement noise, the same role a zero-fee
  ablation played for the GRU (TimeGAN) promotion above. **The seed=0
  rerun also retires a code-drift concern**: it reproduced the original
  seed=0 table (lines 137-165 above) almost exactly (0.3255 vs. 0.325,
  0.4654 vs. 0.466, 1.2394 vs. 1.237 at α=0.5/0.75/0.99) despite running
  through a different code path (single-architecture call vs. the
  four-architecture sweep that produced the original table) — that
  original table needed no revision. Seed 0 does sit at or near the top of
  every α's range rather than the middle, worth flagging honestly rather
  than averaging away — but even the *lowest* seed at each α (1.396 at
  α=0.5, 1.173 at α=0.75, 1.103 at α=0.99) stays **21-39%** above that α's
  paper ratio, so no seed in this 4-seed sample gets close to matching the
  paper, which is the load-bearing claim: this rules out "bad luck on seed
  0" as the *whole* explanation, even if seed 0 isn't perfectly typical of
  the other three. **What's systematically different about Basic RNN's
  training at this scale — since found, and fixed: a learning-rate-specific
  weight-norm blowup. See the section immediately below.**

#### Basic RNN's Part I gap, root-caused and fixed: a learning-rate-specific weight blowup

Direct inspection of training dynamics (loss, gradient norm, recurrent
weight norm, and hidden-state saturation logged every 500 steps of a full
25,000-step run, α=0.99) found a sharp, clean training pathology the
CVaR-loss curve alone didn't make obvious:

- **A sudden blowup between steps 4,000 and 4,500.** `weight_hh` norm jumps
  7.1 → 38.4 in a single 500-step window; hidden-state saturation
  (`|h| > 0.999`) jumps from ~0% to **93.5%** of units in the same window;
  loss jumps from ≈0.93 to ≈4.7 (5x worse) and never returns to its
  pre-blowup level for the remaining 20,500 steps — the network spends the
  back four-fifths of training in a mostly-saturated, elevated-loss regime.
  This, not "not enough training," is why more compute (25,000 vs. the
  earlier 500-step run) made the mismatch *worse*: the extra steps are
  spent stuck past this event, not converging further.
- **`--grad-clip-norm 1.0` — tested, made it worse.** The obvious first
  hypothesis (a single oversized gradient step, the same mechanism fixed
  for GRU/MLP checkpoints elsewhere in this document) doesn't fit: clipped,
  the weight norm grows *continuously* from step 500 onward instead of
  jumping once, reaching **78.8** by step 25,000 — nearly double the
  unclipped run's 42.7 — with saturation still ~95% throughout and no
  improvement in final loss (1.20 either way). Clipping caps each step's
  size but doesn't change its *direction*; if the gradient consistently
  points toward larger weights, many small clipped steps accumulate to a
  larger total displacement than a few large unclipped ones eventually
  self-limit to.
- **`orthogonal_init=True` — also tested, converges to nearly the same
  pathological state as grad_clip_norm.** Weight norm 12.9 → 78.8 (final
  value differs from the clipped run's 78.8 by less than 0.1%), saturation
  ~93%, loss ~1.22. Two different, independently-motivated interventions
  landing on almost identical numbers is itself evidence this isn't about
  initialization or single-step gradient spikes — both interventions leave
  whatever is actually driving the growth untouched.
- **`lr=1e-3` (10x lower than the shared 1e-2 default) — fixes it
  completely.** No blowup at any point in 25,000 steps: saturation stays
  ≤0.2% throughout (vs. 93-95% for every other variant), weight norm grows
  smoothly from 6.5 to 9.0 (vs. 40-79), and final training loss is ~35%
  lower (0.777 vs. ~1.20-1.22). This points to Adam's step size at the
  shared default being simply too large for the vanilla RNN's recurrent
  weights specifically — consistent with LSTM/GRU not showing this failure
  at the same 1e-2 default (their gating provides implicit stability a
  vanilla RNN's plain recurrence doesn't have) and MLP not showing it
  either (no compounding hidden state to blow up in the first place). (The
  paper specifies Adam but not a learning rate — `1e-2` was this repo's own
  uniform choice across architectures, not a paper-mandated value, so this
  is a local hyperparameter fix, not a discrepancy with the paper's own
  setup.)
- **Verified at full scale, both on the actual CVaR-of-PnL metric and
  across seeds — not just the training-loss proxy above.** A 4-seed sweep
  (seeds 0-3, `--rnn-lr 1e-3`, otherwise identical to the earlier gap-sweep)
  closes the gap to the paper almost completely and consistently:

  | α | mean gap vs. paper, lr=1e-2 (4 seeds) | mean gap vs. paper, lr=1e-3 (4 seeds) |
  |---|---|---|
  | 0.50 | 51.2% | **1.05%** (std 0.98pp) |
  | 0.75 | 32.2% | **1.36%** (std 0.52pp) |
  | 0.99 | 41.3% | **-0.43%** (std 1.03pp) |

  Every one of the 4 seeds lands within about 2 percentage points of the
  paper at every α — a complete reversal from the old 21-59% range, and
  tighter than this document's own LSTM/GRU match quality (0.4-34%,
  unaffected by this fix since they were never the problem).
- **Promoted.** `replicate_part1.py::run_part1_replication` gained an
  `rnn_lr` parameter (default `1e-3`, applied only to `architecture="rnn"`;
  `lr` — still defaulting to `1e-2` — continues to apply to MLP/LSTM/GRU
  unchanged) and a matching `--rnn-lr` CLI flag; pass `--rnn-lr 1e-2` to
  reproduce the old, pre-fix numbers. Unlike the WGAN-GP/TimeGAN settings'
  checkpoint-based promotions elsewhere in this document, Part I has no
  checkpoint to swap — the "canonical" result *is* whatever
  `python src/backtester/replicate_part1.py` with no extra flags produces,
  so the fix is a default-parameter change, not a file swap. The headline
  CVaR table and the paper-comparison table above were regenerated from a
  single joint run of all four architectures (`--seed 0`, matching the
  original table's methodology exactly, so all four share one
  Black-Scholes baseline and one out-of-sample test-price draw) — MLP,
  LSTM, and GRU's numbers reproduce the previous table bit-for-bit (each
  architecture resets `torch.manual_seed(seed)` before its own training
  loop, so RNN's changed learning rate provably can't affect the others),
  confirming only the Basic RNN column actually needed to change.

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
| ~~**GRU (TimeGAN)**~~ **GRU (TimeGAN) — substantially improved on aggregate tail metrics, not on its single worst path** | ~~578~~ **402** / 500,000 (~~0.12%~~ 0.08%) | ~~3,608~~ **2,327** | -6033.3 → **-6199.8** (not improved — see writeup) | normal (~~0.0111~~ **0.0082**) |

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
- **Follow-up (a much later session, MPS-accelerated, against a
  from-scratch faithful TimeGAN generator): the untried "lower learning
  rate for the recurrent weights" candidate genuinely fixes the diagnosed
  saturation mechanism — but doesn't fix the actual tail risk.** Trained
  two Basic RNN (TimeGAN) policies against the same `timegan_paper2.pt`
  generator used throughout this document's later LSTM (TimeGAN)
  investigation, both with `--use-bs-baseline` (matching every Basic RNN
  checkpoint's established convention): the default `--lr 1e-2` baseline,
  and `--lr 1e-3` (10x lower, the same value that fixed an analogous
  vanilla-RNN weight-norm blowup in Part I — see
  [above](#basic-rnns-part-i-gap-root-caused-and-fixed-a-learning-rate-specific-weight-blowup);
  `train_policy.py` has no per-parameter-group `--rnn-lr` the way
  `replicate_part1.py` does, so this applies the same idea as a blanket
  `--lr` override for the whole policy). Direct hidden-state inspection
  (mirroring the saturation-percentage methodology above, now on real
  TimeGAN-generated paths rather than a spot grid):

  | Checkpoint | Fraction \|h\|>0.999 | Delta span | `weight_hh_l0` norm | `weight_hh_l1` norm |
  |---|---|---|---|---|
  | baseline (lr=1e-2) | **1.0000** | 0.173 | 19.19 | 17.91 |
  | lr=1e-3 | **0.2427** | 1.000 | 5.78 | 11.72 |

  **The saturation mechanism is decisively fixed** — 100% of hidden units
  pinned at the boundary drops to 24%, delta span goes from ≈0 (constant
  0.7055 regardless of input, matching the original spot-grid diagnosis
  almost exactly) to the full [0,1] range, and recurrent weight norms drop
  3-4x, directly confirming the weight-norm-blowup hypothesis this
  document's item 2 above already established for Basic RNN's Part I gap.
  **But the actual regime-switching stress test barely moves, and if
  anything gets slightly worse**:

  | Checkpoint | CVaR₉₅ | CVaR₉₉ | Skew | Excess kurtosis |
  |---|---|---|---|---|
  | baseline (lr=1e-2) | 4.70 | 17.73 | -75.1 | 10,027.2 |
  | lr=1e-3 | 4.77 | 17.26 | -83.6 | **11,413.1** |

  CVaR₉₉ improves by only 2.6%; kurtosis is 14% *worse*. **This means
  hidden-state saturation, despite being a real, confirmed, now-fixable
  mechanism, was not the (or not the only) thing driving Basic RNN
  (TimeGAN)'s catastrophic tail risk under the stress test.** A newly
  input-sensitive policy (delta now varies path-to-path: 0.75, 0.89, 0.30,
  0.43, 0.65 at t=0 across five different training paths, vs. the
  saturated baseline's near-identical 0.7055 everywhere) is not
  automatically a *well-calibrated* one — the most likely explanation,
  consistent with this document's mechanism (b) for LSTM/GRU, is that once
  saturation stops masking the network's actual behavior, Basic RNN
  (TimeGAN) runs into the same "generalizes badly to price extremes
  outside TimeGAN's own training distribution" problem LSTM/GRU already
  have, just previously hidden behind a simpler, more obviously-diagnosable
  failure. **Not promoted** — this is a genuine partial advance in
  understanding (the saturation mechanism specifically is now fixable, a
  real result worth recording) but not a fix for the behavior RESULTS.md's
  "Catastrophic tail risk" section actually cares about. `Basic RNN
  (TimeGAN)` stays in the known-bad list. Untried next step motivated
  directly by this finding: apply `--slow-ramp-fraction` (LSTM's own
  promoted fix, targeting exactly this "generalizes badly outside the
  training distribution" failure mode) on top of the lr=1e-3 fix, now that
  saturation is no longer masking whatever the RNN's real behavior is.
- **Follow-up, same session: stacking `--slow-ramp-fraction` on the lr=1e-3
  fix confirms the hypothesis, with a substantial improvement.** Trained
  `--lr 1e-3 --slow-ramp-fraction 0.05` (LSTM's own validated dose)
  together, same seed/generator as the two runs above:

  | Checkpoint | \|h\|>0.999 | Delta span | CVaR₉₉ | Excess kurtosis |
  |---|---|---|---|---|
  | baseline (lr=1e-2) | 1.0000 | 0.173 | 17.73 | 10,027.2 |
  | lr=1e-3 alone | 0.2427 | 1.000 | 17.26 | 11,413.1 |
  | **lr=1e-3 + slow-ramp 0.05** | 0.2426 | 1.000 | **6.34** | **9,012.2** |

  Saturation stays resolved (unsurprising — same lr), but CVaR₉₉ drops a
  further 63% versus the lr-fix-alone checkpoint (17.26 → 6.34) and 64%
  versus baseline, with kurtosis also improving past *both* (9,012 vs.
  baseline's 10,027 and lr-alone's 11,413). Std also drops sharply (3.86 →
  1.35), consistent with a genuinely more stable policy, not just a
  different bad one. **This confirms the hypothesis directly**: once
  saturation stopped masking the network's behavior, Basic RNN (TimeGAN)
  did have the same training-distribution-generalization problem LSTM
  (TimeGAN) had, and the same fix helps here too — a bigger relative win
  here than LSTM's own 23% CVaR₉₉ reduction, though on a much worse
  starting point (17.73 vs. LSTM's 4.23).

  **The caution was warranted — multi-seeding reverses this finding too.**
  Seeds 1-4 added for both conditions, same generator, read the same way:

  | Condition | Mean CVaR₉₉ (5 seeds) | Std CVaR₉₉ | Mean excess kurtosis (5 seeds) |
  |---|---|---|---|
  | baseline (lr=1e-2) | 19.20 | 2.64 | 9,926.1 |
  | lr=1e-3 + slow-ramp 0.05 | 18.47 | **7.82** | **10,671.4** |

  The seed-0 result (CVaR₉₉ 17.73 → 6.34, a 64% improvement) does **not**
  generalize: across 5 seeds, mean CVaR₉₉ improves by only 3.8% (not 63%),
  mean kurtosis is 7.5% *worse*, and only 2 of 5 seeds show any improvement
  at all. The combined fix's CVaR₉₉ standard deviation (7.82) is triple
  baseline's (2.64) — it's less predictable, not more, across seeds. This
  is the same trap LSTM's dose sweep fell into with dose 0.10 (apparent
  single-seed win, reversed at 5 seeds), now confirmed a second time on a
  different architecture. **Not promoted.** `Basic RNN (TimeGAN)` stays in
  the known-bad list. Combined with the earlier lr-alone result, this
  closes out both untried candidates this document had identified for
  Basic RNN (TimeGAN) — the mechanism (recurrent hidden-state saturation)
  is now well-understood and independently fixable, but neither of the two
  natural follow-up fixes (lr alone, lr + LSTM's own slow-ramp augmentation)
  translates into a reliable stress-test improvement. Genuinely open, with
  no further untried candidate identified in this document.
- **Superseded: one untried candidate did exist, and it works.** The
  reasoning above ruled `moneyness_clip` out for this architecture on the
  strength of a test against the *saturated* checkpoint, where it changed
  nothing to four decimal places. That is the expected outcome rather than
  evidence — a hidden state pinned at ±1.0 regardless of input cannot
  respond to its input being clipped — so the clip was never actually tested
  on this architecture. Stacked on `--lr 1e-3`, which de-saturates first, it
  improves 5/5 seeds and takes mean CVaR₉₉ from 20.65 to 3.77. See [the
  rebuild
  section](#rebuilding-every-checkpoint-from-scratch-and-multi-seeding-the-fixes-that-shipped).
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
  it. **What does explain LSTM's failure is now precisely characterized —
  see the [follow-up
  diagnosis](#follow-up-lstm-timegans-failure-is-a-narrow-trajectory-dependent-transition-not-simple-saturation)
  below.**
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
  it only at test time — **done, see the follow-up immediately below**; a
  systematic sweep over the clip bound (still only `(-0.15, 0.10)` was
  tried, chosen from the already-measured training boundary rather than
  from any search); and multi-seed validation, since this is still a
  single checkpoint pair, same caveat as everywhere else in this document.

#### Follow-up: training with the clip active from the start closes more of the gap than the inference-only wrapper did

`RecurrentHedgingAgent` gained a `moneyness_clip: Optional[Tuple[float, float]]`
constructor parameter (default `None`, a no-op — every existing checkpoint's
behavior is reproduced exactly) that clamps the standardized log-moneyness
input before it reaches `self.rnn`, wired to `train_policy.py` as
`--moneyness-clip LO HI`. GRU (TimeGAN) was retrained from scratch with
`--moneyness-clip -0.15 0.10`, otherwise identical hyperparameters to the
production checkpoint (25,000 steps, batch=1,000, seed=0), so the network's
own weights see the clipped distribution throughout training, not just at
test time the way the wrapper experiment above did.

- **Better than both the pre-fix checkpoint and the inference-only wrapper
  on every metric this document treats as most decision-relevant**, at
  full 500,000-path stress-test scale:

  | Metric | pre-fix | inference-only clip | trained-with-clip |
  |---|---|---|---|
  | below −50 (count) | 578 | 456 | **402** |
  | CVaR95 | 8.212 | 8.201 | **6.014** |
  | CVaR99 | 31.979 | 28.422 | **25.519** |
  | real-path collapse rate (final lm>0.5) | 66.3% | 43.4% | **31.5%** |
  | worst_loss | -6033.30 | **-4481.16** | -6199.84 |
  | mean_wealth | -0.0299 | -0.0484 | -0.0412 |

  CVaR95 improves substantially more than the inference-only wrapper managed
  (26.7% vs. essentially 0%), and the real-path collapse rate roughly halves
  again (43.4% → 31.5%). worst_loss and mean_wealth are the two exceptions,
  both explained below rather than left as an unexplained trade-off.
- **worst_loss is driven by one single path out of 500,000, not a general
  regression.** Inspecting it directly: idx=72858, wealth=-6199.8, a
  sustained one-directional rally from log-moneyness 0 to +8.34 over the
  full 29-step path. Delta tracks correctly (0.96-0.98) through the first 7
  steps, then collapses to <0.002 at step 8 (log-moneyness 1.04) and never
  recovers for the remaining 21 steps, all the way out to +8.34 — the same
  qualitative failure this whole mechanism (b) investigation is about, just
  at a somewhat higher threshold (~1.0 log-moneyness) than the pre-fix
  checkpoint's ~0.13, and without pre-fix's partial recovery. This is a
  single rare path (2/500,000 ≈ 0.0004%); it moves worst_loss and (via the
  same path) skew/kurtosis, but not the aggregate metrics above, which is
  exactly why this document has repeatedly treated below_-50/CVaR₉₅/CVaR₉₉
  as more decision-relevant than worst_loss alone for comparing checkpoints
  (see GRU (WGAN-GP)'s `grad_clip_norm` fix above for the same pattern:
  substantially improved, not fully closed).
- **The ramp sweep's earlier 4.0-8.0 degradation is gone, replaced by a
  flat, constant response** (delta ≈0.00138, identical to 5 significant
  figures, at every ramp target from 3.0 to 8.0) rather than the
  inference-only wrapper's erratic near-zero-with-occasional-spikes
  pattern. Still wrong (a deep-ITM call's correct delta is ~1.0, not
  ~0.001), but now a single stable failure mode instead of a chaotic one —
  consistent with training itself (not just inference) having now seen the
  clip boundary repeated across many consecutive steps, at least often
  enough to settle on one answer for it, even if that answer isn't the
  financially correct one.
- **Checked, not assumed: the gain isn't just "trades less."** Mean
  transaction cost drops alongside the CVaR gain (0.0111 → 0.0082), which is
  exactly the confound this document has flagged before — a policy that
  trades less can look better on CVaR while actually just paying less to
  hedge worse (the retrospective read on attempts 1-3's "beats
  Black-Scholes" TimeGAN checkpoints above, and the pattern Basic RNN's
  transaction-cost signature has shown repeatedly). Re-ran both checkpoints
  at `proportional_fee=0.0` (same seed=42 paths, same premium construction)
  to isolate this: the trained-with-clip checkpoint's advantage barely
  moves — CVaR95 8.187 → 5.993, CVaR99 31.937 → 25.480, below_-50 577 → 400,
  all within noise of the with-fee numbers above. The gain survives zeroing
  out the one channel that could have explained it away, so it's a genuine
  reduction in hedging-driven tail risk, not a cost-cutting artifact. (Mean
  wealth is still worse for the trained-with-clip checkpoint even at zero
  fees, -0.0188 vs. -0.0330 — a real, separate trade-off, not resolved by
  this check and not investigated further here.)
- **Promoted.** `checkpoints/hedging_agent_gru_timegan.pt` is now this
  `moneyness_clip=(-0.15, 0.10)` checkpoint (pre-fix version preserved as
  `hedging_agent_gru_timegan.pt.bak-pre-moneyness-clip-fix`); the
  Catastrophic tail risk table and this section's own table above were both
  regenerated from a fresh 500,000-path scan, not hand-edited.
  `tests/test_tail_risk.py` was updated to match. GRU (TimeGAN) remains on
  the "known bad" list (`below_-50_count` is 402, not 0 — this is a real
  improvement, not a fix), now with a dedicated regression test pinning the
  improved bound the same way `test_gru_checkpoint_substantially_improved_but_not_fully_clean`
  already does for GRU (WGAN-GP).
- **LSTM (TimeGAN) was not retrained with this fix.** The inference-only
  wrapper experiment already showed clipping doesn't meaningfully move
  LSTM's numbers (~1% either way, see the fix-attempt writeup above), and
  training LSTM from scratch with the clip active wouldn't be expected to
  change that conclusion — the wrapper test is exactly the ablation that
  would show *if* clipping were going to help, and it didn't. **Its failure
  mechanism is now precisely characterized, though still not fixed — see
  the follow-up immediately below.**

#### Follow-up: LSTM (TimeGAN)'s failure is a narrow, trajectory-dependent transition, not simple saturation

Clipping was tested and ruled out for LSTM above, but *why* it doesn't work
wasn't yet understood — this follow-up root-causes it via the same
training-free direct-inspection methodology used for GRU (WGAN-GP)'s
recovery-lag diagnosis and Basic RNN (TimeGAN)'s hidden-state saturation.
`nn.LSTM` doesn't expose per-step cell state or gate values through its
normal `forward()`, so a manual step-by-step unroll of the same trained
weights was used instead — `common/lstm_introspection.py::unroll_lstm_with_gates`,
committed (not left in a scratch script) since every claim in this
subsection depends on it, with `tests/test_common_lstm_introspection.py`
checking it reproduces `nn.LSTM`'s real forward pass to 1e-6 at both 1 and
2 layers, so this diagnosis stays reproducible from the repo rather than
resting on an ephemeral session artifact.

- **The earlier "moderate logits, not saturated" finding was checking the
  wrong signal.** The top recurrent layer's hidden state `h_t` (bounded
  ±1.0 by its own `tanh`) is `abs(h) > 0.999` on **only 1-2 of 64 units** at
  any point near the cliff — not the broad saturation the aggregate
  `hidden_states.min()/.max()` check (used for Basic RNN's diagnosis)
  would flag, since that check only reports whether *any* unit hits the
  boundary, not how many.
- **Correction to an earlier version of this section**: the original
  writeup here characterized the transition as "+6.3 → −12.0 over a
  0.035-wide window" by comparing the top layer's hidden vector between two
  *different* ramp constructions (`linspace(0, 0.100, 30)` vs.
  `linspace(0, 0.135, 30)`, step sizes 0.00345 vs. 0.00466) — this compares
  two trajectories with different velocities, not one trajectory crossing a
  threshold, the same step-count/level confound this document already
  names elsewhere for the ramp diagnostic. Re-run as a single continuous
  ramp (one trajectory, `linspace(0, 0.30, 30)`, printing delta at every
  step) to isolate it properly: the collapse still happens over a narrow
  band **within the one trajectory** — delta 0.96 at log-moneyness 0.052,
  falling to 0.00001 by 0.093, roughly a 0.04-wide window, matching the
  original claim's order of magnitude despite the flawed original
  methodology — but the exact endpoints shift and shouldn't be quoted to
  three significant figures the way the original version did.
- **A second correction: the "repetition ruled out" claim doesn't survive
  scrutiny either, and the real picture is more intricate than either
  original hypothesis.** The soft-clip test that was reported as "ruling
  out" exact-repetition wasn't actually testing it: at `k=8`,
  `tanh(k·(x−hi))/k` saturates so hard beyond roughly `hi+0.5` that
  consecutive inputs from real paths in that range differ by ~1e-5 to
  1e-7 — distinct in floating point, not distinguishable to the network.
  The soft-clip reproduced the repetition condition instead of removing
  it, which the writeup should have caught before calling it "verified by
  construction." **Redone properly, twice** — the first redo (ramp to 0.09
  over 10 steps vs. ramp to 0.10 over 10 steps) still confounded level with
  velocity (0.09 in 10 steps is a 0.0100 step size, 0.10 in 10 steps is
  0.0111 — both variables moved at once, the identical mismatched-trajectory
  problem being corrected above), so it was redone a second time isolating
  velocity properly: **fix the landing level at exactly 0.09, vary only the
  number of ramp steps used to reach it** (30, 20, 15, 10, 7, 5, 3, 1),
  then hold at 0.09 for 40 further steps in every case. The result is clean
  and monotone in approach speed alone:

  | ramp steps to 0.09 | step size | dip depth (min delta) | recovers after 40-step hold? |
  |---|---|---|---|
  | 30 | 0.0030 | 0.470 | yes (never really dips) |
  | 20 | 0.0045 | 0.305 | yes |
  | 15 | 0.0060 | 0.0085 | yes |
  | 10 | 0.0090 | 0.00015 | yes (by step ~22 of the hold) |
  | 7 | 0.0129 | 0.00000 | yes |
  | 5 | 0.0180 | 0.00000 | **no** |
  | 3 | 0.0300 | 0.00000 | no |
  | 1 | 0.0900 | 0.00000 | no |

  Landing level is identical (0.09) in every row; only the approach speed
  changes, and recovery flips cleanly between step sizes 0.0129 and 0.0180
  — **this is a velocity-dependent transition, not a level-dependent one**,
  confirming the direction the earlier (confounded) 0.09-vs-0.10 comparison
  pointed at but didn't actually isolate. A slow enough approach barely
  dips at all; a fast enough approach not only dips but gets stuck there
  even once the input stops moving and holds steady — a genuine, if
  narrow, velocity-triggered hysteresis in the recurrent dynamics, not
  simply a function of "how far in the money." Repetition (the same input
  value recurring) is present in every row here, including the ones that
  recover, so it isn't the trigger either — velocity is the variable that
  actually explains the split. What *is* still clear from the real-path
  evidence already on record (43.4%/96.8%/97.4% collapse rates persisting
  across every clip variant tried): real market paths that reach this
  region essentially never do so slowly enough (≤0.013 log-moneyness per
  step, sustained) to land in the recovering regime, so this finding
  explains the mechanism without offering a usable fix — clipping can't
  slow down the underlying price path's velocity, only its level.
- **The standing explanation for why clipping doesn't help, net of the
  corrections above**: the transition itself is real, narrow, and
  triggered by how fast log-moneyness moves rather than the level it
  reaches — a variable no input-clamping transform can address, since
  clipping changes what value the network sees, not how quickly the
  underlying price got there. Any clip boundary placed to preserve full
  in-distribution accuracy sits right at the edge of the transition, with
  no safe margin to retreat into if training-time noise or
  evaluation jitter nudges the boundary at all — unlike GRU's broader,
  gentler degradation and (post-fix) partial-hedge basin, which gave the
  clipping approach real room to work with.
- **Not fixed.** No further clipping variant is expected to help, per the
  above — the problem isn't the shape of the clip function, it's that the
  learned transition itself is too sharp and too close to the boundary of
  correct behavior to give any clip room to work with. Untried candidates
  motivated by this more precise characterization: a smoothness/Lipschitz
  penalty on the output layer's sensitivity to the top hidden units near
  this input range during training (directly targeting transition
  *steepness* rather than the input transform); or training-time exposure
  to paths that cross this specific boundary slowly and repeatedly (the
  regime-switching stress test's own paths, not just TimeGAN's), which
  might teach the network a gentler transition the way real data forced GRU
  to have one. Both untried here.

#### Fix attempt: slow-ramp training augmentation — implemented and tested, reduced-scale probe inconclusive (and surfaces a new instability)

A follow-up session picked up the more concrete of the two untried candidates
above: training-time exposure to slow, gradual passes through the critical
log-moneyness zone. `PolicyTrainer` (`policy/train_policy.py`) gained a
`slow_ramp_fraction` parameter (CLI: `--slow-ramp-fraction`, plus
`--slow-ramp-zone`/`--slow-ramp-step` to override the defaults) that replaces
that fraction of each training batch with synthetic price paths whose
standardized log-moneyness ramps from 0 to a random target inside
`slow_ramp_zone` (default `(0.08, 0.14)`, matching the measured transition
band above) at `slow_ramp_step` per step (default 0.0129, the largest step
size the velocity probe found the policy recovers from), then holds near that
target with small jitter for the rest of the path. Implementation:
`PolicyTrainer._inject_slow_ramp_paths` (train_policy.py), a no-op at the
default `slow_ramp_fraction=0.0` and silently skipped for non-recurrent
policies. Six unit tests (`tests/test_policy.py`) check the construction
directly — correct replacement fraction, correct shape, every path starting
at log-moneyness 0, the ramp phase actually staying within `slow_ramp_step`
of the configured velocity, and a full `train_step` smoke test — all passing,
full suite green (110 passed / 14 skipped) with no regressions elsewhere.

**Empirical validation was attempted but is inconclusive, and doesn't
actually test the diagnosed mechanism.** Reproducing the exact paper-scale
setup this session's bug was diagnosed on (TimeGAN: batch=178,
2000/2000/6000-epoch phases ≈10,000 total iterations; LSTM policy: 25,000
gradient steps, batch=1000) was judged too expensive for this session's
budget — the reduced-scale TimeGAN run alone (700 total epochs, ~1/14 of
paper scale) took ~52s, but each LSTM policy retrain at a still-reduced
15,000-step/batch-512 budget took ~13.5 minutes, and a like-for-like
paper-scale pair (baseline + augmented) was estimated at roughly an hour
combined — deferred as future work, not attempted here. What *was* run: a
reduced-scale TimeGAN (real `^GSPC` data via yfinance, paper's
hidden_dim=31/num_layers=3 architecture, but only 150/150/400-epoch phases)
plus two LSTM policies trained against it with identical seed/settings
(15,000 steps, batch=512, lr=3e-3), differing only in
`--slow-ramp-fraction 0.15` vs. the default 0.0.

The baseline from this reduced run does **not** reproduce the documented
recovers-when-slow/stuck-when-fast signature at all. Probing it with the
same velocity-isolated ramp-then-hold construction used above (landing
levels 0.09/0.11/0.13, ramp step sizes 0.003-0.09): at landing 0.09 and 0.11,
every tested velocity — slow and fast alike — ends up stuck near delta≈0
after the ramp; at landing 0.13, every tested velocity instead recovers to
delta≈1. That's a **level**-triggered collapse (a threshold somewhere
between log-moneyness 0.11 and 0.13 that flips the policy's stuck state),
not the velocity-triggered one diagnosed on the paper-scale checkpoint above
— a different bug, an artifact of this particular reduced-scale, real-data
training run rather than the mechanism this fix targets. A quick
flat-input sanity check (constant log-moneyness=0 for 15 steps — delta
should settle, not oscillate, since nothing in the input is changing) shows
why: after a 3-step transient, this baseline locks to a constant delta=1.0
and stays there, so the "stability" is real but the checkpoint itself has
converged to a different failure geometry than the one under investigation,
making it the wrong baseline to test a velocity-specific fix against.

The augmented checkpoint (same seed, same reduced scale,
`slow_ramp_fraction=0.15`) fares worse, not better, on the same
flat-input check: delta on an unchanging log-moneyness=0 input drifts
0.51 → 0.67 → 0.51 → 0.77 → 0.72 → 0.72 → 0.51 → 0.52 → 0.39 → 0.33 → 0.22 →
0.16 → 0.10 → 0.07 → 0.05 over 15 steps and never settles — genuine
instability on an input that isn't moving at all, which the baseline
doesn't show. On the velocity-ramp probe, the augmented checkpoint reaches
delta≈1.0 within one or two steps of *any* nonzero log-moneyness input,
at every tested ramp speed and every tested landing level, with an
identical first-step delta (0.506) regardless of how slow or fast the ramp
is — i.e. it has become an oversensitive near-step-function around
log-moneyness=0, not a smoother, better-behaved transition. This is the
opposite of the intended effect: rather than teaching the policy correct,
graded behavior through the critical zone, injecting slow-ramp paths at
this fraction and scale seems to have destabilized the "do nothing near
ATM" fixed point the (already-flawed) baseline at least had.

**Net assessment**: the fix is implemented, unit-tested at the construction
level, and ready to run (`--slow-ramp-fraction`), but this session's attempt
to validate it empirically used a baseline that doesn't exhibit the bug
being fixed, so nothing here confirms or rules out whether slow-ramp
augmentation helps the actual, paper-scale LSTM (TimeGAN) failure. If
anything, the augmented run's flat-input instability is a caution against
assuming the fraction/zone/step defaults used here (0.15 / (0.08, 0.14) /
0.0129) are safe to promote without a lower-fraction sweep and a stability
check on the resulting checkpoint. **Not fixed, not disproven — the
required next step is the paper-scale baseline + augmented pair this
session's budget didn't cover**, at which point the same velocity-isolated
probe (ramp to a fixed landing level at varying step sizes, checking
whether the 0.0129-recovers/0.0180-fails boundary shifts) is the right
instrument to read the result, exactly as used above.

#### Fix attempt continued: a paper-scale reproduction that actually shows the bug, and a smoothness-penalty candidate that trades the symptom for a worse disease

A follow-up session, given a much larger compute budget (12+ hours offered),
picked this up specifically to nail a *faithful* reproduction before judging
any fix, since the attempt above never established one.

**First, an operator error, not a codebase bug, cost most of the budget.**
Several attempts to build a cheap "faithful testbed" at reduced scale
(smaller TimeGAN/policy training budgets, for fast iteration) each landed in
a *different*, unrelated failure geometry — a globally-dead near-zero
policy, a sign-inverted smooth response, values wildly outside any
economically sensible range — none matching the documented
recovers-when-slow/stuck-when-fast signature. Chasing the last of these (a
paper-scale-looking checkpoint whose raw price paths swung from 8% to
1200% of strike over 30 days, yet whose own post-training fidelity check
reported "OK: diversity 84.5%") led to an hours-long root-cause investigation
— ruling out real-data corruption (the cached `^GSPC` CSV reproduces sane
0.68-1.29 window ratios across 250,000 sampled real windows, checked
directly), a windowing bug (`sample_multivariate_price_windows` reproduces
identically whether called standalone or via the exact `main()` call order),
and a scaler-serialization bug (`torch.save`/`load_state_dict` round-trips
correctly) — before finding the actual cause: **the training command never
passed `--data-source yfinance`**, so it silently used
`train_timegan.py`'s default (`--data-source synthetic`), a GBM placeholder
with `vol=0.2` per step. Worse, `--data-source synthetic` also makes the
post-training fidelity check compare the generator against *itself*
(`sample_real_test_raw = sample_real_raw`, since there's no held-out real
range to split), so it reports "OK" no matter how unrealistic the learned
distribution is — a real gap in the fidelity checker's coverage: it has no
way to flag "the data source itself was never real" the way it flags
diversity/skew/kurtosis mismatches. Every reduced-scale attempt earlier in
this section's writeup was retroactively a real (if smaller) exploration of
this same operator error's consequences, not evidence about the documented
bug specifically.

**Corrected and rerun at true paper scale** (`--data-source yfinance`,
otherwise every default: `hidden_dim=31`, `num_layers=3`, `batch_size=178`,
2000/2000/6000-epoch phases, real `^GSPC` 1950-2010 training data), the
resulting TimeGAN's fidelity check is sane and comparable to the original
attempt 4 (diversity 110.6% vs. attempt 4's 87.3%, skew diff -0.36,
kurtosis diff -1.44 — same order of magnitude, not a exact match, expected
given a fresh random seed and this project's own documented TimeGAN
calibration variance across runs). Directly sampled raw prices confirm this
is sane: 30-step cumulative log-return std 0.054, matching real market
dynamics (the broken synthetic-sourced run's equivalent was 0.94 — a
~17x difference, immediately diagnostic in hindsight). A baseline LSTM
policy trained against it at full paper scale (25,000 steps, batch=1000,
default lr, no clipping, no augmentation) converges to a sane mean wealth
(0.024, right order of magnitude vs. attempt 4's documented -0.040, unlike
the broken run's bizarre +0.90).

**Probed with the same velocity-isolated methodology used throughout this
section** (measuring this checkpoint's *own* natural standardized
log-moneyness range first — q99 here is +0.066, close to attempt 4's
documented ~0.052-0.093 transition band — then ramping to a landing level
near that boundary at varying speeds), the baseline shows the documented
signature cleanly at landing +0.066:

| ramp steps | step size | end-state delta (last-5 avg) |
|---|---|---|
| 30 | 0.0022 | 0.997 |
| 15 | 0.0044 | 0.968 |
| 10 | 0.0066 | 0.755 |
| 7 | 0.0094 | 0.663 |
| 5 | 0.0132 | 0.643 |
| 3 | 0.0220 | 0.426 |
| 1 | 0.0660 | 0.0003 |

A clean, monotone collapse from near-1 (slow approach) to near-0 (single
fast jump) — the same qualitative signature documented for the original
paper-scale checkpoint, now reproduced independently on a fresh TimeGAN and
a fresh LSTM policy, at a boundary the diagnosis's own numbers predicted.
**This is the first faithful reproduction of mechanism (b) this project has
achieved from a from-scratch training run** (every earlier probe in this
document, including the corrected ramp constructions above, worked from
the single pre-existing checkpoint the original diagnosis used).

**The smoothness-penalty candidate** (`--smoothness-penalty-weight`,
`PolicyTrainer._compute_smoothness_penalty` in `train_policy.py`,
implemented and unit-tested this session — see below) was retrained against
the identical setup (`--smoothness-penalty-weight 0.01`, otherwise identical
to the baseline). Unlike the slow-ramp augmentation above, this penalizes
`d(delta)/d(log-moneyness)` directly on each training batch's own sampled
paths via autograd (`torch.autograd.grad` with `create_graph=True`), so it
isn't subject to the synthetic-trajectory-shape confound that undermined
the augmentation approach — and it's a *global* penalty (every time step),
not tied to a hand-picked numeric zone, since this session already learned
a fixed zone doesn't transfer across differently-calibrated TimeGAN
checkpoints (see the reduced-scale attempts above, whose natural boundaries
ranged from ±0.10-0.23 to ±2.3 depending on run). The same velocity probe
on the resulting checkpoint:

| ramp steps | step size | end-state delta (last-5 avg) |
|---|---|---|
| 30 | 0.0022 | 0.214 |
| 15 | 0.0044 | 0.203 |
| 10 | 0.0066 | 0.204 |
| 7 | 0.0094 | 0.203 |
| 5 | 0.0132 | 0.202 |
| 3 | 0.0220 | 0.200 |
| 1 | 0.0660 | 0.194 |

**The velocity-triggered hysteresis is completely gone** — end-state is
flat at ≈0.20 regardless of ramp speed, here and at every other landing
level tested (both signs, ±0.046 to ±0.226). But this checkpoint's
flat-input stability check (constant log-moneyness=0 for 15 steps) also
shows something the baseline didn't: delta drifts 0.51 → 0.67 → ... → down
to 0.05 rather than settling, and at deep-ITM landing levels (+0.086, where
the baseline correctly reaches delta≈1.0 at every tested speed) the
penalized checkpoint stays flat near 0.20 too — it looks like the penalty,
at this weight, suppressed the policy's *overall* responsiveness to
log-moneyness, not just the pathological transition.

**The actual regime-switching stress test (methodology matching
`backtester/evaluate.py`, 100,000 paths, not the full paper-scale 500,000 —
a time-budget reduction, flagged explicitly) settles it, and the answer is
not what the smooth velocity probe suggested:**

| Strategy | Mean | Std | CVaR₉₅ | CVaR₉₉ | Skew | Excess kurtosis | Tx. cost |
|---|---|---|---|---|---|---|---|
| Black-Scholes | -0.031 | 0.289 | 0.91 | 1.41 | -2.00 | 7.6 | 0.0057 |
| LSTM (TimeGAN) baseline | -0.046 | 0.771 | 2.26 | 4.42 | -3.46 | 37.6 | 0.0136 |
| LSTM (TimeGAN), smoothness penalty | -0.049 | 2.039 | 4.35 | **9.91** | **-82.6** | **13,427** | 0.0177 |

The smoothness-penalized policy's CVaR₉₉ is **more than double** the
baseline's, and its excess kurtosis is **~357x worse** (37.6 → 13,427) —
this specific fix candidate doesn't merely fail to help, it makes the
policy substantially *more* dangerous on the metric CVaR training is
actually supposed to optimize. The mechanism is visible in the velocity
probe's own deep-ITM readout above: a global sensitivity penalty, at this
weight, didn't just smooth the pathological transition, it flattened the
policy's response almost everywhere, leaving it broadly under-hedged —
trading a narrow, rare failure mode (real paths crossing the critical zone
fast) for a pervasive one (real paths ending up meaningfully ITM or OTM
and the policy not moving enough to cover the difference). **A useful
methodological note for whoever tries the next candidate**: the smooth,
reassuring-looking velocity probe result here would have been actively
misleading without the stress-test cross-check — eliminating a symptom
measured one way can hide a worse regression measured another way.

**Not fixed.** Untried, better-motivated next steps given this result: a
much smaller `--smoothness-penalty-weight` (0.01 looks over-regularized;
this session didn't have budget left to sweep it — each paper-scale run
took 78 minutes for the baseline and 3h35m for the penalized variant, the
extra `autograd.grad(create_graph=True)` call roughly tripling per-step
cost); restricting the penalty to a zone near the checkpoint's own measured
q95-q99 boundary instead of applying it globally, now that a per-checkpoint
(not hardcoded) zone is understood to be necessary; or combining a small
penalty weight with `moneyness_clip` (untested combination). The
`--slow-ramp-fraction` candidate from the attempt above also still has
never been validated against a real bug-reproducing baseline — that
remains open too, now that this session has established what such a
baseline actually looks like and how to find one (`--data-source yfinance`,
verified via direct raw-price sampling, not just trusting the fidelity
checker's summary line).

#### Fix attempt, third try: `--slow-ramp-fraction` at a lower dose, finally validated against the real baseline — a genuine improvement

Two candidates remained open at this point: sweep `--smoothness-penalty-weight`
down from the over-regularized 0.01, and validate `--slow-ramp-fraction`
(the data-augmentation candidate from the very first attempt, never
properly tested) against the now-real `lstm_baseline_paper2.pt` baseline.
Both were run, in parallel, against the same `timegan_paper2.pt` generator:
`--smoothness-penalty-weight 0.001` (10x lower) and `--slow-ramp-fraction
0.05` (3x lower than the destabilizing 0.15 tried in the first attempt,
which was itself never tested against a real bug-reproducing baseline).

**The velocity probe on the slow-ramp checkpoint is confusing on its own
terms** — at landing +0.066, the pattern isn't a cleaner version of the
baseline's recovers-when-slow/stuck-when-fast signature, it's closer to
*inverted*: the slowest ramp (30 steps) is now the one stuck near 0
(end-state 0.0001), while every faster ramp (1-10 steps) recovers well
(0.92-0.999). The same inversion shows at landing +0.086 (30-step ramp
stuck at 0.00002, everything faster recovering to ≥0.98). This is a
genuinely surprising result given the augmentation specifically trains on
*slow* ramps through this zone — one plausible explanation is that the
augmented training examples (a linear ramp followed by a held, jittered
plateau) are themselves a narrow, out-of-training-distribution *shape*
distinct from a real market path or from this probe's continuous 30-step
ramp construction, and the network learned something specific to that
shape rather than a general "handle slow approaches" rule. Read in
isolation, this probe result would suggest the fix didn't generalize the
way intended.

**But the actual regime-switching stress test (paper-scale, 500,000 paths)
tells a different, much clearer story, and it's the one that matters:**

| Strategy | Mean | Std | CVaR₉₅ | CVaR₉₉ | Skew | Excess kurtosis | Tx. cost |
|---|---|---|---|---|---|---|---|
| Black-Scholes | -0.031 | 0.290 | 0.91 | 1.42 | -2.02 | 8.1 | 0.0057 |
| LSTM (TimeGAN) baseline | -0.039 | 0.756 | 2.20 | 4.23 | -3.36 | 44.6 | 0.0136 |
| LSTM (TimeGAN), slow-ramp-fraction=0.05 | **-0.038** | **0.631** | **1.76** | **3.27** | **-2.36** | **28.2** | **0.0128** |

**Every risk metric improves over the baseline, and by a wide margin**:
CVaR₉₅ down 20% (2.20 → 1.76), CVaR₉₉ down 23% (4.23 → 3.27), excess
kurtosis down 37% (44.6 → 28.2), skew closer to zero (-3.36 → -2.36),
transaction cost slightly lower, mean wealth essentially unchanged. This
holds at both a 100,000-path check and the full paper-scale 500,000-path
batch — not a small-sample artifact. **This is the first genuinely
promotable fix candidate this investigation (across two sessions) has
found.**

**Net assessment**: `--slow-ramp-fraction 0.05` measurably reduces tail
risk for LSTM (TimeGAN) relative to the baseline, confirmed at paper scale
on the metric that actually matters (the stress test), even though the
narrow velocity-ramp probe used to diagnose and chase mechanism (b)
throughout this document doesn't read as a clean "fix" in isolation — a
useful methodological lesson paired with the smoothness-penalty result
above: **neither the velocity probe alone nor the stress test alone tells
the whole story; both are needed, and they can disagree.** Not yet
promoted to a committed checkpoint (this remains a scratch/diagnostic
artifact of this session, matching how every other fix attempt in this
document was validated before promotion) — the concrete next steps are a
`--slow-ramp-fraction` dose sweep (0.05 wasn't chosen for being optimal,
only for being lower than the destabilizing 0.15), a multi-seed check
(everything in this section is single-seed, and this project has
repeatedly found seed-sensitivity to matter for exactly this class of
result — see the α=0.995 dip finding elsewhere in this document), and
promoting the checkpoint into `checkpoints/` proper once those hold up.

#### Follow-up: multi-seed check confirms the fix, and reveals it's suppressing something worse than the single-seed table showed

A later session, with MPS acceleration available (see `common/device.py` —
~5.5x faster than CPU for this exact workload, turning a 78-minute
paper-scale run into ~11 minutes), ran seeds 1-4 for both baseline and
`--slow-ramp-fraction 0.05` against the same `timegan_paper2.pt` generator
(seed 0 already existed from the attempt above). Full 500,000-path
stress-test results, all five seeds:

| Seed | Baseline CVaR₉₉ | Baseline ExKurt | Slow-ramp CVaR₉₉ | Slow-ramp ExKurt |
|---|---|---|---|---|
| 0 | 4.23 | 44.6 | 3.27 | 28.2 |
| 1 | 4.47 | 49.0 | 3.37 | 29.7 |
| 2 | 4.48 | 142.7 | 2.78 | 841.9 |
| 3 | 3.92 | **179,437.9** | 3.97 | 38.4 |
| 4 | 3.32 | 21.1 | 2.61 | 11.6 |

**CVaR₉₉ improves in 4 of 5 seeds** (mean 4.08 → 3.20, a 21.6% reduction,
consistent with the original single-seed 23% finding); seed 3 is the one
exception, a small regression (3.92 → 3.97). But **excess kurtosis is where
this multi-seed run earns its keep**: seed 3's *baseline* checkpoint has a
kurtosis of 179,438 — the same order of magnitude as the original
documented paper-scale LSTM (TimeGAN) catastrophe (~81,035) that motivated
this entire investigation — while every other baseline seed looks merely
elevated (21-143). **This is exactly the seed-sensitivity this project has
repeatedly found elsewhere** (Basic RNN's bimodal seeds, the α=0.995 dip):
the single-seed table above was not simply "the" LSTM (TimeGAN) baseline,
it was one roll of a die that occasionally lands on a catastrophic
outlier. The corresponding slow-ramp checkpoint at that *same* seed drops
kurtosis to 38.4 — a >4,600x reduction — even though its CVaR₉₉ is
marginally worse than baseline there (kurtosis is far more sensitive to a
single extreme path than CVaR₉₉'s 99th-percentile averaging, so a CVaR₉₉
that looks unremarkable can still hide a catastrophic single-path tail,
which is exactly what happened at seed 3). Across all 5 seeds: mean
kurtosis including each condition's own worst outlier is 35,939 for
baseline vs. **190 for slow-ramp** (189x better); the worst single seed's
kurtosis is 213x smaller under slow-ramp (841.9 vs. 179,437.9).

**Revised net assessment: the fix is stronger than the single-seed result
suggested, not weaker.** It doesn't just shave CVaR₉₉/kurtosis in the
typical case — it substantially suppresses baseline's occasional
catastrophic-outlier failure mode, the specific pathology (rare paths with
extreme, seed-dependent tail losses) that RESULTS.md's "Catastrophic tail
risk" section names as this project's headline paper-scale finding. This
is now a confirmed, multi-seed-validated result, not a single lucky draw.

#### Follow-up: dose sweep — 0.05 wasn't the optimum, and the dose-response is sharply non-monotonic

Same seed (0), same generator, `--slow-ramp-fraction` at 0.02, 0.10, 0.15,
0.20, read with the full 500,000-path stress test:

| Dose | CVaR₉₉ | Excess kurtosis |
|---|---|---|
| 0.00 (baseline) | 4.23 | 44.6 |
| 0.02 | 2.69 | 24.5 |
| 0.05 (multi-seed validated above) | 3.27 | 28.2 |
| **0.10** | **2.75** | **9.7** |
| 0.15 | **21.94** | **26,180.9** |
| 0.20 | 4.38 | 246.7 |

**Both 0.02 and 0.10 beat the already-validated 0.05** — this isn't a
monotone "more augmentation is better" relationship. **0.15 is
catastrophic**, not merely worse: CVaR₉₉ and kurtosis both blow up by
1-3 orders of magnitude, consistent with the very first (never-properly-
validated) attempt in this document, which also found 0.15 destabilizing,
on a completely different, non-representative baseline — two independent
signals now point at instability specifically in this dose region, not
just one. **0.20 partially recovers** but is still worse than baseline on
every metric (CVaR₉₉ 4.38 vs. 4.23, kurtosis 246.7 vs. 44.6) — the
training dynamics don't monotonically improve as the dose keeps rising
past the unstable region either.

**Caveat, learned directly from the multi-seed check just above**: this
sweep is single-seed (seed 0 only). Given seed 3's baseline just showed a
179,438 kurtosis where every other baseline seed showed 21-143, a
single-seed dose comparison risks exactly the same trap the original 0.05
finding was in before it got multi-seeded — 0.15's catastrophic reading in
particular could in principle be partly an unlucky seed-0 draw at that
specific dose rather than a property of the dose itself, though the
independent cross-validation against attempt 1's finding (0.15
destabilizing a *different* baseline entirely) makes that less likely than
it would be for an isolated data point. Dose 0.10 (lowest kurtosis, second-
lowest CVaR₉₉) was carried forward to its own 4-seed multi-seed check
before being treated as a promotion candidate — see the follow-up below.

#### Follow-up: dose=0.10's single-seed edge over 0.05 doesn't survive multi-seeding — 0.05 is the one to promote

Dose 0.10 at seeds 1-4 (seed 0 already existed from the sweep above),
read the same way as the earlier multi-seed check:

| Dose | Mean CVaR₉₉ (5 seeds) | Mean excess kurtosis (5 seeds) | Catastrophic seeds (kurtosis > 500) |
|---|---|---|---|
| baseline | 4.08 | 35,939.1 | 1/5 |
| **0.05** | **3.20** | **190.0** | **1/5** |
| 0.10 | 8.08 | 3,806.1 | 2/5 |

**This is exactly the trap the dose-sweep section above warned about, now
confirmed rather than merely hypothesized.** Dose 0.10 looked like the
clear winner at seed 0 alone (CVaR₉₉ 2.75, kurtosis 9.7 — better than
0.05's seed-0 numbers on both metrics). Read across 5 seeds, it's worse
than *baseline* on mean CVaR₉₉ (8.08 vs. 4.08) and has *more* catastrophic
seeds than dose 0.05 (2/5 vs. 1/5) — two of its five seeds (1 and 3) land
on severe outliers (kurtosis 3,546 and 15,433) that seed 0 alone gave no
hint of. **Dose 0.05 remains the best-supported choice of everything
tested in this document** — the only dose validated at 5 seeds with a
consistent, moderate improvement over baseline on every seed-aggregate
metric, rather than a single favorable seed masking higher variance.
**This is the dose to promote**, not 0.10 or any other point in the sweep.

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
| ~~GRU (TimeGAN)~~ **GRU (TimeGAN, `moneyness_clip` fix)** | ~~-0.030~~ -0.041 | ~~8.21~~ **6.01** | ~~**31.97**~~ **25.52** | ~~**-307.8**~~ **-314.3** | ~~**145,523**~~ **126,426** | ~~0.0111~~ 0.0082 |

**GRU (TimeGAN)'s row reflects a promoted fix** (`RecurrentHedgingAgent.moneyness_clip`,
retrained from scratch with the clip active for the full paper-scale 25,000
steps, not just applied at inference time) — see [the fix-attempt
writeup](#fix-attempt-clipping-the-rnns-log-moneyness-input-at-the-training-boundary--free-no-retraining-works-for-gru-does-not-work-for-lstm)
and the [training-from-scratch
follow-up](#follow-up-training-with-the-clip-active-from-the-start-closes-more-of-the-gap-than-the-inference-only-wrapper-did)
for the full story. CVaR₉₅/₉₉ and the below_-50/-10 path counts all improve
20-30%; mean wealth and the single worst-case loss are both slightly worse,
so this is a genuine but partial fix, the same "improved, not fully
closed" pattern already on record for GRU (WGAN-GP)'s `grad_clip_norm` fix
above. LSTM (TimeGAN)'s row above is left unchanged, not struck through —
clipping specifically was tested against it and found not to help (see the
fix-attempt writeup), but a different fix (below) eventually did.

**LSTM (TimeGAN) update, promoted**: `--slow-ramp-fraction 0.05` (data
augmentation exposing the policy to synthetic slow log-moneyness ramps
through the critical transition zone — see the [full
diagnosis](#fix-attempt-continued-a-paper-scale-reproduction-that-actually-shows-the-bug-and-a-smoothness-penalty-candidate-that-trades-the-symptom-for-a-worse-disease)
and its [multi-seed](#follow-up-multi-seed-check-confirms-the-fix-and-reveals-its-suppressing-something-worse-than-the-single-seed-table-showed)
and [dose-sweep](#follow-up-dose05s-single-seed-edge-over-010-doesnt-survive-multi-seeding--005-is-the-one-to-promote)
follow-ups) is now `checkpoints/hedging_agent_lstm_timegan.pt`. Unlike
GRU's row above, this is **not** a like-for-like fix against the *same*
generator checkpoint this table's other rows used — that original attempt-4
TimeGAN checkpoint wasn't preserved, so a fresh one was retrained at
identical scale/methodology (`--data-source yfinance`, batch=178,
2000/2000/6000-epoch phases) for this fix's own baseline-vs-fixed
comparison, which came out somewhat differently calibrated (diversity
110.6% vs. this table's 87.3%). Read the numbers below as "this fix helps
substantially, confirmed at 5 seeds, against an independently-retrained but
equivalent-methodology generator" — not as a literal replacement for the
42.13/-249.4/81,035 above, which remain the historical record for that
specific (unpreserved) checkpoint. Official `backtester/evaluate.py`
methodology (`run_backtest`, seed=42, 500,000 paths), against the
promoted checkpoint:

| Strategy | Mean wealth | CVaR 95% | CVaR 99% | Skew | Excess kurtosis | mean tx. cost |
|---|---|---|---|---|---|---|
| Black-Scholes | -0.033 | 1.20 | 1.85 | -2.24 | 7.6 | 0.0065 |
| LSTM (TimeGAN), `slow_ramp_fraction=0.05` | -0.040 | 1.75 | **3.24** | **-2.21** | **24.5** | 0.0128 |

CVaR₉₉ 3.24 and kurtosis 24.5 here are both close to Black-Scholes' own
7.6/1.85 — a dramatically smaller gap than the original 42.13/81,035, and
consistent with (not just a rerun of) the 5-seed validation above (mean
CVaR₉₉ 3.20, mean kurtosis 190 across seeds 0-4 on the scratch-methodology
comparison). This is the first LSTM (TimeGAN) checkpoint promoted into
`checkpoints/` since the catastrophic-tail-risk finding was first
documented.

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
catastrophic paths, 238 vs. 402-793). Read against the tail-risk finding,
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

### Investigating why the best-fidelity generator produced the worst policies: `validate.py` checks the wrong invariant

A later session picked this open question up directly, motivated by a
concrete hypothesis from this document's own LSTM (TimeGAN) mechanism (b)
work: that fix (`--slow-ramp-fraction`) targets *path velocity* — how fast
log-moneyness moves step to step — not any property of the terminal price
distribution. `validate.py`'s fidelity checker, by contrast, only ever
inspects the terminal/cumulative return distribution (diversity ratio,
mean bias, skewness, kurtosis — all computed on the price at the end of
the path). **If TimeGAN can satisfy a terminal-only check while generating
paths with unrealistic intra-path dynamics, that would directly explain
"best terminal fidelity, worst policy behavior": the checker is measuring
an invariant the policies aren't actually sensitive to, and blind to the
one (path velocity) they are.**

Tested directly: 5,000 real 31-day `^GSPC` windows vs. 5,000 synthetic
paths from `timegan_paper2.pt` (this session's own from-scratch,
fidelity-checker-"OK" TimeGAN — diversity 110.6%, skew diff -0.36,
kurtosis diff -1.44, all within threshold). Per-step (not terminal) log-return
statistics:

| Metric | Real | Synthetic | Ratio |
|---|---|---|---|
| Per-step log-return std | 0.00964 | 0.01929 | **2.00x** |
| \|return\| lag-1 autocorrelation (vol clustering) | -0.019 | **0.409** | — |
| Signed-return lag-1 autocorrelation (momentum) | 0.066 | **0.468** | — |
| Fraction of steps with \|return\| > 4% | 0.48% | 3.71% | **7.7x** |
| Max single-step \|return\| (p99, across all steps) | 0.070 | 0.142 | 2.0x |
| Terminal (31-day cumulative) log-return std | 0.0521 | 0.0540 | **1.04x** |

**The per-step volatility is exactly double real markets', with 7.7x more
frequent large single-step moves and much stronger short-lag momentum —
yet the terminal, 31-day cumulative std is almost identical (1.04x, not
the ~2x an i.i.d.-steps model would predict from doubled per-step vol).**
This is the mechanism: TimeGAN's synthetic paths are locally far more
volatile and momentum-heavy step-to-step than real data, but this doesn't
compound into terminal-distribution error the way it would for genuinely
i.i.d. steps, because something in the architecture (plausibly the
Recovery network, trained via reconstruction loss against real 31-day
window *shapes*, implicitly regularizing how far the decoded endpoint can
drift regardless of how jumpy the Generator/Supervisor's own latent
trajectory is) pulls the endpoint back toward a realistic overall span. A
terminal-only fidelity check is structurally blind to this: it can't
distinguish a path that reaches a realistic endpoint via realistic
day-to-day moves from one that reaches the same endpoint via a much
jumpier, more clustered, unrealistic route — and it's exactly the *route*,
not the endpoint, that a recurrent policy consuming the whole path step by
step is sensitive to (directly demonstrated by this document's own LSTM
mechanism (b): the failure was a *velocity*-triggered transition, not a
level-triggered one).

**This is a known, documented limitation of TimeGAN specifically, not an
artifact of this implementation** — the broader time-series-GAN literature
describes exactly this tension: models that optimize for marginal/terminal
distributional fidelity can distort temporal dynamics and autocorrelation
structure along the way, since nothing in a marginal-distribution loss
constrains the path taken to get there.

**This reframes item 7's open question below, not just answers it.**
"TimeGAN's diversity is closer to 100% but still undershoots" was always
being read as *the* fidelity gap to close; this finding says the diversity
ratio (a terminal-distribution statistic) was never going to be sufficient
evidence of a good generator for recurrent-policy training regardless of
how close to 100% it gets, because it doesn't check the thing that
actually matters downstream.

**Implemented, not just proposed.** `validate.py::validate_generator_fidelity`
now runs three additional, independent path-dynamics checks alongside the
four terminal-distribution ones: per-step return volatility ratio
(`STEP_VOL_RATIO_LOW_THRESHOLD`/`_HIGH_THRESHOLD`, 0.67-1.5x), signed-return
lag-1 autocorrelation (momentum/mean-reversion,
`SIGNED_AUTOCORR_DIFF_WARNING_THRESHOLD` 0.25), and \|return\| lag-1
autocorrelation (volatility clustering,
`ABS_AUTOCORR_DIFF_WARNING_THRESHOLD` 0.25) — the exact three statistics
this investigation measured, now permanent (`common/stats.py::step_log_returns`,
`lag1_autocorrelation`), with their own unit tests (constructed fixtures
that isolate each new check from every other statistic in the file,
including from each other) and end-to-end verified against the real
checkpoint that motivated this investigation: `timegan_paper2.pt`, which
previously printed a clean "OK", now correctly prints `WARNING: per-step
volatility is 200.0% of real ... per-step return autocorrelation off by
+0.40 ... volatility clustering off by +0.43`. Applies automatically to
both generators (`train_gan.py` and `train_timegan.py` share this one
function), and to every future TimeGAN calibration attempt in this
document going forward — this class of failure can no longer print "OK".

#### Follow-up: can a TimeGAN be trained to pass the new path-dynamics checks? A 4-variant sweep, none fully clean, one close

With the checker now able to see path dynamics, the natural next question
is whether any hyperparameter change actually fixes them, or whether this
is a structural TimeGAN limitation no amount of tuning escapes. Targeted
hypothesis: `--lambda-supervised` is the one loss term specifically
designed to enforce realistic one-step-ahead latent dynamics (temporal
coherence), while `--lambda-moment`/`--lambda-diversity` only ever pull
toward *terminal* statistics — if the terminal-focused losses dominate
optimization pressure away from what the supervised loss wants, boosting
supervised (or weakening the terminal losses) should help. Four variants,
same seed (1, not seed 0, to also check whether the original finding
was seed-specific), paper scale, `--data-source yfinance`:

| Variant | Step-vol ratio | Signed autocorr diff | \|return\| autocorr diff | Skew diff |
|---|---|---|---|---|
| baseline (fresh seed) | OK | **+0.68** | **+0.62** | OK |
| `--lambda-supervised 10.0` | **236.4%** (worse) | OK | +0.34 | OK |
| `--lambda-moment 0.2 --lambda-diversity 0.2` | OK | +0.56 | +0.52 | OK |
| combined (both above) | OK | +0.25 (borderline) | OK | **-0.85** (new failure) |

**No variant passes cleanly, but the pattern is informative, not just
noisy.** The fresh-seed baseline reproduces (and somewhat worsens) the
original autocorrelation problem this investigation was built on — this
isn't a seed-0-specific artifact. `--lambda-supervised` alone trades one
problem for another: it fixes signed-return autocorrelation (momentum)
completely, but step-vol *worsens* to 236% (from an already-passing
baseline) and volatility clustering only partially improves — apparently,
without the terminal-distribution losses actively constraining scale, a
stronger supervised loss lets per-step amplitude drift further from real.
Weakening the terminal losses *alone*, without boosting supervised,
makes autocorrelation worse, not better — contradicting the simple
"terminal losses are fighting supervised" framing; if anything they were
incidentally helping constrain dynamics. **The combined variant is the
closest of the four to passing everything**: only 2 residual issues
(borderline momentum, and a new skew problem from weakening
`--lambda-moment` too far) instead of 2-3 severe path-dynamics failures in
every other variant — suggesting the right region is roughly "high
supervised weight, low-but-not-zero terminal-loss weights," not any single
lever in isolation. A refined attempt (`--lambda-supervised 20.0
--lambda-moment 0.5 --lambda-diversity 0.2`, restoring some moment-loss
weight to recover skew control while keeping the other two changes) was
run as a direct follow-up to this table — see below for the result.

**The refinement made things worse, not better — this isn't a simple knob
to turn.** Same seed, `--lambda-supervised 20.0 --lambda-moment 0.5
--lambda-diversity 0.2`:

| Variant | Step-vol ratio | Signed autocorr diff | \|return\| autocorr diff | Skew diff |
|---|---|---|---|---|
| combined (supervised=10, moment=0.2, diversity=0.2) | OK | +0.25 (borderline) | OK | -0.85 |
| refined (supervised=20, moment=0.5, diversity=0.2) | OK | **+0.31** (worse) | **+0.28** (newly failing) | -0.56 (better, still fails) |

Doubling `--lambda-supervised` further (10 → 20) made signed-return
autocorrelation *worse* (+0.25 → +0.31), not better, and volatility
clustering — which had passed cleanly in the "combined" variant — newly
failed (+0.28). Partially restoring `--lambda-moment` (0.2 → 0.5) did
improve skew (-0.85 → -0.56) but not enough to clear the 0.5 threshold.
**Five runs (one baseline plus four hyperparameter variants) show a
non-monotonic relationship between these loss weights and the resulting
path dynamics** — "more supervised loss weight" is not simply "better
temporal coherence," and there is real interaction between all three
terms that a handful of single-axis or two-axis nudges doesn't cleanly
navigate.

**Honest stopping point for this sub-investigation, not a resolved
question.** No configuration tried passes all 7 fidelity checks
simultaneously. This is consistent with (not proof of) the literature
framing cited above — that marginal/terminal-distribution fidelity and
temporal-dynamics fidelity are in genuine tension for GAN-based time-series
models, not simply under-weighted relative to each other — but five runs
across a 2-D corner of the hyperparameter space isn't enough to distinguish
"this needs a proper grid search or a different loss formulation entirely"
from "this specific tension is structurally unresolvable with TimeGAN's
current loss design." A real grid search (more values per axis, ideally
with multi-seed replication given this project's repeated experience with
seed-sensitivity), or a structurally different intervention (e.g. an
explicit path-dynamics loss term added directly to phase 3 training,
analogous to how the moment- and diversity-matching losses were added for
the terminal-distribution gaps this document's TimeGAN attempts 1-4
chased) are the two directions this leaves open. Neither is started.

#### Follow-up: a dedicated path-dynamics loss term passes all 7 checks — the first TimeGAN checkpoint in this project's history to do so

The second of the two open directions above, tried directly rather than
left open: an explicit path-dynamics-matching loss added to phase 3
training (`TimeGANTrainer`'s `lambda_dynamics`/`target_step_std`/
`target_signed_autocorr`/`target_abs_autocorr`, `--lambda-dynamics` on the
CLI, default enabled), targeting the exact three statistics
`validate.py`'s new checks measure — per-step return std ratio, signed
lag-1 autocorrelation, and \|return\| lag-1 autocorrelation — the same
way `lambda_moment`/`lambda_diversity` already targeted the four terminal
ones. Backed by `common/stats.py::lag1_autocorrelation_tensor` (a
differentiable sibling of the existing float version, with a deliberate
0.0-not-NaN fallback for degenerate batches so it can't poison an
unrelated gradient), and 6 new tests (differentiability, a
zero-loss-without-target check, and a "pulls synthetic step-std toward
target over 250 training steps" behavioral test mirroring the existing
diversity-loss test).

**Paper scale, all defaults (every lambda at 1.0, including the new
`lambda_dynamics`), `--data-source yfinance`, seed=0 — passes all 7 checks
on the first attempt:**

| Check | Value | Threshold | Verdict |
|---|---|---|---|
| Diversity ratio | 83.8% | 30-170% | OK |
| Mean bias | +0.2σ | 2.0σ | OK |
| Skew diff | -0.03 | 0.5 | OK |
| Kurtosis diff | -0.87 | 2.0 | OK |
| **Step-vol ratio** | **115.3%** | 67-150% | **OK** |
| **Signed autocorr diff** | **-0.07** | 0.25 | **OK** |
| **\|return\| autocorr diff** | **+0.17** | 0.25 | **OK** |

**This is the first TimeGAN checkpoint anywhere in this document's
four-plus-attempt history to pass both terminal-distribution and
path-dynamics fidelity simultaneously**, and it did so without any
hyperparameter search at all — plain defaults, first try. Contrast this
directly with the 5-run reweighting sweep just above, which never got
closer than 2 residual failures even after deliberately tuning
`lambda_supervised`/`lambda_moment`/`lambda_diversity` across several
configurations: **a dedicated loss term aimed directly at the target
statistic beat reweighting existing, indirectly-related losses**, the same
lesson this project already learned once before for the terminal
distribution (moment-matching/diversity-matching losses were added, not
found by tuning the adversarial/supervised loss weights). Skew diff (-0.03)
and kurtosis diff (-0.87) are both tighter here than any of attempts 1-4's
own terminal-only-optimized checkpoints achieved (compare attempt 4's
skew diff -0.007/kurtosis diff -0.99 — closely matched, not exceeded, but
achieved *simultaneously* with passing path dynamics, which no attempt-1-4
checkpoint even measured, let alone passed).

**Not yet a promoted checkpoint — this is a fidelity-check result, not a
downstream validation.** Every other fix in this document was validated by
retraining a policy against the corrected artifact and stress-testing it
before promotion (the LSTM `--slow-ramp-fraction` fix, GRU's
`moneyness_clip`, GRU's `grad_clip_norm`). This checkpoint
(`timegan_dynamics_loss.pt`, scratch only) hasn't been through that step:
it's unknown whether training a policy against a generator that's
path-dynamics-realistic (not just terminal-realistic) produces better
downstream stress-test behavior than `--slow-ramp-fraction` already
achieves against the old-style generator, makes no difference, or
interacts with it in some other way. That is the natural next experiment
this finding opens up, not yet run.

#### Follow-up: training against the path-dynamics-fixed generator makes the downstream policy dramatically *worse*, not better

The obvious next experiment, run directly: baseline and
`--slow-ramp-fraction 0.05` LSTM (TimeGAN) policies trained against
`timegan_dynamics_loss.pt` instead of the old-style generator, same seed
(0), full 500,000-path stress test:

| Config | CVaR₉₉ | Excess kurtosis |
|---|---|---|
| old generator, baseline | 4.23 | 44.6 |
| old generator, `--slow-ramp-fraction 0.05` (promoted) | 3.27 | 28.2 |
| **new (path-dynamics-fixed) generator, baseline** | **37.52** | **12,499.2** |
| new generator, `--slow-ramp-fraction 0.05` | 34.34 | 13,098.0 |

**This is the opposite of the hypothesis this whole sub-investigation was
built on.** Fixing the generator's path dynamics didn't reduce the
downstream policy's tail risk — it made it roughly **9x worse on CVaR₉₉
and ~280x worse on kurtosis**, and `--slow-ramp-fraction`, which cut
tail risk substantially on the old generator, barely moves the needle here
(and if anything worsens kurtosis slightly, 12,499 → 13,098).

**Likely mechanism, directly traceable to a number already on record**:
the new generator's own diversity ratio is 83.8% of real — passing the
fidelity checker's 30-170% threshold cleanly, but *meaningfully narrower*
than the old generator's 110.6%. A narrower generator means the policy
sees a narrower range of terminal outcomes during training, which is
*exactly* the ingredient behind mechanism (b) as documented throughout
this section: TimeGAN-trained recurrent policies generalize badly to
price extremes their own training distribution didn't cover. Fixing path
dynamics (this session's new loss term) and fixing training-distribution
coverage (diversity) are two different levers, and this generator
improved one at what looks like a cost to the other, even though neither
number individually crosses a fidelity-checker threshold. Passing all 7
checks was never a guarantee of producing a *better* training
distribution for a recurrent policy, only a more accurately-*described*
one against real data's own statistics — real data's diversity and real
data's path dynamics are both real properties, but matching them more
tightly doesn't automatically make the resulting synthetic distribution
more informative for a policy that needs to see extremes to hedge them.

**Single seed — flagged, not overclaimed, but the magnitude here is much
larger than any prior seed-noise range this document has measured** (the
LSTM multi-seed check's own worst-to-best CVaR₉₉ spread was roughly
2.6-4.5; this is a 9x jump). Given this project's repeated experience with
single-seed results reversing under multi-seeding (LSTM dose=0.10, Basic
RNN's stacked fix), a multi-seed check is the responsible next step before
concluding the path-dynamics-fixed generator is actually worse in general
rather than unlucky at seed 0 — but a magnitude this large makes "purely
seed noise" a less likely explanation than it was for those smaller
(2-3x) prior cases. **Not promoted. The old generator + `--slow-ramp-fraction
0.05` combination remains the best validated result in this document.**

## Rebuilding every checkpoint from scratch, and multi-seeding the fixes that shipped

This section is a single session's work, run after every gitignored
checkpoint except `timegan.pt` and `hedging_agent_lstm_timegan.pt` had been
lost from disk — i.e. from the state a fresh clone starts in. That accident
turned out to be the most informative experiment in this document: it forced
every published row to be re-derived rather than cited, and four of them
didn't survive. Everything below was run at this document's own scale
(500,000-path regime-switching stress test, seed 42) on an M5 Pro with MPS.

### Two bugs that made the repo unreproducible

**`tests/test_tail_risk.py` failed rather than skipped.** Its
`_checkpoints_available` guard globs `hedging_agent*.pt`, which the surviving
TimeGAN LSTM checkpoint matches, so the scan ran with every WGAN-GP
checkpoint absent. `_load_all_policies` then hit its demo-MLP fallback and
returned that untrained policy under the `"MLP"` display name, and the
known-good regression test graded it as a production checkpoint (66
catastrophic paths against a required 0). `scan_checkpoint_tail_risk`
compares against this document's per-checkpoint figures, so a demo policy
there is worse than no policy: it reads as a catastrophically-regressed
production checkpoint. `_load_all_policies` now takes `allow_demo_fallback`
(default `True`, preserving the `evaluate.py` CLI's behaviour); the scan
passes `False`.

**`train_gan.py` crashed immediately on any Apple Silicon machine.**
`select_device()` returns MPS there, but WGAN-GP's gradient penalty
double-backwards through the LSTM discriminator, which MPS does not
implement as of torch 2.8 — so this repo's own README quickstart
(`python src/generator/train_gan.py --epochs 200`) died with `derivative for
lstm_mps_backward is not implemented` before the first epoch. This is the
limitation `common/device.py` already documents for
`PolicyTrainer.smoothness_penalty_weight`, except unconditional rather than
flag-gated: the penalty *is* the objective, so there is no MPS-compatible
mode. Auto-detection now steps down to CPU with a printed note; an explicit
`--device mps` raises with the reason.

### What reproduced, and what didn't

A fresh WGAN-GP generator (1500 epochs, lr 3e-4, `^GSPC`) plus fresh seed-0
policies at promoted settings:

| Strategy | CVaR₉₅ (published) | CVaR₉₉ (published) | Excess kurtosis (published) | `<-50` |
|---|---|---|---|---|
| MLP | 2.24 (2.38) | 3.48 (3.69) | 7.5 (6.9) | 0 |
| Basic RNN | 1.59 (1.64) | 2.46 (2.58) | 7.4 (7.5) | 0 |
| LSTM | 2.24 (2.17) | 3.58 (3.49) | 8.6 (8.5) | 0 |
| GRU | **4.09** (2.14) | **12.28** (3.81) | **212,251** (3,078) | **126** (4) |

The regenerated generator is over-dispersed at 352.7% of real terminal-return
std — but so was the original (469%; its committed
`results/gan_fidelity_summary.json` verdict predates the overshoot check and
never flagged it). Less over-dispersed, not more, so generator quality does
not explain GRU's regression.

**Basic RNN (TimeGAN) reproduced to five significant figures** — saturated
fraction 1.0000, delta span 0.17308 (documented 0.173), `weight_hh_l0` norm
19.186 (19.19), `weight_hh_l1` 17.911 (17.91). That it matches across a
*regenerated* generator is itself evidence for the diagnosis: a hidden state
pinned at ±1.0 regardless of input produces an input-independent end state,
so which generator supplied the inputs barely matters.

**MLP (TimeGAN) did not.** It comes back with 0.88% of paths below -10
against `tests/test_tail_risk.py`'s `< 0.1%` known-good bound — no
catastrophic paths, but not clean either. (This single-seed figure is
revisited [below](#re-anchoring-all-four-timegan-rows-to-the-surviving-generator): across 5 seeds the rate spans 0.012%-0.877%,
and 0.877% is the worst of the five.) The cause is provenance, not
regression: this document already records that attempt 4's generator was not
preserved and that `checkpoints/timegan.pt` is a later retrain. The TimeGAN
table rows are anchored to a generator that no longer exists.

### The "reproduction anchor" numbers identify a dead policy, not a checkpoint

The α=0.995 sweep above uses a reproduction anchor: α=0.997 seed 0 matching
`worst_loss` -6202.48, 814 paths `<-50`, and CVaR₉₅/CVaR₉₉/skew/kurtosis
11.76 / 43.15 / -248.9 / 80,781. An explicit `delta ≡ 0` policy on the same
scenario scores:

| Metric | never-hedge | anchor |
|---|---|---|
| CVaR₉₅ | 11.756 | 11.76 |
| CVaR₉₉ | 43.154 | 43.15 |
| skew | -248.86 | -248.9 |
| excess kurtosis | 80,780.7 | 80,781 |
| `worst_loss` | -6202.4834 | -6202.48 |
| `<-50` | 814 | 814 |

Exact on all six. These are the constant *any* degenerate policy produces
here. The anchor remains valid as a harness/scenario check — it does prove
the scan and the original agree exactly — but the inference built on top of
it does not: matching these numbers establishes "this policy is degenerate",
not "this is that checkpoint". The dead-policy provenance claim rests on the
logit inspection cited alongside it, not on the anchor. The upside is a free,
instant dead-policy detector, which is how the GRU (TimeGAN) collapse below
was caught.

### α=0.995: closed, and validated against the failure that motivated it

This document's highest-priority open item was that α=0.995 never received
the `grad_clip_norm=1.0` its α=0.99/0.997 neighbours got, on the evidence
that its own seed-1 draw would ship 5,544 stress-test paths losing more than
10x the premium. Retrained both ways at seed 1:

| α=0.995, seed 1 | CVaR₉₅ | CVaR₉₉ | `worst_loss` | `<-50` | `<-10` |
|---|---|---|---|---|---|
| unclipped | 9.61 | 17.84 | -56.4 | 6 | **8,495** |
| `--grad-clip-norm 1.0` | 2.16 | 3.37 | -8.9 | **0** | **0** |

The failure reproduces more severely than documented (8,495 rather than
5,544 below -10, plus 6 catastrophic paths the original sweep did not report
at this α), and clipping eliminates it completely. Seed 0 is clean too.
**Closed on direct evidence at two seeds**, and the promoted
`checkpoints/hedging_agent_mlp_alpha0_995.pt` is now the clipped run.

### Both promoted GRU fixes fail multi-seed validation

Five seeds each, paired (conditions share a seed, so the comparison is
paired), 500,000-path stress test throughout.

**GRU (WGAN-GP), `--grad-clip-norm 1.0` — inert.**

| Condition (5 seeds) | CVaR₉₅ | CVaR₉₉ | `<-50` |
|---|---|---|---|
| baseline | 3.61 ± 0.99 | 9.46 ± 3.45 | 78.4 ± 51.6 |
| `--grad-clip-norm 1.0` | 3.71 ± 0.82 | 9.84 ± 3.52 | 92.6 ± 65.6 |

Improves 2/5 seeds on CVaR₉₉, 2/5 on CVaR₉₅, 2/5 on `<-50`; the mean is
slightly *worse* on every headline metric. Seed 1 shows exactly 0.000% change
on all six — its clipped and unclipped checkpoints have **bit-identical
weights** (max absolute difference 0.0).

That is the mechanism. Measured pre-clip gradient norms over 3,000 steps:
median 0.022, p99 0.164, max 0.707 at seed 0; median 0.024, p99 0.213, max
0.715 at seed 1 — **0.00% of steps exceed 1.0 in either**. The threshold sits
~45x above the median gradient norm. It is not a regularizer on this
workload; it is a rare-event trigger that fires only on occasional
late-training spikes, and whether any spike crosses 1.0 at all is
seed-dependent.

**This resolves a puzzle this document explicitly left open.** The
`grad_clip_norm` fix-attempt writeup records that the weight-growth
hypothesis motivating it was falsified — "clipped and unclipped final weight
norms are nearly identical" — and treats that as a loose end that also
weakens the case for the untried LR-warmup variant. Measured per seed:

| seed | weights identical? | max abs weight diff | final total norm (base → clipped) |
|---|---|---|---|
| 0 | no — fired | 12.37 | 239.7 → 245.5 (×1.024) |
| 1 | **yes — never fired** | 0 | 243.7 → 243.7 (×1.000) |
| 2 | no — fired | 9.48 | 242.0 → 232.1 (×0.959) |

Clipping does not constrain weight magnitude (norms within 4%, exactly as
observed) but does relocate the solution entirely (individual weights differ
by up to 12.4) — same-radius sphere, different point. Falsified hypothesis
and near-identical norms are the same fact, not two. Functionally the flag
behaves as a **seed perturbation**, which predicts precisely the 2/5
coin-flip above, and the documented 34 → 4 improvement was a favourable draw.

**Follow-up: clipping is the wrong intervention here, not merely
mis-tuned.** Because the promoted threshold barely engages, it never tested
the underlying idea. Rerun at thresholds picked from the measured gradient
distribution — 0.05 clips roughly the top 10-25% of steps, 0.10 the top 5%:

| Condition | CVaR₉₉ | `<-50` | seeds improved |
|---|---|---|---|
| baseline (5 seeds) | 9.46 ± 3.45 | 78.4 ± 51.6 | — |
| `--grad-clip-norm 1.0` (5) | 9.84 ± 3.52 | 92.6 ± 65.6 | 2/5 (inert) |
| `--grad-clip-norm 0.05` (5) | 13.29 ± 3.19 | 143.8 ± 55.7 | 1/5 |
| `--grad-clip-norm 0.10` (3) | 13.30 ± 5.19 | 155 ± 91.5 | 1/3 |

Inert where it doesn't engage, consistently harmful where it does (mean
CVaR₉₉ +40%, catastrophic paths +83% at 0.05). That is what this document's
own diagnosis predicts: GRU (WGAN-GP)'s failure is a hidden-state *recovery
lag*, which gradient magnitude has no bearing on. The fix was borrowed from
the MLP's sigmoid-saturation mechanism — a different failure that clipping
genuinely does fix, as α=0.995 above re-confirms. It also closes the untried
"lower peak LR / LR warmup for the recurrent weights" candidate by
implication: that was motivated by the same weight-growth story shown above
never to have been the mechanism.

**GRU (TimeGAN), `--moneyness-clip` — actively harmful.**

| Condition (5 seeds) | CVaR₉₅ | CVaR₉₉ | `<-50` |
|---|---|---|---|
| baseline | 4.16 ± 3.94 | 12.14 ± 15.7 | 150 ± 320 |
| `--moneyness-clip -0.15 0.10` | 6.67 ± 3.38 | 24.97 ± 12.3 | 411.8 ± 255 |

Improves 1/5 seeds. Mean CVaR₉₉ doubles; mean catastrophic paths nearly
triple. At seed 0 the clipped run is degenerate (delta mean 2.3e-06, caught
by the never-hedge signature above); seeds 1-4 remain healthy policies that
are simply much worse, so the collapse is seed-specific but the harm is
systematic. The one seed it helps (3) is the one where the baseline was
itself catastrophic. It behaves as a variance compressor toward a
mediocre-bad middle — its std is *lower* than baseline's while its mean is
twice as bad: **baseline reaches 0-3 catastrophic paths in 3 of 5 seeds; the
clipped version's best seed is 185.** The clip removes the good outcomes. On
the surviving generator the untreated baseline mean (CVaR₉₉ 12.14) is already
better than the documented post-fix figure (25.52).

**The common cause is what this document has been warning about throughout.**
Baseline GRU is so seed-sensitive — CVaR₉₉ spanning 2.78-39.67 (TimeGAN) and
5.37-13.11 (WGAN-GP) with no intervention at all — that between-seed spread
dwarfs any between-condition difference (σ 3.45 vs. a 0.38 condition gap for
WGAN-GP). A single-seed comparison on this architecture measures the seed.
That is exactly how both fixes came to be promoted.

### Basic RNN (TimeGAN): the open item, now substantially fixed

This document closed out Basic RNN (TimeGAN) as "genuinely open, with no
further untried candidate identified." One candidate was in fact untried.
`--moneyness-clip` had been ruled out — but against the *saturated*
checkpoint, where it changed nothing to four decimal places. That is the
expected result rather than evidence: a hidden state pinned at ±1.0
regardless of input cannot respond to its input being clipped. `--lr 1e-3`
was later shown to de-saturate the network, and the conclusion drawn then was
that the de-saturated RNN runs into mechanism (b) — which is what
`--moneyness-clip` targets. The stack was never run.

De-saturation first reproduced against the current generator (saturated
fraction 1.0000 → 0.24297, documented 0.2427; delta span 0.173 → 0.99999,
documented 1.000), so the clip is being tested on a network that can actually
respond to its input. Then five seeds, paired:

| Condition (5 seeds) | CVaR₉₅ | CVaR₉₉ | `worst_loss` | `<-50` | `<-10` |
|---|---|---|---|---|---|
| baseline (lr=1e-2) | 5.40 ± 0.86 | 20.65 ± 3.39 | -2,664 | 324.6 ± 63.9 | 2,310 |
| `--lr 1e-3` alone | 3.24 ± 1.61 | 10.48 ± 7.20 | -1,392 | 122.4 ± 118 | 1,022 |
| **`--lr 1e-3 --moneyness-clip -0.15 0.10`** | **1.62 ± 0.55** | **3.77 ± 1.90** | **-333** | **16.8 ± 20.5** | **185** |

**5/5 seeds improve** on CVaR₉₅, CVaR₉₉, `worst_loss`, and both path counts;
4/5 on kurtosis. Mean CVaR₉₉ down 82%, catastrophic paths down 95%. **Seeds 2
and 4 are fully clean** — 0/500,000 below -50, worst losses -19.7 and -11.0,
excess kurtosis 29.6 and 12.1 — the first clean checkpoints this architecture
has produced here.

The three-way ordering shows the clip does work beyond the learning rate:
20.65 → 10.48 → 3.77, with CVaR₉₉ std collapsing 7.20 → 1.90. `--lr 1e-3`
alone helps but erratically (4/5 seeds, huge variance); the clip improves both
mean and consistency. Note also that lr-alone's *tail-risk* effect is
seed-dependent — a no-op at seeds 0 and 1, a 90% CVaR₉₉ reduction at seed 2 —
so this document's earlier single-seed "de-saturates but doesn't fix tail
risk" conclusion sampled the no-op case. The de-saturation itself is
perfectly reliable; its downstream effect is not.

This is not the pattern that has fooled this project before: the lr+slow-ramp
stack was 2/5 seeds and a 3.8% mean improvement, against 5/5 and 82% here.

**Not a full close.** Three of five seeds still show 15-50 catastrophic paths
and elevated kurtosis; the outcome is bimodal (either fully clean or
substantially narrowed). The clip bound `(-0.15, 0.10)` was inherited from
GRU and never tuned for this architecture, and everything here is against the
single surviving generator. **Promoted** to
`checkpoints/hedging_agent_rnn_timegan.pt` at seed 0 (the repo's
production-checkpoint convention; the pre-fix checkpoint is preserved as
`.bak-pre-lr-clip-fix`). Seed 0 rather than one of the clean seeds
deliberately — promoting the best of five is the practice that produced the
two failed GRU fixes above.

### Where GRU's seed variance comes from

The open item above says GRU is "dominated by seed variance rather than by any
fix" and that what it needs is an explanation of the variance itself. This is
that explanation. Nothing here is a new fix attempt; it is forensics on the
checkpoints already trained, plus one training-free probe.

**Every seed fails on the same kind of path.** Pulling the catastrophic paths
(`< -50`) out of the shared 500,000-path stress set for each of the 5 GRU
(WGAN-GP) baseline seeds and characterising them by shape:

| seed | catastrophic | median dip (first 10 steps) | median final log-moneyness | matches down-then-rally |
|---|---|---|---|---|
| s0 | 67 | -0.61 | +4.31 | 67% |
| s1 | 22 | -1.04 | +4.18 | 86% |
| s2 | 142 | -0.59 | +4.39 | 72% |
| s3 | 40 | -0.69 | +4.04 | 68% |
| s4 | 121 | -0.47 | +4.46 | 54% |

Against a population rate of **0.53%** for the same down-then-rally signature
(dip < -0.4 and final > +2), so every seed's failures are 100-160x enriched
for it. This is the mechanism
[already diagnosed](#follow-up-diagnosis-gru-wgan-gp-is-a-gru-specific-hidden-state-recovery-lag-not-saturation)
on a single checkpoint; it is not seed-specific.

**But the failing paths are almost disjoint.** Pairwise Jaccard overlap
between seeds' catastrophic sets runs 0.07-0.22, and of a 265-path union
exactly **2 paths fail for all five seeds**. So this is not a fixed set of
paths that GRU cannot hedge — each seed fails on its own subset of a shared
at-risk population.

**What the seed changes is a severity level, not the shape.** Conditional
failure rate among at-risk paths (final > +2), binned by shock depth:

| dip depth | n | s0 | s1 | s2 | s3 | s4 |
|---|---|---|---|---|---|---|
| < -1.5 | 241 | 2.90% | 2.07% | 4.98% | 1.24% | 3.32% |
| -1.5 to -1.0 | 448 | 1.79% | 1.56% | 4.24% | 1.34% | 1.79% |
| -1.0 to -0.7 | 655 | 2.29% | 0.46% | 3.97% | 1.22% | 4.12% |
| -0.7 to -0.4 | 1,312 | 1.14% | 0.30% | 3.43% | 0.76% | 1.68% |
| -0.4 to -0.2 | 1,689 | 0.47% | 0.06% | 1.24% | 0.18% | 1.66% |
| -0.2 to 0 | 4,783 | 0.27% | 0.04% | 0.29% | 0.06% | 0.44% |

Monotonic in depth for every seed — same curve shape, different level, a ~5x
spread between s1 (0.72% over all at-risk paths) and s2 (3.84%). The shift is
not confined to the tail either: **median** wealth on at-risk paths runs -0.41
(s1) to -6.67 (s2), so the whole conditional distribution moves, and the
`<-50` count is just where it becomes visible.

That combination — a shared deterministic defect, a seed-dependent severity,
and an at-risk population of only ~0.5% of paths — is what makes the tail
metrics so noisy. CVaR₉₉ is reading a small, severity-sensitive subsample.
Between-condition effects have to clear that, and mostly cannot.

**The severity is measurable without training, in milliseconds.**
`src/backtester/recovery_probe.py` runs the down-then-rally probe from the
original diagnosis across a sweep of shock depths and reports the mean number
of steps delta spends below 0.5 after the shock. Within each GRU arm that
number ranks the seeds by their measured 500,000-path tail risk:

| arm | lag range (steps) | CVaR₉₉ range | Spearman(lag, CVaR₉₉) |
|---|---|---|---|
| GRU (TimeGAN) baseline | 0.1-18.1 | 2.78-39.67 | **+1.000** |
| GRU (TimeGAN) `--moneyness-clip` | 1.0-19.0 | 12.68-43.09 | +0.900 |
| GRU (WGAN-GP) `--grad-clip-norm 1.0` | 2.6-14.3 | 5.37-13.43 | +0.900 |
| GRU (WGAN-GP) baseline | 2.6-12.1 | 5.37-13.11 | +0.700 |

**This ranking claim is bounded by later out-of-sample data** — on 14
checkpoints trained afterwards it holds on one arm (+0.821) and collapses on
another (+0.254), because most of those checkpoints sit in the clean band
where lag has no dynamic range. What survives is *detection* of the collapse
mode rather than ranking; see [below](#why-a-seed-lands-at-a-given-severity-neither-initialization-nor-data-draw). The four arms below span
wide lag ranges, which is why the correlation looks uniformly strong here.

Deliberately *duration*, not final delta: the same checkpoints scored by
recovered delta in the realistic depth band give a weaker -0.768 (pooled),
because final delta saturates at 1.0 for most checkpoints. Duration is also
what the mechanism predicts should matter — the price path is exponential, so
the largest absolute increments land early in the rally, while delta is still
catching up.

**Controls.** Final training CVaR loss does *not* predict stress tail risk:
Spearman -0.600, -0.900, +0.100 across the three arms with training logs — the
wrong sign twice, inconsistent, and on GRU (TimeGAN) seed 3 (the 39.67-CVaR₉₉
near-collapse) its training loss is the second-*best* of its arm. Whatever the
seed is doing, it is invisible in-distribution.

Probe-construction sensitivity was checked by varying the rally target
(3.50/4.87/6.00) and the dip's end step (7/10/14): the GRU (TimeGAN) arm holds
at +1.000 under all four constructions, while the GRU (WGAN-GP) baseline arm
moves between +0.400 and +0.900. The reason is resolution — three of its five
seeds have lags within 2 steps of each other (7.4, 8.7, 9.2), and their
ordering is not stable. **The honest claim is that the probe separates seeds
whose lags differ by more than ~2 steps and does not resolve seeds inside
that band.**

**The predictor is specific to this defect, which is the right behaviour.**
Run across the LSTM and Basic RNN arms too, it carries no signal where the
defect is absent: pooled Spearman across the 20 non-GRU sweep checkpoints is
**+0.005**. Those arms have lag ranges of 0.2-2.2 steps (LSTM) and 0.1-0.7
(Basic RNN after `--lr 1e-3 --moneyness-clip`) — the lag is simply gone, and
their remaining tail-risk variation comes from something else. Basic RNN
*baseline*, which does carry a lag (0.0-10.5 steps), scores +0.800. So the
probe tracks recovery lag specifically, not tail risk in general.

**What this does and does not settle.** It explains the variance — a shared
mechanism at seed-dependent severity, amplified by a rare trigger — and it
makes severity cheap to measure, which turns "GRU is unpredictable" into "GRU
checkpoints are screenable before a 500,000-path evaluation". It does not
explain *why* a seed lands at a given severity — **answered separately
[below](#why-a-seed-lands-at-a-given-severity-neither-initialization-nor-data-draw): neither the initialization nor the training-data draw
determines it, so nothing here could have connected the initialisation to the
resulting lag.** It also does not license picking the best-probing seed and
promoting it — that is the practice that produced two retracted fixes in this
document.

Raw data: `sweep_data/PROBE_recovery_lag.json` (40 checkpoints, all four
architectures).


#### Why a seed lands at a given severity: neither initialization nor data draw

The section above left this open, and noted that nothing in it connected an
initialization to the resulting lag. It could not have: a single `--seed` sets
both the policy initialization and the training noise stream, so every run in
this project has confounded the two. `--data-seed` (added for this) re-seeds
immediately before the training loop, leaving `--seed` to govern
initialization and the premium estimate alone.

A 3x3 factorial on GRU (TimeGAN) — the arm with the widest severity range,
where seed 3 is the 39.67-CVaR₉₉ near-collapse (probe lag 18.1) and seeds 2
and 4 are the cleanest (0.1, 0.2):

| probe lag / CVaR₉₉ | data 2 | data 3 | data 4 |
|---|---|---|---|
| **init 2** | 0.3 / 3.20 | 0.2 / 2.79 | 0.8 / 2.92 |
| **init 3** | 0.1 / 1.68 | 0.1 / 3.95 | 0.4 / 2.89 |
| **init 4** | **12.3 / 36.93** | 0.1 / 3.49 | 0.2 / 4.09 |

**Neither factor is a main effect, and each is ruled out by a direct
contradiction rather than a weak correlation:**

- **Not the initialization.** Init 3 produced the arm's worst checkpoint in
  the original runs and is clean in all three cells here. Init 4 was clean
  originally (lag 0.2) and is the severe cell here.
- **Not the data draw.** Data seed 2 gives 0.3 with init 2, 0.1 with init 3,
  and 12.3 with init 4.

The severe cell is a genuine collapse, not a marginal case: CVaR₉₉ 36.93, 664
catastrophic paths, worst loss -6,202.2 — the
[degenerate never-hedge constant](#the-reproduction-anchor-numbers-identify-a-dead-policy-not-a-checkpoint),
and the same profile as the original seed 3 (39.67, 723, -6,202.3). Every
other cell lands in CVaR₉₉ 1.68-4.09.

**What one event in nine cells can and cannot support.** It cannot establish
an interaction *term* — that would need many more cells. What it establishes
is the negative: both main effects are dead, so there is no pre-training
property of the initialization, and no property of the data stream, that
predicts where a run will land. Severity is decided by the joint trajectory.
That is also why the correlational hunt that preceded this found no reliable
initialization-side predictor — the strongest candidate, layer-0's
candidate-gate spectral radius, correlated with lag at -0.90/-0.90/-1.00 in
three arms and +0.10 in a fourth, and its value at initialization spans only
0.578-0.634 across seeds before training moves it to 18-28.

**Base-rate control.** Six of the first seven factorial cells came back clean,
which under the original arm's rate (2 of 5 above one lag-step) is roughly a
1% run of luck — enough to suspect that `--data-seed` was suppressing the
effect it was built to measure, since it resets a stream the original path let
run on. Five fresh seeds (5-9) on the **original** code path settle it:

| seed | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|
| probe lag | 0.5 | **11.2** | 0.2 | 0.1 | 0.2 |
| CVaR₉₉ | 5.97 | 17.06 | 10.32 | 2.04 | 3.04 |

Two severe in ten original-path draws (seeds 0-9) against one in nine
instrumented cells — indistinguishable. The instrument is neutral and the
factorial's negative result stands. Note the suspicion was raised on
arithmetic and retired on data; the seventh cell (`i4_d2`, severe) had already
falsified it before this control ran.

**This bounds the previous section's probe claim.** That section reported
Spearman +0.70 to +1.00 between probe lag and CVaR₉₉ within every GRU arm.
On these 14 new checkpoints the same correlation is +0.821 on the control arm
but **+0.254** across the factorial — because eight of nine cells sit in the
clean band, where lag has no dynamic range to carry information. Seed 7 is the
sharpest case: lag 0.2, indistinguishable from clean seed 8's 0.1, yet 93
catastrophic paths against 8's zero.

What survives out-of-sample is **detection, not ranking**. Across all 14 new
checkpoints, lag > 5 gives CVaR₉₉ 17.1-36.9 (n=2) and lag ≤ 5 gives 1.7-10.3
(n=12), with no overlap. The probe is a cheap screen for the collapse mode; it
does not order the checkpoints that survive it. The original claim was
measured on four arms that happened to span wide lag ranges, and generalised
past what that supported.

**Practical consequence.** There is no way to pick a good GRU run in advance,
and no cheap post-training way to rank runs that have not collapsed. Multi-seed
evaluation is not optional for this architecture — which is what this document
has been finding empirically since the two retracted single-seed fixes, now
with a reason attached.

**Still open:** *when* in training a run's severity is decided, and what
distinguishes the trajectory that collapses. That is a training-dynamics
question — periodic checkpointing through a severe and a clean run, comparing
where they separate — and it is not attempted here.

Raw data: `sweep_data/{PROBE,RESULT}_seed_decomposition.json` (the factorial)
and `sweep_data/{PROBE,RESULT}_gru_tg_baserate.json` (the control).


#### When severity is decided: at a run-specific step, with no in-distribution signal

The decomposition above left this as the remaining question. Four runs
checkpointed every 1,000 of 25,000 steps answer it — two severe and two clean,
paired so that one pair (`i4_d2` vs `i4_d3`) shares an initialization and
differs only in the noise stream, and the other (`s6` vs `s8`) uses the
original single-seed code path as a check against instrument artifacts. All
four reproduce their known endpoints exactly, so the trajectories are the real
ones.

CVaR₉₉ against the 500,000-path stress set, every other checkpoint:

| step | 1k | 3k | 5k | 7k | 9k | 11k | 13k | 15k | 17k | 19k | 21k | 23k | 25k |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `i4_d2` **severe** | 2.96 | 4.63 | 4.95 | 4.62 | 5.75 | 3.17 | 5.33 | 3.87 | 4.27 | **11.89** | 27.59 | 30.49 | **36.93** |
| `s6` **severe** | 5.52 | 4.15 | 5.23 | **13.69** | 20.70 | 20.60 | 15.36 | 34.12 | 25.72 | 24.65 | 26.00 | 22.20 | 17.06 |
| `i4_d3` clean | 2.58 | 2.05 | 2.51 | 2.46 | 2.15 | 2.47 | 2.60 | 2.69 | 3.05 | 2.79 | 2.55 | 2.84 | 3.49 |
| `s8` clean | 3.05 | 2.26 | 2.29 | 3.22 | 2.61 | 2.43 | 2.68 | 2.95 | 2.69 | 2.26 | 2.25 | 2.33 | 2.04 |

**Both clean runs are flat for the entire schedule** — 2.02-3.53 and 2.04-3.22
across 25 checkpoints each, `s8` ending at its own best value with 0
catastrophic paths at 15 of its 25 checkpoints. Nothing accumulates in a run
that does not break.

**Both severe runs break, at unrelated steps and in unrelated ways.** `i4_d2`
is indistinguishable from a clean run through step 18,000 (4.03, 17
catastrophic paths) and then goes 11.89 → 22.33 → 27.59 over the next three
checkpoints, climbing monotonically to 36.93 at the end. `s6` starts
degrading at step 5,000 and rises through 7.08 → 13.69 → 20.95 by step 8,000,
then plateaus in the teens-to-thirties for the remaining 17,000 steps,
fluctuating without direction and ending at 17.06 — its best value after the
break.

So severity is decided at a **run-specific step**: between 18k and 19k in one
case, between 5k and 8k in the other, and never in the two clean runs. There
is no schedule position at which it happens.

**This kills the early-stopping rule the first pair appeared to support.**
Read alone, `i4_d2` vs `i4_d3` looks like a clean prescription: stop at 17,000
steps and the severe run scores 4.27 instead of 36.93 while the clean run
loses nothing (3.05 vs 3.49). `s6` refutes it — at step 17,000 it scores 25.72
against a final 17.06, so the same rule makes that run *worse*. The
prescription was an artifact of one pair having its transition late.

**The collapse is invisible in-distribution at every granularity checked.**
Across the steps where `i4_d2`'s stress CVaR₉₉ goes 4.27 → 11.89 → 36.93, its
training CVaR loss reads -0.0054 → -0.0064 → -0.0064: flat, and marginally
*better* at the end. The policy is not getting worse at what it is trained on.
This is the same finding as the earlier training-loss control, now at
1,000-step resolution within single runs rather than across checkpoints.

**The recovery-lag probe is not a monitor either, in either direction.** In
`i4_d2` it trails the damage — still reading 0.6 at step 19,000, where the
stress test already shows CVaR₉₉ 11.89 and 118 catastrophic paths. In `s6` it
runs ahead: 4.7 at step 1,000, elevated from the very first checkpoint, while
that run's tail risk is still a healthy 5.52 with 0 catastrophic paths and
stays healthy for another 4,000 steps. One run where the signal is late, one
where it is early by 6,000 steps, is not a monitor. (That `s6` was
distinguishable from the other three runs at step 1,000 by lag alone, and did
go on to collapse, is worth recording as an observation. It is one run, and
`i4_d2` collapsed with no such early elevation, so absence of the signal
clearly does not indicate safety.)

**What this settles.** Severity is not carried from initialization, not
determined by the data draw, not accumulated steadily, and not scheduled. It
is a run-specific transition that no in-distribution quantity announces and no
fixed intervention can preempt. Combined with the factorial above, that closes
off the class of cheap fixes — there is nothing to tune, no step to stop at,
and no pre-training property to select on. What remains available is exactly
what this document already practises: train several seeds, evaluate all of
them at 500,000 paths, and discard the ones that broke.

**Not attempted here:** what changes inside the network at the transition.
Both severe runs' checkpoints straddling their break points are on disk, so
the weight-space question — what moves between step 18,000 and 20,000 in
`i4_d2` — is set up but unanswered.

Raw data: `sweep_data/{PROBE,RESULT}_severity_trajectory.json`, 100
checkpoints across the four runs.


### Re-anchoring all four TimeGAN rows to the surviving generator

Every TimeGAN row in this document was measured against attempt 4's
generator, which was never preserved. Those rows cannot be re-derived — only
re-anchored to `checkpoints/timegan.pt`, the surviving retrain. That is what
this section does, at 5 seeds per row rather than the single seed the
originals used, since this same session retracted two promoted fixes that
turned out to be single-seed artifacts.

MLP and LSTM were trained here (15 runs: MLP ×5, `--slow-ramp-fraction 0.05`
LSTM ×5, and — to make the LSTM fix a paired comparison rather than a
before/after against a number from a different generator — untreated LSTM
×5). Basic RNN and GRU already had 5-seed sets from the sweeps above, run
against this same generator. All 20 checkpoints are evaluated on the one
500,000-path seed-42 regime-switching scenario used throughout this document.

Each architecture in its current production configuration:

| Architecture (production config) | CVaR₉₅ | CVaR₉₉ | worst path | `<-50` | `<-10` |
|---|---|---|---|---|---|
| MLP (default) | 5.15 ± 1.66 | 8.53 ± 3.09 | -46.3 | **0.0 ± 0.0** | 1,368 |
| Basic RNN (`--lr 1e-3 --moneyness-clip -0.15 0.10`) | 1.62 ± 0.55 | 3.77 ± 1.90 | -836.7 | 16.8 ± 20.5 | 185 |
| LSTM (`--slow-ramp-fraction 0.05`) | 1.74 ± 0.24 | 3.27 ± 0.51 | -799.1 | 0.2 ± 0.4 | 20 |
| GRU (default; fix retracted above) | 4.16 ± 3.94 | 12.14 ± 15.73 | -6,202 | 150.0 ± 320.5 | 1,341 |

`worst path` is the single worst loss anywhere in the arm, not a per-seed
mean; the `±` columns are cross-seed mean ± std.

**MLP: the "not clean" verdict was a single-seed draw at the bad end.** The
rebuild above reported 0.88% of paths below -10 against the `<0.1%`
known-good bound. Across 5 seeds that figure spans **0.012% to 0.877%** — a
72× range — and 0.877% is seed 0, the worst of the five. Seeds 2 and 3
(0.012%, 0.054%) sit inside the bound. **No seed produces a single
catastrophic path**: 0/500,000 below -50 at all five, worst loss anywhere
-46.3. This is the same single-seed error as the two retracted GRU fixes,
running in the pessimistic direction: the architecture is seed-dependent
around the bound, not broken. It remains true that the original row cannot be
reproduced — attempt 4's generator is gone — but "no longer clean" overstated
what one seed could support.

**LSTM: the fix holds, but its headline improvement does not reproduce,
because the failure it fixed doesn't occur on this generator.** Paired across
seeds:

| seed | ramp CVaR₉₉ | baseline CVaR₉₉ | ramp `<-50` | baseline `<-50` |
|---|---|---|---|---|
| 0 | 3.60 | 3.91 | 0 | 0 |
| 1 | 3.31 | 4.45 | 0 | 0 |
| 2 | 2.91 | 4.47 | 1 | 3 |
| 3 | 3.89 | 3.61 | 0 | 6 |
| 4 | 2.62 | 3.24 | 0 | 0 |
| **mean** | **3.27 ± 0.51** | **3.94 ± 0.53** | **0.2** | **1.8** |

`--slow-ramp-fraction 0.05` improves CVaR₉₉ on 4/5 seeds and cuts mean
catastrophic paths 1.8 → 0.2, so it is a real effect and stays promoted. But
this document records it as "CVaR₉₉ 42.13 → 3.24". The post-fix half
re-anchors almost exactly (3.27 ± 0.51 vs. 3.24). The pre-fix half does not
come close: **untreated LSTM's worst seed here is 4.47**, and no seed exceeds
it. The documented 42.13 — with excess kurtosis 81,035, both within a few
percent of the degenerate never-hedge constants tabulated
[above](#the-reproduction-anchor-numbers-identify-a-dead-policy-not-a-checkpoint) —
describes a collapsed policy that the surviving generator simply does not
produce. So the fix's measured value here is ~17% on CVaR₉₉, not ~92%. That
is a smaller claim than the table made, and it is the claim the current
generator supports.

**LSTM's remaining tail is one path, and it survives the fix.** Ramp seed 2
has a single -799 path out of 500,000. It barely moves CVaR₉₉ (2.91, the
second-best of the arm) while driving excess kurtosis to 321,011 — the
aggregate-moments-look-fine signature that motivated per-path counting in the
first place. Baseline seed 2 shows 3 such paths and baseline seed 3 shows 6,
so the fix reduces this without eliminating it. "Fixed, promoted" is right;
"clean" would not be.

Raw records: `sweep_data/RESULT_timegan_rows_5seed.json`. The harness that
produced them (`src/backtester/stress_eval.py`) is now committed — it had
previously existed only in a scratch directory outside the repo, which is the
same way attempt 4's generator was lost.


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
2. **~~Part I's training budget doesn't match the paper~~ — resolved, with
   a mixed-then-fixed result.** The paper's "50 epochs" is over a fixed
   500,000-scenario dataset at batch_size=1000, i.e. 25,000 gradient
   steps in this codebase's per-step convention — now the default. This
   is a genuine unit reconciliation, not just a bigger number: an earlier
   push to 500 steps (thought at the time to be "10x the paper's 50") was
   actually only ~2% of the paper's real budget. At the corrected scale,
   Black-Scholes matches the paper's absolute CVaR to 2-3% at every α
   (expected, it's analytic); LSTM/GRU are mixed (0.4% at α=0.75, up to
   34% at α=0.5, unchanged by anything below). ~~Basic RNN's mismatch got
   *worse*, not better (58-59% at α=0.5/0.99, up from 32% at the old, far
   smaller scale)~~ — **root-caused and fixed**: direct inspection of
   training dynamics found a clean weight-norm blowup specific to the
   vanilla RNN cell at the shared lr=1e-2 default (confirmed not fixed by
   `--grad-clip-norm` or `--orthogonal-init`, both of which converge to
   nearly the same pathological end state); `--rnn-lr 1e-3` eliminates it
   entirely and closes the gap to the paper to 0.1-2.3% at every α across
   4 seeds — tighter than LSTM/GRU's own match quality. See [the
   diagnosis](#basic-rnns-part-i-gap-root-caused-and-fixed-a-learning-rate-specific-weight-blowup)
   for the full story. Scaling up training exposed a real, fixable
   architecture-specific optimization issue that a smaller budget was too
   short to trigger — not evidence against scaling, in retrospect, but it
   did mean "just train longer" alone wasn't the fix.
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
     or both together, all tested at reduced scale. **No longer open**:
     `--lr 1e-3` de-saturates the hidden state (saturated fraction 1.0000 →
     0.243, delta span 0.173 → 1.000) and, stacked with `--moneyness-clip`,
     improves 5/5 seeds on every stress-test risk metric with 2/5 seeds
     fully clean — see [the rebuild
     section](#rebuilding-every-checkpoint-from-scratch-and-multi-seeding-the-fixes-that-shipped).
     Substantially fixed and promoted, though not a full close (3/5 seeds
     retain 15-50 catastrophic paths).
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
     input at the training boundary — works for GRU, not for LSTM.**
     Tried first as a free, no-retraining inference-only wrapper
     (worst_loss/below_-50/CVaR99 improve 11-26%, real-path collapse rate
     66.3% → 43.4%; LSTM ~1% either way, consistent with LSTM's failure
     being closer to deterministic/history-independent than GRU's). See the
     [fix-attempt
     writeup](#fix-attempt-clipping-the-rnns-log-moneyness-input-at-the-training-boundary--free-no-retraining-works-for-gru-does-not-work-for-lstm).
     **Then retrained from scratch with the clip active throughout
     training** (`RecurrentHedgingAgent.moneyness_clip`, `train_policy.py
     --moneyness-clip`) — beats both the pre-fix checkpoint and the
     inference-only wrapper on every metric this document treats as most
     decision-relevant (CVaR₉₅ 8.21 → **6.01**, CVaR₉₉ 31.98 → **25.52**,
     below_-50 578 → **402**, real-path collapse rate 66.3% → **31.5%**);
     worst_loss and mean wealth are both slightly worse, traced to a single
     rare sustained-rally path out of 500,000, the same qualitative failure
     at a higher threshold rather than a new one. **Promoted** — see the
     [training-from-scratch
     follow-up](#follow-up-training-with-the-clip-active-from-the-start-closes-more-of-the-gap-than-the-inference-only-wrapper-did)
     for the full numbers and the worst-path inspection.
     GRU is now a substantially narrower, better-characterized (and now
     partially fixed) problem than LSTM within mechanism (b). **LSTM's
     failure mechanism is better characterized than before, though still
     unfixed and not fully resolved**: a manual step-by-step unroll of the
     trained LSTM (verified to match `nn.LSTM`'s real output to 1e-6) shows
     only 1-2 of 64 hidden units are ever individually saturated — the
     earlier "not saturated" finding was real but incomplete — while a
     narrow band of log-moneyness (roughly 0.04 wide within a single ramp
     trajectory) separates confident hedging from collapse, a real, steep,
     learned transition rather than a numerical dead zone. Whether clipping
     could work in principle was tested carefully, including two rounds of
     self-correction after earlier probes overclaimed cleaner results than
     they actually supported (a soft-clip meant to rule out exact input
     repetition instead reproduced it via `tanh` saturation; a first
     hold-at-fixed-level probe confounded landing level with approach
     velocity). **Properly isolated** (fixed landing level at 0.09, only
     the number of ramp steps used to reach it varied): recovery is clean
     and monotone in approach speed alone, flipping between step sizes
     0.0129 (recovers within the 40-step hold) and 0.0180 (never does) —
     a genuine velocity-triggered hysteresis, not a level threshold and not
     input repetition. This narrow recovery basin isn't one real market
     paths reliably land in (collapse rate stays 96-97% across every clip
     variant tried, since clipping controls level, not velocity), so it
     explains the mechanism without offering a usable fix. See the [full
     diagnosis](#follow-up-lstm-timegans-failure-is-a-narrow-trajectory-dependent-transition-not-simple-saturation)
     for the corrected numbers and the two rounds of self-correction.
     Clipping doesn't touch it (it controls level, not velocity). **Since
     resolved and promoted**: a later, much larger investigation (a
     from-scratch paper-scale reproduction confirming this exact
     recovers-when-slow/stuck-when-fast signature independently, a
     Lipschitz-style smoothness penalty tried and decisively ruled out —
     eliminates the symptom but makes real stress-test tail risk 2-3
     orders of magnitude worse — and `--slow-ramp-fraction 0.05`, training-
     time exposure to synthetic slow ramps through the critical zone,
     validated at 5 seeds and a dose sweep) found and promoted a genuine
     fix: CVaR₉₉ 42.13 → 3.24, excess kurtosis 81,035 → 24.5 against a
     freshly-trained but equivalent-methodology TimeGAN generator (the
     original checkpoint wasn't preserved, so this isn't a strict
     same-checkpoint comparison — see the caveat where the numbers are
     reported). See the [attempt-4 promoted-checkpoint
     update](#attempt-4-paper-scale-batch178-10000-iterations-and-the-papers-own-temporal-traintest-split)
     above and the [full fix-attempt
     writeup](#fix-attempt-continued-a-paper-scale-reproduction-that-actually-shows-the-bug-and-a-smoothness-penalty-candidate-that-trades-the-symptom-for-a-worse-disease)
     for the complete story.
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
     → -137.5) and was promoted — **but that result was a single seed and
     has since been retracted.** At 5 paired seeds the flag improves 2/5 and
     leaves the mean slightly worse, and at one seed it produces
     bit-identical weights because the threshold sits ~45x above the median
     pre-clip gradient norm: it is a rare-event trigger, not a regularizer,
     and functionally acts as a seed perturbation. It also explains why this
     document's weight-growth hypothesis appeared falsified — clipping
     barely fires, so of course final weight norms match. The production
     checkpoint is now a plain default run. The underlying recovery-lag
     diagnosis below still stands; what does not stand is that anything here
     fixed it. See [the full
     diagnosis](#follow-up-diagnosis-gru-wgan-gp-is-a-gru-specific-hidden-state-recovery-lag-not-saturation),
     [the fix attempt](#fix-attempt-grad_clip_norm-substantially-improves-gru-wgan-gp-does-not-fully-close-it),
     and [the multi-seed
     retraction](#rebuilding-every-checkpoint-from-scratch-and-multi-seeding-the-fixes-that-shipped).
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
   entirely (see attempt 4's writeup above for the full argument).
   **Since root-caused, not just flagged as puzzling**: a later
   investigation ([full writeup](#investigating-why-the-best-fidelity-generator-produced-the-worst-policies-validatepy-checks-the-wrong-invariant))
   found the diversity ratio (and every other statistic `validate.py`
   checks) is a *terminal*-distribution-only measurement, structurally
   blind to path-level dynamics. Direct measurement found this session's
   own fidelity-checker-"OK" TimeGAN generator produces per-step
   volatility 2x real markets' with 7.7x more frequent large single-step
   moves and much stronger momentum/clustering — none of which shows up in
   the terminal (31-day cumulative) statistics the checker inspects,
   because those compound back toward a realistic endpoint anyway.
   Diversity-tuning was never going to fix downstream policy behavior on
   its own, regardless of how close to 100% it gets, because it optimizes
   an invariant recurrent policies aren't primarily sensitive to (the
   endpoint) rather than the one they are (the path). **Since implemented,
   not left as a proposal**: `validate.py::validate_generator_fidelity`
   now runs three path-dynamics checks (per-step volatility ratio, signed
   and \|return\| lag-1 autocorrelation) alongside the four
   terminal-distribution ones, with their own tests, and end-to-end
   verified to correctly flag the exact checkpoint that motivated this
   investigation (previously "OK", now `WARNING`). This class of failure
   can no longer pass silently for any future TimeGAN (or WGAN-GP)
   calibration attempt. **And since resolved, not just detected**: a
   dedicated path-dynamics-matching loss (`--lambda-dynamics`, mirroring
   how `--lambda-moment`/`--lambda-diversity` target the terminal
   statistics) produces a checkpoint passing all 7 checks simultaneously
   at paper scale on the first attempt, no hyperparameter search needed —
   see the [full writeup](#follow-up-a-dedicated-path-dynamics-loss-term-passes-all-7-checks--the-first-timegan-checkpoint-in-this-projects-history-to-do-so).
   Not yet validated downstream (no policy has been retrained against this
   checkpoint and stress-tested), so "diversity-tuning was never going to
   fix downstream policy behavior on its own" above should now be read as
   "terminal-only diversity-tuning" — a generator that's also
   path-dynamics-realistic is an open, promising, untested question.
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
- ~~Investigate Basic RNN's CVaR gap vs. the paper's own RNN figure at Part
  I~~ — **done, and it's a genuine systematic gap, not seed-sensitivity.** A
  4-seed sweep (seeds 0-3, full 25,000-step scale, seed=0 rerun through the
  same single-architecture code path as 1-3 to avoid an apples-to-oranges
  test-set draw) found every seed's RNN/Black-Scholes CVaR ratio well above
  the paper's own ratio at every α, with the gap monotonically shrinking as
  α rises (not U-shaped — an earlier draft of this bullet claimed α=0.75
  showed the smallest gap in raw-percentage terms and speculated about a
  tails-of-the-distribution effect; restated in ratio terms, which is the
  correct normalization since the paper's own ratio itself varies by α,
  there's no U-shape and that speculation doesn't hold up, so it's retracted
  here). See [above](#part-i-frictionless-replication) for the full
  per-seed ratio table and the noise-floor control (Black-Scholes CVaR
  varies <1% across seeds/runs at a given α; Basic RNN's varies 12-18%,
  ruling out test-set sampling noise as the explanation). ~~What's still
  open: *why*~~ — **also done**: direct inspection of training dynamics
  (loss/grad-norm/weight-norm/saturation logged every 500 steps) found a
  clean weight-norm blowup specific to the vanilla RNN cell at the shared
  lr=1e-2 default, confirmed not fixed by `--grad-clip-norm` or
  `--orthogonal-init` (both converge to nearly the same pathological end
  state); `--rnn-lr 1e-3` eliminates it and closes the gap to 0.1-2.3% at
  every α across 4 seeds, now the default for `architecture="rnn"` in
  `replicate_part1.py`. See the [full
  diagnosis](#basic-rnns-part-i-gap-root-caused-and-fixed-a-learning-rate-specific-weight-blowup).
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
- ~~Apply the same 8-seed-sweep methodology used to catch Basic RNN's
  seed-sensitivity to every other single-seed claim in this document~~ —
  **done for the two that mattered most, and both failed.** The two
  single-seed fixes still shipping in the headline tables — GRU (WGAN-GP)'s
  `--grad-clip-norm 1.0` and GRU (TimeGAN)'s `--moneyness-clip` — were rerun
  at 5 paired seeds each. The first is inert (2/5 seeds, bit-identical
  weights at a third because the threshold sits ~45x above the median
  gradient norm); the second is actively harmful (1/5 seeds, mean CVaR₉₉ 2x
  worse). Both are retracted; see [the rebuild
  section](#rebuilding-every-checkpoint-from-scratch-and-multi-seeding-the-fixes-that-shipped).
  That makes it five single-seed conclusions overturned in this project, not
  three. The remaining unvalidated one is the `--lambda-dynamics` negative
  result below.
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
    **Then retrained from scratch with the clip active throughout
    training** (not just wrapped around inference) — closes more of the gap
    than the wrapper did (CVaR₉₅ 8.21 → 6.01, CVaR₉₉ 31.98 → 25.52,
    below_-50 578 → 402, real-path collapse rate 66.3% → 31.5%) and is now
    **promoted** to `checkpoints/hedging_agent_gru_timegan.pt`; see the
    [training-from-scratch
    follow-up](#follow-up-training-with-the-clip-active-from-the-start-closes-more-of-the-gap-than-the-inference-only-wrapper-did).
    Still not a full close: worst_loss is slightly worse than pre-fix
    (-6033.3 → -6199.8), traced to a single rare sustained-rally path out of
    500,000 where delta collapses around log-moneyness 1.0 and never
    recovers through +8.34 — the same qualitative failure mode at a higher
    threshold, not a new one, but confirmation this fix narrows mechanism
    (b) rather than eliminating it for GRU. Remaining unexplored ideas: a
    systematic clip-bound sweep (only `(-0.15, 0.10)` has been tried);
    multi-seed validation (single seed throughout, same caveat as
    everywhere else in this document). **LSTM's failure is better
    characterized than before, though not fully resolved** (a manual
    gate-level unroll found only 1-2 of 64 hidden units are individually
    saturated, not the broad saturation an aggregate check would suggest;
    a velocity-isolated hold-at-fixed-level probe — fixing the landing
    level and varying only the number of steps used to reach it — found a
    clean, monotone result: recovery depends on how fast log-moneyness
    approaches the transition, not the level it reaches or whether the
    input value repeats, flipping cleanly between approach step sizes
    0.0129 (recovers) and 0.0180 (doesn't); see the [full
    diagnosis](#follow-up-lstm-timegans-failure-is-a-narrow-trajectory-dependent-transition-not-simple-saturation)
    including two corrections made to earlier drafts of this finding after
    confounded probes overclaimed cleaner results than the evidence
    supported), but still not fixed — a
    steepness/Lipschitz penalty on the transition during training, or
    training-time exposure to paths that cross this specific boundary
    slowly and repeatedly, are two candidates this characterization
    suggests, both untried, and a fuller characterization of the
    velocity/level/duration interaction remains open too. An
    adversarial/extreme-scenario
    data augmentation step targeted at GRU's remaining single-path failure
    (rather than the input transform) remains untried. **Basic RNN (TimeGAN)
    specifically is a narrower, better-characterized sub-problem**: its
    vanilla RNN's hidden state is saturated at tanh's ±1.0 boundary
    regardless of input — confirmed via direct inspection, and confirmed
    *not* fixed by `grad_clip_norm`, `orthogonal_init` (now both wired to
    the CLI), or the combination — **and, checked once `moneyness_clip`
    existed as an option, not fixed by that either (identical numbers to
    the 4th decimal place, worst_loss/below_-50/CVaR95/CVaR99 all exactly
    unchanged)**, as expected once the hidden state was directly confirmed
    saturated (100% of units at `|h|>0.999`) even at log-moneyness exactly
    0 — a completely in-distribution input clipping has no reason to touch,
    since the problem was never about extreme inputs in the first place.
    The final checkpoint's recurrent weight
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
- ~~Investigate the alpha-sweep's puzzling α=0.995 dip~~ — **done: severity
  trends with α, but the clean/elevated split at a given α is seed-driven,
  not α-driven — the single-seed table's ordering doesn't generalize.** The
  single-seed table's picture (α=0.99 elevated, α=0.995 clean, α=0.997
  catastrophic) used seed 0 for every α; a 4-seed sweep (seeds 0-3,
  unclipped MLP/WGAN-GP, otherwise identical settings, stress-tested the
  same way as every other tail-risk scan in this document — 500,000
  regime-switching paths, seed=42, `<-50`/`<-10` path counts) shows that
  ordering doesn't hold seed-to-seed:

  | α | seed 0 | seed 1 | seed 2 | seed 3 |
  |---|---|---|---|---|
  | 0.99 | worst −49.1, 5275 `<-10` | worst −42.5, 2919 `<-10` | worst −29.3, 639 `<-10` | worst −10.2, 1 `<-10` (clean) |
  | 0.995 | worst −10.2, 1 `<-10` (clean) | worst −48.9, **5544** `<-10` | worst −10.1, 1 `<-10` (clean) | worst −9.7, 0 `<-10` (clean) |
  | 0.997 | worst **−6202**, **814** `<-50` (catastrophic) | worst −49.2, 2935 `<-10` | worst −59.1, 4 `<-50` | worst −34.1, 2709 `<-10` |

  **Reproduction anchor**: α=0.997 seed 0 in this sweep is bit-for-bit
  identical to the previously-documented, since-fixed pre-`grad_clip_norm`
  checkpoint (`hedging_agent_mlp_alpha0_997.pt.bak-pre-gradclip-fix`) —
  same worst_loss (−6202.48), same 814-path `<-50` count, and the same
  CVaR₉₅/CVaR₉₉/skew/kurtosis (11.76 / 43.15 / −248.9 / 80,781) already in
  this document — confirming this sweep's harness and the original scan
  agree exactly, not just approximately. α=0.99 seed 0 and α=0.995 seed 0
  likewise reproduce the original single-seed "elevated" vs. "clean"
  finding, so the other seeds' disagreement is informative, not a
  methodology artifact.

  **The actual pattern**: `below_-50 > 0` (catastrophic) appears only at
  α=0.997 — but even there the two occurrences aren't the same severity.
  Seed 0's checkpoint is, by the reproduction anchor above, bit-for-bit
  identical to the pre-`grad_clip_norm` checkpoint whose −250-range
  pre-activation logits and exact sigmoid-underflow were diagnosed
  directly in [mechanism (a)](#mechanism-a-root-caused-and-fixed-sigmoid-output-saturation-not-sparse-gradients)
  above — that provenance, not a new probe, is what grounds "dead policy"
  for seed 0. Seed 2 is only reported as *marginal* on path counts (worst
  −59.1, 4 paths past the −50 line, vs. seed 0's −6202/814): no logit
  inspection was run on seed 2's own checkpoint, so this scan doesn't
  establish whether it shares seed 0's exact saturation mechanism or is a
  different, less severe failure — only that it's the smaller of the two
  catastrophic outcomes by an order of magnitude. Even reading it
  conservatively as 1-of-4 clearly catastrophic rather than 2-of-4,
  catastrophic outcomes are exclusive to α=0.997, consistent with CVaR's
  `1/(1-α)` loss amplification making the underlying sparse-gradient
  instability worse as α climbs. But **the clean/elevated split within a
  single α is seed-driven, not α-driven**: α=0.995 seed 1 (5,544 paths
  `<-10`) is worse than three of the four α=0.99 seeds and worse than two
  of the four α=0.997 seeds; α=0.99 seed 3 (1 path `<-10`) is exactly as
  clean as the cleanest α=0.995 seeds. The original table's "α=0.995
  trained cleanly" was true of seed 0 specifically, not of α=0.995 as a
  risk level — a different seed at the same α would have looked exactly as
  elevated as α=0.99.

  **This has a live production consequence.** α=0.99 and α=0.997's
  checkpoints were already retrained with `grad_clip_norm=1.0` and
  promoted (mechanism (a) above); α=0.995's was not —
  `hedging_agent_mlp_alpha0_995.pt` is still the original unclipped seed-0
  run, kept on the strength of a single training run this sweep now shows
  isn't representative (its own seed 1 would have shipped a checkpoint with
  5,544 stress-test paths losing more than 10x the premium). It hasn't been
  retrained here — this sweep's checkpoints are diagnostic scratch runs,
  not a promotion — but the case for giving α=0.995 the same
  `grad_clip_norm=1.0` treatment as its two neighbors is now direct
  evidence, not speculation, and should be the first thing whoever revisits
  this does.
- ~~Attempt LSTM (TimeGAN)'s velocity-hysteresis fix~~ — **resolved with a
  genuine, paper-scale-validated improvement: `--slow-ramp-fraction 0.05`.**
  See the
  [full writeup](#fix-attempt-continued-a-paper-scale-reproduction-that-actually-shows-the-bug-and-a-smoothness-penalty-candidate-that-trades-the-symptom-for-a-worse-disease)
  and its [follow-up](#fix-attempt-third-try-slow-ramp-fraction-at-a-lower-dose-finally-validated-against-the-real-baseline--a-genuine-improvement)
  for the complete story, including an hours-long detour caused by an
  operator error (a training command silently defaulted to synthetic
  placeholder data instead of real `^GSPC` data, `--data-source yfinance`
  omitted) that produced three misleadingly-different-looking "failures"
  before the actual bug was reproduced. Once corrected, a from-scratch
  paper-scale TimeGAN + LSTM policy pair reproduced the documented
  recovers-when-slow/stuck-when-fast signature cleanly (end-state delta
  0.997 at a 30-step ramp vs. 0.0003 at a 1-step jump, same landing level)
  — the first independent reproduction of mechanism (b) this project has
  achieved from scratch, and the first real baseline any fix candidate has
  been tested against. Two candidates were then tested against it:
  `--smoothness-penalty-weight` (a global Lipschitz-style penalty on
  `d(delta)/d(log-moneyness)`) eliminates the velocity-hysteresis symptom
  completely at weight=0.01 (flat ≈0.20 end-state at every ramp speed) but
  makes the real stress test dramatically worse (CVaR₉₉ 4.42 → 9.91,
  kurtosis 37.6 → 13,427). **A 10x-lower weight (0.001) doesn't fix this —
  if anything it's worse**: still a mostly-flattened velocity response
  (0.26-0.37 across every ramp speed, vs. weight=0.01's tighter 0.19-0.21
  band), and the stress test is *more* catastrophic, not less (CVaR₉₉
  15.83 vs. weight=0.01's 9.91, kurtosis 13,172 — the same order of
  magnitude, not an improvement). Two weights spanning a 10x range both
  producing severely elevated tail risk suggests this isn't a tuning
  problem — something about this specific penalty formulation (autograd
  through the whole training batch each step, `create_graph=True`, added
  directly into the CVaR loss) destabilizes training in a way that doesn't
  scale down smoothly with the weight; a per-step accounting bug or an
  interaction with CVaR's own sparse-gradient dynamics is more likely than
  "the network is being too smooth." **Not promoted, and not recommended
  as a direction to keep pursuing without first understanding why weight
  doesn't behave monotonically here.**
  **`--slow-ramp-fraction 0.05`** (3x lower dose than the destabilizing 0.15
  tried in the first, never-properly-validated attempt) is the one that
  worked: every stress-test risk metric improves over the baseline at
  paper scale (500,000 paths) — CVaR₉₅ down 20% (2.20 → 1.76), CVaR₉₉ down
  23% (4.23 → 3.27), excess kurtosis down 37% (44.6 → 28.2), skew closer to
  zero (-3.36 → -2.36) — confirmed at both a 100,000- and 500,000-path
  batch. Oddly, the narrow velocity-ramp probe used throughout this
  document to diagnose the bug reads this checkpoint's fix as *inverted*
  rather than clean (slow ramps now stuck, fast ramps now recovering) —
  the stress test is what actually settles it, a methodological lesson
  paired with the smoothness-penalty result: neither instrument alone
  tells the whole story.
  **Multi-seed check (5 seeds, MPS-accelerated) confirms and strengthens
  the finding**: CVaR₉₉ improves in 4/5 seeds (mean 21.6% reduction); more
  importantly, one baseline seed shows a kurtosis of 179,438 — the same
  order of magnitude as the original paper-scale catastrophe this whole
  investigation started from — while the matching slow-ramp checkpoint at
  that seed shows 38.4, a >4,600x reduction. Across all 5 seeds, mean
  kurtosis (including each condition's own worst outlier) is 189x better
  under slow-ramp (35,939 → 190). This isn't a single lucky seed 0 draw —
  it's suppressing baseline's occasional catastrophic-outlier failure mode,
  the specific pathology this document's "Catastrophic tail risk" section
  names as the headline paper-scale finding. See the
  [full multi-seed writeup](#follow-up-multi-seed-check-confirms-the-fix-and-reveals-its-suppressing-something-worse-than-the-single-seed-table-showed).
  **Dose sweep run** (0.02/0.10/0.15/0.20, single seed): non-monotonic —
  0.10 looked *better* than 0.05 at seed 0 (CVaR₉₉ 2.75, kurtosis 9.7), 0.15
  catastrophically destabilized (CVaR₉₉ 21.94, kurtosis 26,181, consistent
  with the first attempt's destabilization at the same dose on a different
  baseline), 0.20 still underperformed baseline. **Multi-seeding dose 0.10
  reversed its apparent single-seed win**: across 5 seeds its mean CVaR₉₉
  (8.08) is worse than baseline (4.08), with 2/5 seeds landing on severe
  outliers vs. dose 0.05's 1/5 — a direct demonstration of the exact trap
  the dose-sweep section warns about. **`--slow-ramp-fraction 0.05` is the
  dose to promote** — the only one validated at 5 seeds with a consistent
  improvement, not a favorable single draw.
