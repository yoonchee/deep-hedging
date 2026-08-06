# Mathematical Specification

## 1. European Call Option Payoff

Payoff at maturity $T$ for strike $K$:

$$
\text{Payoff}(S_T) = \max(S_T - K, 0)
$$

## 2. Transaction Cost Model

Proportional transaction cost incurred when rebalancing the hedge position at time $t$:

$$
\text{Cost}_t = \kappa \, |\delta_t - \delta_{t-1}| \, S_t
$$

where $\kappa$ is the proportional fee rate, $\delta_t$ is the hedge ratio (delta) at time $t$, and $S_t$ is the underlying asset price at time $t$.

## 3. Convex Risk Measure: Expected Shortfall (CVaR)

CVaR at confidence level $\alpha \in (0, 1)$, via the Rockafellar–Uryasev representation:

$$
\text{CVaR}_\alpha(X) = \inf_{h \in \mathbb{R}} \left\{ h + \frac{1}{1 - \alpha} \, \mathbb{E}\big[ \max(-X - h, 0) \big] \right\}
$$

where $X$ is the P&L (or terminal wealth) random variable and $h$ is an auxiliary variable optimized jointly with the hedging strategy.

## 4. Time-Series GAN Loss (WGAN-GP)

Critic (discriminator) loss with gradient penalty:

$$
\mathcal{L}_D = \mathbb{E}[D(x_{\text{real}})] - \mathbb{E}[D(x_{\text{fake}})] - \lambda \, \mathbb{E}\Big[ \big( \lVert \nabla_{\hat{x}} D(\hat{x}) \rVert_2 - 1 \big)^2 \Big]
$$

where $\hat{x}$ is sampled along straight lines between real and generated (fake) samples, and $\lambda$ is the gradient penalty coefficient.

### 4.1 Moment-matching penalty

Added to the generator loss to correct a tail-shape blind spot the adversarial loss alone doesn't penalize (matching the first two moments is enough to fool the critic):

$$
\mathcal{L}_{\text{moment}} = \big( \text{skew}(x_{\text{fake}}) - \text{skew}(x_{\text{real}}) \big)^2 + \big( \text{kurt}(x_{\text{fake}}) - \text{kurt}(x_{\text{real}}) \big)^2
$$

computed on terminal log-returns $\log(S_T / S_0)$, with $\text{skew}$/$\text{kurt}$ (excess) the standard third/fourth standardized moments. $\text{skew}(x_{\text{real}})$/$\text{kurt}(x_{\text{real}})$ are fixed targets estimated once from a large real-data sample, not recomputed per minibatch (too few samples to estimate reliably). Used by both the WGAN-GP generator (`generator/train_gan.py`) and TimeGAN (section 5 below).

## 5. TimeGAN Losses

Yoon et al. (2019)'s architecture (`generator/timegan.py`): Embedder $E$, Recovery $R$, Generator $G$, Supervisor $S$, Discriminator $D$, trained in three phases (`generator/train_timegan.py`). This repo's $D$ uses the WGAN-GP loss above (section 4) applied to latent codes, rather than the original paper's binary cross-entropy — a deliberate deviation for consistency with this project's existing WGAN-GP machinery and its demonstrated training stability (see RESULTS.md).

**Phase 1 — reconstruction loss** (pretrains $E, R$):

$$
\mathcal{L}_{\text{recon}} = \mathbb{E}\big[ \lVert x - R(E(x)) \rVert_2^2 \big]
$$

**Phase 2 — supervised loss** (pretrains $S$ on real dynamics only):

$$
\mathcal{L}_{S} = \mathbb{E}\big[ \lVert h_{t+1} - S(h_{1:t}) \rVert_2^2 \big], \quad h = E(x)
$$

**Phase 3 — joint adversarial training**, alternating:

$$
\mathcal{L}_D = \mathbb{E}[D(h_{\text{real}})] - \mathbb{E}[D(\hat h)] - \lambda \, \mathbb{E}\Big[ \big( \lVert \nabla_{\hat h} D(\hat h) \rVert_2 - 1 \big)^2 \Big], \quad \hat h = S(G(z))
$$

$$
\mathcal{L}_{G,S} = -\mathbb{E}[D(\hat h)] + \eta \, \mathcal{L}_{S,\text{fake}} + \mu \, \mathcal{L}_{\text{moment}}
$$

where $\mathcal{L}_{S,\text{fake}}$ is the section-4-style supervised loss applied to $G$'s own output (keeping $\hat h$'s stepwise dynamics realistic as $G$ updates), $\mathcal{L}_{\text{moment}}$ (section 4.1) is applied to the price channel of $R(\hat h)$, and $\eta$/$\mu$ are the `--lambda-supervised`/`--lambda-moment` weights. $E, R$ continue training on $\mathcal{L}_{\text{recon}}$ throughout phase 3 with a small weight, so the embedding doesn't drift while $G$/$D$ train.
