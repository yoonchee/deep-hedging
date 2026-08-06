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
