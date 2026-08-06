"""Trains the Deep Hedging policy (src/policy/hedging_agent.py) to minimize
the CVaR (src/loss/cvar.py) of terminal wealth over synthetic asset price
paths produced by the Market Generator (src/generator/market_gan.py).
"""

from typing import Annotated, Dict, Optional

import torch

from environment.market_env import MarketEnvironment
from generator.market_gan import Generator
from loss.cvar import CVaRLoss
from policy.hedging_agent import HedgingAgent


class PolicyTrainer:
    """Single training-step encapsulation for the CVaR-hedging policy."""

    def __init__(
        self,
        policy: HedgingAgent,
        environment: MarketEnvironment,
        generator: Generator,
        cvar_loss: CVaRLoss,
        implied_vol: Annotated[float, "implied volatility fed into the policy state"] = 0.2,
        lr: Annotated[float, "Adam learning rate for policy params and CVaR threshold h"] = 1e-3,
        device: Optional[torch.device] = None,
    ) -> None:
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy = policy.to(self.device)
        self.generator = generator.to(self.device)
        self.environment = environment
        self.cvar_loss = cvar_loss.to(self.device)
        self.implied_vol = implied_vol

        params = list(self.policy.parameters()) + list(self.cvar_loss.parameters())
        self.optimizer = torch.optim.Adam(params, lr=lr)

    def train_step(
        self,
        batch_size: Annotated[int, "number of synthetic paths per step"],
        seq_len: Annotated[int, "number of price observations per path"],
    ) -> Annotated[Dict[str, float], "{'loss': CVaR loss, 'mean_wealth': mean terminal wealth}"]:
        # The generator acts as a fixed, pretrained market simulator here: no
        # gradient is needed through its parameters when training the policy.
        with torch.no_grad():
            z = self.generator.sample_noise(batch_size, seq_len, device=self.device)
            prices = self.generator(z)  # [Batch, seq_len, 1]

        wealth = self.environment.simulate(self.policy, prices, self.implied_vol)  # [Batch]
        loss = self.cvar_loss(wealth)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {"loss": loss.item(), "mean_wealth": wealth.mean().item()}
