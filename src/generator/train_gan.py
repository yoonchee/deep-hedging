"""WGAN-GP training step for the Market Generator.

Implements the critic loss from math_spec.md, section 4:

    L_D = E[D(real)] - E[D(fake)] - lambda * E[(||grad D(interp)|| - 1)^2]

The discriminator is trained to maximize L_D, i.e. minimize -L_D. The
generator is trained with the standard WGAN objective to maximize E[D(fake)].
"""

from typing import Annotated, Optional

import torch

from generator.market_gan import Discriminator, Generator


def gradient_penalty(
    discriminator: Discriminator,
    real: Annotated[torch.Tensor, "[Batch, Time_Steps, 1] real price paths"],
    fake: Annotated[torch.Tensor, "[Batch, Time_Steps, 1] generated price paths"],
    device: torch.device,
) -> Annotated[torch.Tensor, "scalar gradient penalty E[(||grad D(interp)|| - 1)^2]"]:
    batch_size = real.size(0)

    # [Batch, 1, 1] -> [Batch, Time_Steps, 1] (broadcast interpolation weight)
    epsilon = torch.rand(batch_size, 1, 1, device=device).expand_as(real)

    # [Batch, Time_Steps, 1] (x_hat sampled along straight lines real -> fake)
    interpolates = (epsilon * real + (1 - epsilon) * fake).requires_grad_(True)

    # [Batch, Time_Steps, 1] -> [Batch, 1]
    d_interpolates = discriminator(interpolates)

    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=torch.ones_like(d_interpolates),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]

    # [Batch, Time_Steps, 1] -> [Batch, Time_Steps] -> [Batch]
    grad_norm = gradients.reshape(batch_size, -1).norm(2, dim=1)

    # [Batch] -> scalar
    penalty = ((grad_norm - 1) ** 2).mean()
    return penalty


class WGANGPTrainer:
    """Encapsulates a single WGAN-GP training iteration for the Market GAN."""

    def __init__(
        self,
        generator: Generator,
        discriminator: Discriminator,
        lr: Annotated[float, "learning rate for both Adam optimizers"] = 1e-4,
        betas: Annotated[tuple, "Adam beta coefficients"] = (0.5, 0.9),
        lambda_gp: Annotated[float, "gradient penalty coefficient (lambda)"] = 10.0,
        n_critic: Annotated[int, "number of critic updates per generator update"] = 5,
        device: Optional[torch.device] = None,
    ) -> None:
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.generator = generator.to(self.device)
        self.discriminator = discriminator.to(self.device)
        self.lambda_gp = lambda_gp
        self.n_critic = n_critic

        self.optimizer_g = torch.optim.Adam(self.generator.parameters(), lr=lr, betas=betas)
        self.optimizer_d = torch.optim.Adam(self.discriminator.parameters(), lr=lr, betas=betas)

    def train_discriminator_step(
        self, real: Annotated[torch.Tensor, "[Batch, Time_Steps, 1] real price paths"]
    ) -> Annotated[float, "critic loss value for this step (-L_D)"]:
        real = real.to(self.device)
        batch_size, seq_len, _ = real.shape

        z = self.generator.sample_noise(batch_size, seq_len, device=self.device)
        fake = self.generator(z).detach()

        d_real = self.discriminator(real)
        d_fake = self.discriminator(fake)
        gp = gradient_penalty(self.discriminator, real, fake, self.device)

        # Minimize -L_D = E[D(fake)] - E[D(real)] + lambda * gradient_penalty
        loss_d = d_fake.mean() - d_real.mean() + self.lambda_gp * gp

        self.optimizer_d.zero_grad()
        loss_d.backward()
        self.optimizer_d.step()
        return loss_d.item()

    def train_generator_step(
        self,
        batch_size: Annotated[int, "number of paths to sample"],
        seq_len: Annotated[int, "number of time steps per path"],
    ) -> Annotated[float, "generator loss value for this step"]:
        z = self.generator.sample_noise(batch_size, seq_len, device=self.device)
        fake = self.generator(z)

        # Maximize E[D(fake)] <=> minimize -E[D(fake)]
        loss_g = -self.discriminator(fake).mean()

        self.optimizer_g.zero_grad()
        loss_g.backward()
        self.optimizer_g.step()
        return loss_g.item()

    def train_step(
        self, real: Annotated[torch.Tensor, "[Batch, Time_Steps, 1] real price paths"]
    ) -> Annotated[dict, "{'loss_d': float, 'loss_g': Optional[float]}"]:
        batch_size, seq_len, _ = real.shape

        loss_d = None
        for _ in range(self.n_critic):
            loss_d = self.train_discriminator_step(real)

        loss_g = self.train_generator_step(batch_size, seq_len)
        return {"loss_d": loss_d, "loss_g": loss_g}
