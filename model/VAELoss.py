import torch
from torch import nn
from torch.nn import functional as F


class VAELoss(nn.Module):
    def __init__(self):
        super(VAELoss, self).__init__()

    def forward(self, x_input, x_distribution_params, z_distribution_params, beta=None):
        # compute the reconstrution loss
        x_input = x_input.view(-1, 4 * 32 * 32)
        log_prob_loss = -1 * F.mse_loss(x_input, x_distribution_params[0], reduction="mean")

        # kl divergence (positive KL divergence)
        z_mu, z_log_var = z_distribution_params
        kl_divergence_loss = 0.5 * torch.sum(z_log_var.exp() + z_mu.pow(2) - 1 - z_log_var, dim=1).mean()

        # combine the loss
        if beta is None:
            loss = -1 * log_prob_loss + kl_divergence_loss
        else:
            loss = -1 * log_prob_loss + beta * kl_divergence_loss

        return loss, -1 * log_prob_loss, kl_divergence_loss

