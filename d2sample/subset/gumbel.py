import numpy as np
import torch
from torch import nn, Tensor

EPSILON = np.finfo(np.float32).tiny

def k_gumbel_softmax(logits: Tensor, k=1, hard=False, tau=1.0) -> Tensor:
    """Samples a k-subset from the logits using a relaxation of top-k Gumbel Softmax.
    [1] Xie, S.M. and Ermon, S., 2019. Reparameterizable subset sampling via continuous relaxations. arXiv preprint arXiv:1901.10517.

    The code is adapted from their repository: https://github.com/ermongroup/subsets.

    Args:
        logits (torch.Tensor): [*, num_categories] of unnormalized log probabilities.
        k (int, optional): size of the subset to sample. Defaults to 1.
        hard (bool, optional): whether to return the hard samples with straight-through estimation. Defaults to False.
        tau (float, optional): temperature of the softmax. Defaults to 1.0.

    Returns:
        samples (torch.Tensor): [*, num_categories] of binary values corresponding to the samples.
    """
    
    # Draws Gumbel noise and perturb the logits
    gumbel = torch.distributions.gumbel.Gumbel(torch.zeros_like(logits), torch.ones_like(logits)).sample()
    logits = logits + gumbel

    # Continuous top-k relaxation (see https://github.com/ermongroup/subsets)
    khot = torch.zeros_like(logits)
    onehot_approx = torch.zeros_like(logits)
    for i in range(k):
        khot_mask = torch.max(1.0 - onehot_approx, torch.tensor([EPSILON]).to(logits.device))
        logits = logits + torch.log(khot_mask)
        onehot_approx = torch.nn.functional.softmax(logits / tau, dim=-1)
        khot = khot + onehot_approx

    if hard:
        # Straight-through estimation
        khot_hard = torch.zeros_like(khot)
        _, ind = torch.topk(khot, k, dim=-1)
        khot_hard = khot_hard.scatter_(1, ind, 1)
        res = khot_hard - khot.detach() + khot
    else:
        res = khot

    return res

class TopKGumbelSoftmax(nn.Module):
    # Simple wrapper around k_gumbel_softmax
    def __init__(self, k=1, hard=False, tau=1.0):
        super().__init__()
        self.k = k
        self.hard = hard
        self.tau = tau
    
    def forward(self, x):
        return k_gumbel_softmax(x, k=self.k, hard=self.hard, tau=self.tau)