import torch
from torch import nn, Tensor

# Add IMLE path
import os
import sys
submodule_name = 'imle'
(parent_folder_path, current_dir) = os.path.split(os.path.dirname(__file__))
sys.path.append(os.path.join(parent_folder_path, '../', submodule_name))

from imle.wrapper import imle
from imle.target import TargetDistribution
from imle.noise import SumOfGammaNoiseDistribution

def topk_solver(logits: Tensor, k=1):
    """Top-k solver for the logits.

    Args:
        logits (torch.Tensor): [*, num_categories] of unnormalized log probabilities.
        k (int, optional): size of the top-k. Defaults to 1.

    Returns:
        solution (torch.Tensor): [*, num_categories] binary mask corresponding to the top-k values.
    """
    _, idx = torch.topk(logits, k=k)
    mask = torch.zeros_like(logits)
    mask.scatter_(1, idx, 1.)
    return mask

TARGET_DISTRIBUTION = TargetDistribution(alpha=0.0, beta=10.0)
NOISE_DISTRIBUTION = SumOfGammaNoiseDistribution(k=1, nb_iterations=50)

class KIMLESampler(nn.Module):
    """Samples a k-subset with I-MLE.

    [1] Niepert, M., Minervini, P. and Franceschi, L., 2021. 
    Implicit MLE: backpropagating through discrete exponential family distributions. Advances in Neural Information Processing Systems, 
    34, pp.14567-14579.

    We use the original code from https://github.com/uclnlp/torch-imle imle. If you want to modify the hyper-parameters,
    refer to the original implementation.
    """
    def __init__(self, k=1, target_distribution=TargetDistribution(alpha=0.0, beta=10.0),
      noise_distribution=SumOfGammaNoiseDistribution(k=1, nb_iterations=50),
      nb_samples=1, input_noise_temperature=1.0, target_noise_temperature=1.0):
      super().__init__()
      self.sampler = imle(lambda logits: topk_solver(logits, k=k), target_distribution=target_distribution, noise_distribution=noise_distribution,
      nb_samples=nb_samples, input_noise_temperature=input_noise_temperature, target_noise_temperature=target_noise_temperature)

    def forward(self, x):
        return self.sampler(x)