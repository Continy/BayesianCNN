import torch
from torch.distributions import Normal
import sys


def monte_carlo(mean, var, true_label):
    criterion = torch.nn.CrossEntropyLoss()
    dist = Normal(mean, var)

    samples = dist.sample((1, ))
    samples = samples.squeeze(0)
    loss = criterion(samples, true_label)
    return loss
