
import torch.nn.functional as F


def demo_loss(outputs, target):
    return F.cross_entropy(outputs, target)
