import torch
from loguru import logger


def demo_metric(outputs, target):
    with torch.no_grad():
        pred = torch.argmax(outputs, dim=1)
        assert pred.shape[0] == len(target)
        correct = 0
        correct += torch.sum(pred == target).item()
    return correct / len(target)
