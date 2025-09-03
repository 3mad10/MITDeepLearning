import torch
import torch.nn as nn

network = nn.Sequential(
    nn.Linear(2, 5),
    nn.ReLU(),
    nn.Linear(5, 1),
    nn.Sigmoid()
)

print(network)

loss = nn.functional.binary_cross_entropy()
