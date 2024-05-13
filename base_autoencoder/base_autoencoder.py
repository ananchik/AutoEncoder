from abc import ABC, abstractmethod
import torch
import torch.nn as nn


class AutoEncoder(ABC, nn.Module):
    def  __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs,)

    @abstractmethod
    def forward(self, x):
        pass

    def bottleneck_emb():
        pass

    def fit():
        pass

    def predict():
        pass