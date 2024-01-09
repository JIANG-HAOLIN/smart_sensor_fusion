import torch
from typing import Dict, List, Optional


class EarlySum(torch.nn.Module):
    def __init__(self, mod_names: List, dim: int = 256):
        super().__init__()
        for modname in mod_names:
            self.__setattr__(f"{modname}_gammas", torch.nn.Parameter(torch.randn(1, 1, dim)))

    def forward(self, multimod_input):
        sum = 0
        for idx, (modname, modvalue) in enumerate(multimod_input.items()):
            gamma = self.__getattr__(f"{modname}_gammas")
            sum = sum + gamma * modvalue
        return sum

    def __repr__(self):
        a = [(key, value.shape) for key, value in self.__dict__['_parameters'].items()]
        return f'{self.__class__.__name__} containing: {a}'


class EarlySumLinear(torch.nn.Module):
    def __init__(self, mod_names: List, dim: int = 256):
        super().__init__()
        for modname in mod_names:
            self.__setattr__(f"{modname}_gammas", torch.nn.Linear(dim, dim, bias=False))

    def forward(self, multimod_input):
        sum = 0
        for idx, (modname, modvalue) in enumerate(multimod_input.items()):
            gamma = self.__getattr__(f"{modname}_gammas")
            sum = sum + gamma(modvalue)
        return sum

    def __repr__(self):
        a = [(key, value) for key, value in self.__dict__['_modules'].items()]
        return f'{self.__class__.__name__} containing: {a}'


# print(sum.shape)
