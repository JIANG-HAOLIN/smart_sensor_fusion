import torch


class EarlySum(torch.nn.Module):
    def __init__(self, mod_names, dim: int = 256):
        super().__init__()
        for modname in mod_names:
            self.__setattr__(f"{modname}_gammas", torch.nn.Parameter(torch.randn(1, 1, dim)))

    def forward(self, multimod_input):
        sum = 0
        for idx, (modname, modvalue) in enumerate(multimod_input.items()):
            gamma = self.__getattr__(f"{modname}_gammas")
            sum = sum + gamma * modvalue
        return sum




