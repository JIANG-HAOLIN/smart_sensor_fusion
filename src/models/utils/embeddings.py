import torch
from typing import Optional
import numpy as np


class ModalTypeEmbedding(torch.nn.Module):
    """Embedding for different model types by adding the modality emb vector"""

    def __init__(self, num_type: int = 2, emb_dim: int = 256, **kwargs):
        super().__init__()
        self.type_emb = torch.nn.Embedding(num_type, emb_dim)

    def forward(self, x: torch.Tensor, index: int) -> torch.Tensor:
        """
        args:
            x - input sequence of type [batch size, sequence len, token dim]
        """
        return x + self.type_emb(torch.full(x.shape[:-1], index, device=x.device))


class MdlLbCat(torch.nn.Module):
    """Embedding for different model types using concatenation of modality label (one-hot)"""

    def __init__(self, num_type: int = 2, emb_dim: int = 256, **kwargs):
        super().__init__()
        input_label = torch.zeros((num_type, num_type), device='cuda', requires_grad=False)
        type_emb = input_label.scatter_(1, torch.from_numpy(np.arange(num_type)).unsqueeze(1).to('cuda'), 1.0)
        self.register_buffer('type_emb', type_emb)
        print(self.type_emb)

    def forward(self, x: torch.Tensor, index: int) -> torch.Tensor:
        """
        args:
            x - input sequence of type [batch size, sequence len, token dim]
            index - which dim to be embedded
        """

        embs = self.type_emb[index].view(1, 1, -1)
        embs = embs.expand(x.shape[0], x.shape[1], -1)
        return torch.cat((x, embs.to(x.device)), dim=-1)


class MdlEmbCat(torch.nn.Module):
    """Embedding for different model types by concatenating the modality emb vector"""

    def __init__(self, num_type: int = 2, emb_dim: int = 256, **kwargs):
        super().__init__()
        self.type_emb = torch.nn.Embedding(num_type, emb_dim)

    def forward(self, x: torch.Tensor, index: int) -> torch.Tensor:
        """
        args:
            x - input sequence of type [batch size, sequence len, token dim]
            index - which dim to be embedded
        """

        return torch.cat((x, self.type_emb(torch.full(x.shape[:-1], index, device=x.device))), dim=-1)


class VitPatchEmbedding(torch.nn.Module):
    """Embedding for tokens of ViT"""

    def __init__(self, num_patches: int, emb_dim: int = 256, **kwargs):
        super().__init__()
        self.pos_embedding = torch.nn.Parameter(torch.randn(1, num_patches, emb_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        args:
            x - input sequence of type [batch size, sequence len, token dim]
        """
        if x.shape[1] > self.pos_embedding.shape[1]:
            raise RuntimeError(f'the input sequence length {x.shape[1]} is'
                               f'larger than the maximum number of usable positional {self.pos_embedding.shape[1]} '
                               f'embedding vectors, please use larger num_patches !')
        return x + self.pos_embedding[:, :x.shape[1]]


class LearnablePosEmb(torch.nn.Module):
    """General embedding for token sequence of any shape alone single dimension"""

    def __init__(self, num_emb: int, emb_dim: int = 256, **kwargs):
        super().__init__()
        self.pos_embedding = torch.nn.Parameter(torch.randn(num_emb, emb_dim))

    def forward(self, x: torch.Tensor, which_dim: int = 1, **kwargs) -> torch.Tensor:
        """
        args:
            x - input sequence of type [batch size, len1, len2, len3, token dim]
            which_dim - embded along which dimension
        """
        out_shape = len(x.shape)
        pos_embedding = self.pos_embedding
        for i in range(out_shape - 1):
            if i != which_dim:
                pos_embedding = pos_embedding.unsqueeze(i)
        embs = pos_embedding.index_select(which_dim, torch.arange(x.shape[which_dim], device=x.device))
        return x + embs
