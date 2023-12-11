import torch
from typing import Optional


class ModalTypeEmbedding(torch.nn.Module):
    """Embedding for different model types"""

    def __init__(self, num_type: int = 2, emb_dim: int = 256, **kwargs):
        super().__init__()
        self.type_emb = torch.nn.Embedding(num_type, emb_dim)

    def forward(self, x: torch.Tensor, index: int) -> torch.Tensor:
        """
        args:
            x - input sequence of type [batch size, sequence len, token dim]
        """
        return x + self.type_emb(torch.full(x.shape[:-1], index, device=x.device))


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
            raise RuntimeError('the input sequence length is larger than the maximum number of usable positional '
                               'embedding vectors, please use larger num_patches !')
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
