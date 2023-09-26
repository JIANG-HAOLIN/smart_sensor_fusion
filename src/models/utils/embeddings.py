import torch


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
