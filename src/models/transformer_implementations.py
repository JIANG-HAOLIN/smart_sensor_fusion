import torch.nn as nn
import torch
from typing import Optional


class TransformerEncoder(nn.Module):
    """ Implementation for transformer encoder with self attention mechanism using MultiHeadAttention layer"""

    def __init__(self, token_dim: int, num_blocks: int, num_heads: int,
                 middle_dim_mlp: Optional[int] = None, dropout: float = 0.,
                 batch_first: bool = True):
        """

        Args:
            token_dim: the input dimension of embedded tokens
            num_blocks: number of blocks
            num_heads: number of attention heads
            middle_dim_mlp: the intermediate dimension of feedforward network
        """
        super().__init__()
        # self.norm = nn.LayerNorm(token_dim, eps=1e-5)
        self.layers = nn.ModuleList([])
        middle_dim_mlp = 2 * token_dim if middle_dim_mlp is None else middle_dim_mlp
        for _ in range(num_blocks):
            self.layers.append(nn.ModuleList([
                nn.modules.activation.MultiheadAttention(embed_dim=token_dim, kdim=None, vdim=None,
                                                         num_heads=num_heads,
                                                         batch_first=batch_first,
                                                         dropout=dropout,
                                                         bias=True,
                                                         add_bias_kv=False,
                                                         add_zero_attn=False, ),
                nn.Sequential(nn.LayerNorm(token_dim),
                              nn.Linear(token_dim, middle_dim_mlp),
                              nn.ReLU(),
                              nn.Linear(middle_dim_mlp, token_dim), ),
                nn.LayerNorm(token_dim),
            ]))

    def forward(self, x: torch.Tensor) -> (torch.Tensor, list):
        """

        Args:
            x: Input features of shape [Batch, SeqLen, input_dim]
        Returns:
            Output features of shape [Batch, SeqLen, input_dim]
        """
        attn_maps = []
        for attention, feedforward, out_norm in self.layers:
            x_, attn_map = attention(query=x,
                                     key=x,
                                     value=x,
                                     key_padding_mask=None,
                                     need_weights=True,
                                     attn_mask=None,
                                     average_attn_weights=False,
                                     is_causal=False)
            x = x + x_
            attn_maps.append(attn_map)
            x = out_norm(feedforward(x) + x)
        return x, attn_maps
