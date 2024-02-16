import torch.nn as nn
import torch
from typing import Optional


class TransformerEncoder(nn.Module):
    """ Implementation for transformer encoder with self attention mechanism using MultiHeadAttention layer"""

    def __init__(self, token_dim: int, num_blocks: int, num_heads: int,
                 middle_dim_mlp: Optional[int] = None, dropout: float = 0.,
                 batch_first: bool = True, norm_first: bool = False):
        """

        Args:
            token_dim: the input dimension of embedded tokens and embedded q,k,v dimension
            num_blocks: number of blocks
            num_heads: number of attention heads
            middle_dim_mlp: the intermediate dimension of feedforward network
        """
        super().__init__()
        self.token_dim = token_dim
        self.norm_first = norm_first
        self.final_norm = nn.LayerNorm(token_dim, eps=1e-5) if norm_first else nn.Identity()
        self.layers = nn.ModuleList([])
        middle_dim_mlp = 2 * token_dim if middle_dim_mlp is None else middle_dim_mlp
        for _ in range(num_blocks):
            self.layers.append(nn.ModuleList([
                nn.LayerNorm(token_dim) if norm_first else nn.Identity(),
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
                nn.Identity() if norm_first else nn.LayerNorm(token_dim),
            ]))

    def forward(self, x: torch.Tensor) -> (torch.Tensor, list):
        """

        Args:
            
            x: Input features of shape [Batch, SeqLen, input_dim]
        Returns:
            Output features of shape [Batch, SeqLen, input_dim]

        """
        """
        norm_first True
            i
            |
            LN
            |
        ----|
        |   |
        |   |
        |   |
        |  Attn
        |   |
        |   |
        |   |
        ----|
            |
        ----|
        |   |
        |   |
        |   LN
        |   |
        |   |
        |  FFN
        |   |
        |   |
        |   |
        ----|
            |
            |
            o  
        
        norm_first False
            i
            |
        ----|
        |   |
        |  Attn
        |   |
        ----|
            |
        ----|
        |   LN
        |   |
        |  FFN
        |   |
        ----|
            LN
            |
            o        
        """
        attn_maps = []
        for in_norm, attention, feedforward, out_norm in self.layers:
            x = in_norm(x)
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
        return self.final_norm(x), attn_maps


class TransformerEncoderVanilla(nn.Module):
    """ Implementation for transformer encoder with self attention mechanism using MultiHeadAttention layer with
    vanilla setup """

    def __init__(self, token_dim: int, num_blocks: int, num_heads: int,
                 middle_dim_mlp: Optional[int] = None, dropout: float = 0.,
                 batch_first: bool = True, norm_first: bool = True):
        """

        Args:
            token_dim: the input dimension of embedded tokens and embedded q,k,v dimension
            num_blocks: number of blocks
            num_heads: number of attention heads
            middle_dim_mlp: the intermediate dimension of feedforward network
        """
        super().__init__()
        self.output_dim = token_dim
        self.layers = nn.ModuleList([])
        middle_dim_mlp = 2 * token_dim if middle_dim_mlp is None else middle_dim_mlp
        for _ in range(num_blocks):
            self.layers.append(TransformerEncoderLayerVanilla(token_dim=token_dim,
                                                              num_heads=num_heads,
                                                              middle_dim_mlp=middle_dim_mlp,
                                                              dropout=dropout,
                                                              batch_first=batch_first,
                                                              norm_first=norm_first))

    def forward(self, x: torch.Tensor) -> (torch.Tensor, list):
        """

        Args:

            x: Input features of shape [Batch, SeqLen, input_dim]
        Returns:
            Output features of shape [Batch, SeqLen, input_dim]
        """
        attn_maps = []
        for layer in self.layers:
            x, attn_map = layer(x)
            attn_maps.append(attn_map)
        return x, attn_maps


class TransformerEncoderLayerVanilla(nn.Module):
    """ Implementation for transformer encoder layer with self attention mechanism using MultiHeadAttention layer with
    vanilla setup """

    def __init__(self, token_dim: int, num_heads: int,
                 middle_dim_mlp: Optional[int] = None, dropout: float = 0.,
                 batch_first: bool = True, norm_first: bool = True):
        """

        Args:
            token_dim: the input dimension of embedded tokens and embedded q,k,v dimension
            num_blocks: number of blocks
            num_heads: number of attention heads
            middle_dim_mlp: the intermediate dimension of feedforward network
        """
        super().__init__()
        self.norm_first = norm_first
        self.layers = nn.ModuleList([])
        middle_dim_mlp = 2 * token_dim if middle_dim_mlp is None else middle_dim_mlp

        self.norm1 = nn.LayerNorm(token_dim, eps=1e-5)
        self.attn = nn.modules.activation.MultiheadAttention(embed_dim=token_dim, kdim=None, vdim=None,
                                                             num_heads=num_heads,
                                                             batch_first=batch_first,
                                                             dropout=dropout,
                                                             bias=True,
                                                             add_bias_kv=False,
                                                             add_zero_attn=False, )
        self.norm2 = nn.LayerNorm(token_dim, eps=1e-5)
        self.ffn = nn.Sequential(nn.Linear(token_dim, middle_dim_mlp),
                                 nn.ReLU(),
                                 nn.Linear(middle_dim_mlp, token_dim), )

    def forward(self, x: torch.Tensor) -> (torch.Tensor, list):
        """

        Args:

            x: Input features of shape [Batch, SeqLen, input_dim]
        Returns:
            Output features of shape [Batch, SeqLen, input_dim]
        """
        """
        norm_first True
        
            i
        ----|
        |   LN
        |   |
        |  Attn
        ----|
        ----|
        |   LN
        |   |
        |  FFN
        ----|
            o  

        norm_first False
        
            i
        ----|
        |  Attn
        ----|
            LN
            |
        ----|
        |  FFN
        ----|
            LN
            |
            o        
        """
        x_ = x
        if self.norm_first:
            x = self.norm1(x)
            x, attn_map = self.attn(x, x, x, need_weights=True, average_attn_weights=False, is_causal=False)
            x += x_
            x = x + self.ffn(self.norm2(x))
        else:
            x, attn_map = self.attn(x, x, x, need_weights=True, average_attn_weights=False, is_causal=False)
            x = self.norm1(x + x_)
            x = self.norm2(x + self.ffn(x))

        return x, attn_map
