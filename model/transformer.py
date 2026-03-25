import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

try:
    from flash_attn.cute import flash_attn_func
    FA4_AVAILABLE = True
except ImportError:
    FA4_AVAILABLE = False


# src: https://github.com/pytorch/benchmark/blob/main/torchbenchmark/models/llama/model.py#L28
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)

        return output * self.weight.type_as(x)

class MLP(nn.Module):

    def __init__(self, dim, inter_multi=4, bias=False):
        super().__init__()
        intermediate_dim = int(dim * inter_multi)
        self.c_fc = nn.Linear(dim, intermediate_dim, bias=bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(intermediate_dim, dim, bias=bias)

    def forward(self, x, *args):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x

class SelfAttention(nn.Module):
    """
    Self-attention layer
    Reference: https://github.com/facebookresearch/dino/blob/7c446df5b9f45747937fb0d72314eb9f7b66930a/vision_transformer.py#L68-L92
    """

    def __init__(
        self,
        dim,
        head_dim,
        use_qk_norm=True,
        causal=False,
        bias=False,
    ):
        super().__init__()
        assert dim % head_dim == 0
        self.dim = dim
        self.head_dim = head_dim

        self.to_qkv = nn.Linear(dim, 3 * dim, bias=bias)
        self.c_proj = nn.Linear(dim, dim, bias=bias)
        self.use_qk_norm = use_qk_norm

        if self.use_qk_norm:
            self.q_norm = RMSNorm(head_dim)
            self.k_norm = RMSNorm(head_dim)

        self.causal = causal

    def forward(self, x, prope, stage, *args):
        """
        x: (b, l, d)
        """
        qkv = self.to_qkv(x)
        if FA4_AVAILABLE:
            q, k, v = rearrange(qkv, "b l (qkv nh dh) -> qkv b l nh dh", qkv=3, dh=self.head_dim)
        else:
            q, k, v = rearrange(qkv, "b l (qkv nh dh) -> qkv b nh l dh", qkv=3, dh=self.head_dim)
        if self.use_qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)          
        if stage == 1:
            if FA4_AVAILABLE:
                x, _ = flash_attn_func(q, k, v, causal=self.causal)
            else:
                x = F.scaled_dot_product_attention(q, k, v, is_causal=self.causal)

        elif stage == 2:
            if prope:
                w2c = args[0]["w2c"]
                Ks = args[0]["Ks"]
                attn_fn = args[0]["attn2"]

                x = attn_fn(
                    q, k, v,
                    viewmats=w2c,
                    Ks=Ks,
                )
            else:
                attn_fn = args[0]["attn2"]
                x = attn_fn(
                    q, k, v,
                    viewmats=None,
                    Ks=None,                    
                )
        elif stage == 3:
            if prope:
                w2c = args[0]["w2c"]
                Ks = args[0]["Ks"]
                attn_fn = args[0]["attn3"]

                x = attn_fn(
                    q, k, v,
                    viewmats=w2c,
                    Ks=Ks,
                )
            else:
                attn_fn = args[0]["attn3"]
                x = attn_fn(
                    q, k, v,
                    viewmats=None,
                    Ks=None,                    
                )
        if FA4_AVAILABLE:
            x = rearrange(x, "b l nh dh -> b l (nh dh)")
        else:
            x = rearrange(x, "b nh l dh -> b l (nh dh)")

        x = self.c_proj(x)
        return x
    
class TransformerBlock(nn.Module):
    def __init__(self, dim, bias, head_dim, inter_multi, use_qk_norm):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim, bias=bias, eps=1e-5)
        self.attn = SelfAttention(dim=dim, bias=bias, head_dim=head_dim, use_qk_norm=use_qk_norm)

        self.ln2 = nn.LayerNorm(dim, bias=bias, eps=1e-5)
        self.mlp = MLP(dim=dim, bias=bias, inter_multi=inter_multi)

    def forward(self, x, prope, stage, info):
        x = x + self.attn(self.ln1(x), prope, stage, info)
        x = x + self.mlp(self.ln2(x))
        return x