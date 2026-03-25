# MIT License
#
# Copyright (c) Authors of
# "Cameras as Relative Positional Encoding" https://arxiv.org/pdf/2507.10496
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# FlashAttention-compatible version of PRoPE attention.
# All q, k, v tensors use shape (batch, seqlen, num_heads, head_dim) — the
# native layout for flash_attn — eliminating transposes vs. the original file.
#
# How to use PRoPE attention for self-attention:
#
# 1. Easiest way (fast):
#    attn = PropeDotProductAttentionFA(...)
#    o = attn(q, k, v, viewmats, Ks)
#
# 2. More flexible way (fast):
#    attn = PropeDotProductAttentionFA(...)
#    attn._precompute_and_cache_apply_fns(viewmats, Ks)
#    q = attn._apply_to_q(q)
#    k = attn._apply_to_kv(k)
#    v = attn._apply_to_kv(v)
#    o = flash_attn_func(q, k, v, **kwargs)
#    o = attn._apply_to_o(o)
#
# How to use PRoPE attention for cross-attention:
#
#    attn_src = PropeDotProductAttentionFA(...)
#    attn_tgt = PropeDotProductAttentionFA(...)
#    attn_src._precompute_and_cache_apply_fns(viewmats_src, Ks_src)
#    attn_tgt._precompute_and_cache_apply_fns(viewmats_tgt, Ks_tgt)
#    q_src = attn_src._apply_to_q(q_src)
#    k_tgt = attn_tgt._apply_to_kv(k_tgt)
#    v_tgt = attn_tgt._apply_to_kv(v_tgt)
#    o_src = flash_attn_func(q_src, k_tgt, v_tgt, **kwargs)
#    o_src = attn_src._apply_to_o(o_src)

from functools import partial
from typing import Callable, Optional, Tuple, List

import torch
from flash_attn.cute import flash_attn_func


class PropeDotProductAttention(torch.nn.Module):
    """PRoPE attention with precomputed RoPE coefficients.
    All q/k/v/o tensors use FlashAttention layout: (batch, seqlen, num_heads, head_dim).
    """

    coeffs_x_0: torch.Tensor
    coeffs_x_1: torch.Tensor
    coeffs_y_0: torch.Tensor
    coeffs_y_1: torch.Tensor

    def __init__(
        self,
        head_dim: int,
        patches_x: int,
        patches_y: int,
        image_width: int,
        image_height: int,
        num_register_tokens: int = 0,
        freq_base: float = 100.0,
        freq_scale: float = 1.0,
    ):
        super().__init__()
        self.head_dim = head_dim
        self.patches_x = patches_x
        self.patches_y = patches_y
        self.image_width = image_width
        self.image_height = image_height
        self.num_register_tokens = num_register_tokens

        coeffs_x_input = torch.tile(torch.arange(patches_x), (patches_y,))
        coeffs_y_input = torch.repeat_interleave(torch.arange(patches_y), patches_x)

        if num_register_tokens > 0:
            coeffs_x_input = coeffs_x_input + 1
            coeffs_y_input = coeffs_y_input + 1
            pos_special_x = torch.zeros(num_register_tokens, dtype=coeffs_x_input.dtype)
            pos_special_y = torch.zeros(num_register_tokens, dtype=coeffs_y_input.dtype)
            coeffs_x_input = torch.cat([pos_special_x, coeffs_x_input])
            coeffs_y_input = torch.cat([pos_special_y, coeffs_y_input])

        # Coefficients for PRoPE (camera-aware): head_dim split as [proj: dh/2, RoPE-x: dh/4, RoPE-y: dh/4]
        coeffs_x: Tuple[torch.Tensor, torch.Tensor] = _rope_precompute_coeffs(
            coeffs_x_input, freq_base=freq_base, freq_scale=freq_scale, feat_dim=head_dim // 4,
        )
        coeffs_y: Tuple[torch.Tensor, torch.Tensor] = _rope_precompute_coeffs(
            coeffs_y_input, freq_base=freq_base, freq_scale=freq_scale, feat_dim=head_dim // 4,
        )
        # Coefficients for plain RoPE (no camera): head_dim split as [RoPE-x: dh/2, RoPE-y: dh/2]
        coeffs_x_single: Tuple[torch.Tensor, torch.Tensor] = _rope_precompute_coeffs(
            coeffs_x_input, freq_base=freq_base, freq_scale=freq_scale, feat_dim=head_dim // 2,
        )
        coeffs_y_single: Tuple[torch.Tensor, torch.Tensor] = _rope_precompute_coeffs(
            coeffs_y_input, freq_base=freq_base, freq_scale=freq_scale, feat_dim=head_dim // 2,
        )

        # Not persistent: re-derived from camera params at test time.
        self.register_buffer("coeffs_x_0", coeffs_x[0], persistent=False)
        self.register_buffer("coeffs_x_1", coeffs_x[1], persistent=False)
        self.register_buffer("coeffs_y_0", coeffs_y[0], persistent=False)
        self.register_buffer("coeffs_y_1", coeffs_y[1], persistent=False)
        self.register_buffer("coeffs_x_single_0", coeffs_x_single[0], persistent=False)
        self.register_buffer("coeffs_x_single_1", coeffs_x_single[1], persistent=False)
        self.register_buffer("coeffs_y_single_0", coeffs_y_single[0], persistent=False)
        self.register_buffer("coeffs_y_single_1", coeffs_y_single[1], persistent=False)

    def load_state_dict(self, state_dict, strict=True):
        for key in [
            "coeffs_x_0", "coeffs_x_1", "coeffs_y_0", "coeffs_y_1",
            "coeffs_x_single_0", "coeffs_x_single_1",
            "coeffs_y_single_0", "coeffs_y_single_1",
        ]:
            state_dict.pop(key, None)
        super().load_state_dict(state_dict, strict)

    def forward(
        self,
        q: torch.Tensor,       # (batch, seqlen, num_heads, head_dim)
        k: torch.Tensor,       # (batch, seqlen, num_heads, head_dim)
        v: torch.Tensor,       # (batch, seqlen, num_heads, head_dim)
        viewmats: Optional[torch.Tensor],  # (batch, cameras, 4, 4)
        Ks: Optional[torch.Tensor],        # (batch, cameras, 3, 3)
        **kwargs,
    ) -> torch.Tensor:
        if viewmats is None:
            return prope_dot_product_attention_fa(
                q, k, v,
                viewmats=None,
                Ks=None,
                patches_x=self.patches_x,
                patches_y=self.patches_y,
                image_width=self.image_width,
                image_height=self.image_height,
                num_register_tokens=self.num_register_tokens,
                coeffs_x=(self.coeffs_x_single_0, self.coeffs_x_single_1),
                coeffs_y=(self.coeffs_y_single_0, self.coeffs_y_single_1),
                **kwargs,
            )
        else:
            return prope_dot_product_attention_fa(
                q, k, v,
                viewmats=viewmats,
                Ks=Ks,
                patches_x=self.patches_x,
                patches_y=self.patches_y,
                image_width=self.image_width,
                image_height=self.image_height,
                num_register_tokens=self.num_register_tokens,
                coeffs_x=(self.coeffs_x_0, self.coeffs_x_1),
                coeffs_y=(self.coeffs_y_0, self.coeffs_y_1),
                **kwargs,
            )

    def _precompute_and_cache_apply_fns(
        self, viewmats: torch.Tensor, Ks: Optional[torch.Tensor]
    ):
        (batch, cameras, _, _) = viewmats.shape
        assert viewmats.shape == (batch, cameras, 4, 4)
        assert Ks is None or Ks.shape == (batch, cameras, 3, 3)
        self.cameras = cameras

        self.apply_fn_q, self.apply_fn_kv, self.apply_fn_o = _prepare_apply_fns(
            head_dim=self.head_dim,
            viewmats=viewmats,
            Ks=Ks,
            patches_x=self.patches_x,
            patches_y=self.patches_y,
            image_width=self.image_width,
            image_height=self.image_height,
            coeffs_x=(self.coeffs_x_0, self.coeffs_x_1),
            coeffs_y=(self.coeffs_y_0, self.coeffs_y_1),
        )

    def _apply_to_q(self, q: torch.Tensor) -> torch.Tensor:
        (batch, seqlen, num_heads, head_dim) = q.shape
        assert seqlen == self.cameras * self.patches_x * self.patches_y
        assert head_dim == self.head_dim
        assert self.apply_fn_q is not None
        return self.apply_fn_q(q)

    def _apply_to_kv(self, kv: torch.Tensor) -> torch.Tensor:
        (batch, seqlen, num_heads, head_dim) = kv.shape
        assert seqlen == self.cameras * self.patches_x * self.patches_y
        assert head_dim == self.head_dim
        assert self.apply_fn_kv is not None
        return self.apply_fn_kv(kv)

    def _apply_to_o(self, o: torch.Tensor) -> torch.Tensor:
        (batch, seqlen, num_heads, head_dim) = o.shape
        assert seqlen == self.cameras * self.patches_x * self.patches_y
        assert head_dim == self.head_dim
        assert self.apply_fn_o is not None
        return self.apply_fn_o(o)


def prope_dot_product_attention_fa(
    q: torch.Tensor,  # (batch, seqlen, num_heads, head_dim)
    k: torch.Tensor,  # (batch, seqlen, num_heads, head_dim)
    v: torch.Tensor,  # (batch, seqlen, num_heads, head_dim)
    *,
    viewmats: Optional[torch.Tensor],  # (batch, cameras, 4, 4)
    Ks: Optional[torch.Tensor],        # (batch, cameras, 3, 3)
    patches_x: int,
    patches_y: int,
    image_width: int,
    image_height: int,
    num_register_tokens: int,
    coeffs_x: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    coeffs_y: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    **kwargs,
) -> torch.Tensor:
    """PRoPE attention with FlashAttention layout (batch, seqlen, num_heads, head_dim)."""
    (batch, seqlen, num_heads, head_dim) = q.shape

    if viewmats is not None:
        cameras = viewmats.shape[1]
        assert q.shape == k.shape == v.shape
        assert viewmats.shape == (batch, cameras, 4, 4)
        assert Ks is None or Ks.shape == (batch, cameras, 3, 3)
        assert seqlen == cameras * patches_x * patches_y + cameras * num_register_tokens

        apply_fn_q, apply_fn_kv, apply_fn_o = _prepare_apply_fns(
            head_dim=head_dim,
            viewmats=viewmats,
            Ks=Ks,
            patches_x=patches_x,
            patches_y=patches_y,
            image_width=image_width,
            image_height=image_height,
            coeffs_x=coeffs_x,
            coeffs_y=coeffs_y,
        )
        out, _ = flash_attn_func(apply_fn_q(q).to(q), apply_fn_kv(k).to(k), apply_fn_kv(v).to(v), **kwargs)
        out = apply_fn_o(out)
    else:
        apply_fn_q, apply_fn_kv = _prepare_apply_fns_rope(
            head_dim=head_dim,
            patches_x=patches_x,
            patches_y=patches_y,
            image_width=image_width,
            image_height=image_height,
            coeffs_x=coeffs_x,
            coeffs_y=coeffs_y,
        )
        out, _ = flash_attn_func(apply_fn_q(q).to(q), apply_fn_kv(k).to(k), v, **kwargs)

    assert out.shape == (batch, seqlen, num_heads, head_dim)
    return out


def _prepare_apply_fns(
    head_dim: int,
    viewmats: Optional[torch.Tensor],  # (batch, cameras, 4, 4)
    Ks: Optional[torch.Tensor],        # (batch, cameras, 3, 3)
    patches_x: int,
    patches_y: int,
    image_width: int,
    image_height: int,
    coeffs_x: Optional[torch.Tensor] = None,
    coeffs_y: Optional[torch.Tensor] = None,
) -> Tuple[
    Callable[[torch.Tensor], torch.Tensor],
    Callable[[torch.Tensor], torch.Tensor],
    Callable[[torch.Tensor], torch.Tensor],
]:
    """Prepare PRoPE transforms for FA-layout (batch, seqlen, num_heads, head_dim) tensors."""
    device = viewmats.device
    (batch, cameras, _, _) = viewmats.shape

    if Ks is not None:
        Ks_norm = torch.zeros_like(Ks)
        Ks_norm[..., 0, 0] = Ks[..., 0, 0] / image_width
        Ks_norm[..., 1, 1] = Ks[..., 1, 1] / image_height
        Ks_norm[..., 0, 2] = Ks[..., 0, 2] / image_width - 0.5
        Ks_norm[..., 1, 2] = Ks[..., 1, 2] / image_height - 0.5
        Ks_norm[..., 2, 2] = 1.0
        del Ks

        P     = torch.einsum("...ij,...jk->...ik", _lift_K(Ks_norm), viewmats)
        P_T   = P.transpose(-1, -2)
        P_inv = torch.einsum("...ij,...jk->...ik", _invert_SE3(viewmats), _lift_K(_invert_K(Ks_norm)))
    else:
        P     = viewmats
        P_T   = P.transpose(-1, -2)
        P_inv = _invert_SE3(viewmats)

    assert P.shape == P_inv.shape == (batch, cameras, 4, 4)

    if coeffs_x is None:
        coeffs_x = _rope_precompute_coeffs(
            torch.tile(torch.arange(patches_x, device=device), (patches_y * cameras,)),
            freq_base=100.0, freq_scale=1.0, feat_dim=head_dim // 4,
        )
    if coeffs_y is None:
        coeffs_y = _rope_precompute_coeffs(
            torch.tile(
                torch.repeat_interleave(torch.arange(patches_y, device=device), patches_x),
                (cameras,),
            ),
            freq_base=100.0, freq_scale=1.0, feat_dim=head_dim // 4,
        )

    assert head_dim % 4 == 0
    # head_dim split: [proj dh/2 | RoPE-x dh/4 | RoPE-y dh/4]
    transforms_q = [
        (partial(_apply_tiled_projmat, matrix=P_T),        head_dim // 2),
        (partial(_rope_apply_coeffs, coeffs=coeffs_x),     head_dim // 4),
        (partial(_rope_apply_coeffs, coeffs=coeffs_y),     head_dim // 4),
    ]
    transforms_kv = [
        (partial(_apply_tiled_projmat, matrix=P_inv),      head_dim // 2),
        (partial(_rope_apply_coeffs, coeffs=coeffs_x),     head_dim // 4),
        (partial(_rope_apply_coeffs, coeffs=coeffs_y),     head_dim // 4),
    ]
    transforms_o = [
        (partial(_apply_tiled_projmat, matrix=P),          head_dim // 2),
        (partial(_rope_apply_coeffs, coeffs=coeffs_x, inverse=True), head_dim // 4),
        (partial(_rope_apply_coeffs, coeffs=coeffs_y, inverse=True), head_dim // 4),
    ]

    apply_fn_q  = partial(_apply_block_diagonal, func_size_pairs=transforms_q)
    apply_fn_kv = partial(_apply_block_diagonal, func_size_pairs=transforms_kv)
    apply_fn_o  = partial(_apply_block_diagonal, func_size_pairs=transforms_o)
    return apply_fn_q, apply_fn_kv, apply_fn_o


def _prepare_apply_fns_rope(
    head_dim: int,
    patches_x: int,
    patches_y: int,
    image_width: int,
    image_height: int,
    coeffs_x: Optional[torch.Tensor] = None,
    coeffs_y: Optional[torch.Tensor] = None,
) -> Tuple[
    Callable[[torch.Tensor], torch.Tensor],
    Callable[[torch.Tensor], torch.Tensor],
]:
    """Prepare plain 2-D RoPE transforms (no camera info) for FA-layout tensors."""
    assert head_dim % 2 == 0
    # head_dim split: [RoPE-x dh/2 | RoPE-y dh/2]
    transforms_q = [
        (partial(_rope_apply_coeffs, coeffs=coeffs_x), head_dim // 2),
        (partial(_rope_apply_coeffs, coeffs=coeffs_y), head_dim // 2),
    ]
    transforms_kv = [
        (partial(_rope_apply_coeffs, coeffs=coeffs_x), head_dim // 2),
        (partial(_rope_apply_coeffs, coeffs=coeffs_y), head_dim // 2),
    ]
    apply_fn_q  = partial(_apply_block_diagonal, func_size_pairs=transforms_q)
    apply_fn_kv = partial(_apply_block_diagonal, func_size_pairs=transforms_kv)
    return apply_fn_q, apply_fn_kv


# ---------------------------------------------------------------------------
# Low-level helpers — adapted for FA layout (batch, seqlen, num_heads, head_dim)
# ---------------------------------------------------------------------------

def _apply_tiled_projmat(
    feats: torch.Tensor,   # (batch, seqlen, num_heads, feat_dim)
    matrix: torch.Tensor,  # (batch, cameras, D, D)
) -> torch.Tensor:
    """Apply a per-camera projection matrix to the first feat_dim//D sub-block."""
    (batch, seqlen, num_heads, feat_dim) = feats.shape
    cameras = matrix.shape[1]
    assert seqlen > cameras and seqlen % cameras == 0
    D = matrix.shape[-1]
    assert matrix.shape == (batch, cameras, D, D)
    assert feat_dim % D == 0

    # (b, seqlen, nh, dh) -> (b, cameras, patches_per_cam, nh, dh//D, D)
    feats_r = feats.reshape(batch, cameras, -1, num_heads, feat_dim // D, D)
    # matrix[b,c,i,j] * feats[b,c,p,n,k,j]  ->  result[b,c,p,n,k,i]
    result = torch.einsum("bcij,bcpnkj->bcpnki", matrix, feats_r)
    return result.reshape(feats.shape)


def _rope_precompute_coeffs(
    positions: torch.Tensor,  # (seqlen,)
    freq_base: float,
    freq_scale: float,
    feat_dim: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Precompute RoPE (cos, sin) coefficients with FA layout broadcast shape.

    Returns tensors of shape (1, seqlen, 1, num_freqs) so they broadcast
    correctly over (batch, seqlen, num_heads, head_dim).
    """
    assert len(positions.shape) == 1
    assert feat_dim % 2 == 0
    num_freqs = feat_dim // 2
    # (1, 1, 1, num_freqs) — broadcast over all dims except frequencies
    freqs = freq_scale * (
        freq_base ** (-torch.arange(num_freqs, device=positions.device)[None, None, None, :] / num_freqs)
    )
    # (1, seqlen, 1, num_freqs)
    angles = positions[None, :, None, None] * freqs
    assert angles.shape == (1, positions.shape[0], 1, num_freqs)
    return torch.cos(angles), torch.sin(angles)


def _rope_apply_coeffs(
    feats: torch.Tensor,  # (batch, seqlen, num_heads, feat_dim)
    coeffs: Tuple[torch.Tensor, torch.Tensor],
    inverse: bool = False,
) -> torch.Tensor:
    """Apply RoPE with split (non-interleaved) convention.
    cos/sin shape: (1, seqlen, 1, feat_dim//2), broadcasts over batch and num_heads.
    """
    cos, sin = coeffs
    # Tile over cameras if coeffs cover one image worth of tokens.
    if cos.shape[1] != feats.shape[1]:
        n_repeats = feats.shape[1] // cos.shape[1]
        cos = cos.repeat(1, n_repeats, 1, 1)
        sin = sin.repeat(1, n_repeats, 1, 1)
    assert len(feats.shape) == len(cos.shape) == len(sin.shape) == 4
    assert cos.shape[-1] == sin.shape[-1] == feats.shape[-1] // 2

    x_in = feats[..., : feats.shape[-1] // 2]
    y_in = feats[..., feats.shape[-1] // 2 :]
    if not inverse:
        return torch.cat([cos * x_in + sin * y_in, -sin * x_in + cos * y_in], dim=-1)
    else:
        return torch.cat([cos * x_in - sin * y_in,  sin * x_in + cos * y_in], dim=-1)


def _apply_block_diagonal(
    feats: torch.Tensor,  # (..., dim)
    func_size_pairs: List[Tuple[Callable[[torch.Tensor], torch.Tensor], int]],
) -> torch.Tensor:
    """Apply a block-diagonal function to the last dimension of feats."""
    funcs, block_sizes = zip(*func_size_pairs)
    assert feats.shape[-1] == sum(block_sizes)
    x_blocks = torch.split(feats, block_sizes, dim=-1)
    out = torch.cat([f(x) for f, x in zip(funcs, x_blocks)], dim=-1)
    assert out.shape == feats.shape
    return out


# ---------------------------------------------------------------------------
# SE(3) / intrinsics utilities (unchanged from prope_custom.py)
# ---------------------------------------------------------------------------

def _invert_SE3(transforms: torch.Tensor) -> torch.Tensor:
    assert transforms.shape[-2:] == (4, 4)
    Rinv = transforms[..., :3, :3].transpose(-1, -2)
    out = torch.zeros_like(transforms)
    out[..., :3, :3] = Rinv
    out[..., :3, 3] = -torch.einsum("...ij,...j->...i", Rinv, transforms[..., :3, 3])
    out[..., 3, 3] = 1.0
    return out


def _lift_K(Ks: torch.Tensor) -> torch.Tensor:
    assert Ks.shape[-2:] == (3, 3)
    out = torch.zeros(Ks.shape[:-2] + (4, 4), device=Ks.device)
    out[..., :3, :3] = Ks
    out[..., 3, 3] = 1.0
    return out


def _invert_K(Ks: torch.Tensor) -> torch.Tensor:
    assert Ks.shape[-2:] == (3, 3)
    out = torch.zeros_like(Ks)
    out[..., 0, 0] = 1.0 / Ks[..., 0, 0]
    out[..., 1, 1] = 1.0 / Ks[..., 1, 1]
    out[..., 0, 2] = -Ks[..., 0, 2] / Ks[..., 0, 0]
    out[..., 1, 2] = -Ks[..., 1, 2] / Ks[..., 1, 1]
    out[..., 2, 2] = 1.0
    return out
