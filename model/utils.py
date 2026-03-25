import torch
from einops import rearrange


def compute_plucmap(fxfycxcy, c2w, h, w):
    """Transform target before computing loss
    Args:
        fxfycxcy (torch.tensor): [b, v, 4]
        c2w (torch.tensor): [b, v, 4, 4]
    Returns:
        ray_o: (b, v, 3, h, w)
        ray_d: (b, v, 3, h, w)
    """
    b, v = fxfycxcy.size(0), fxfycxcy.size(1)

    # Efficient meshgrid equivalent using broadcasting
    idx_x = torch.arange(w, device=c2w.device)[None, :].expand(h, -1)  # [h, w]
    idx_y = torch.arange(h, device=c2w.device)[:, None].expand(-1, w)  # [h, w]

    # Reshape for batched matrix multiplication
    idx_x = idx_x.flatten().expand(b * v, -1)           # [b*v, h*w]
    idx_y = idx_y.flatten().expand(b * v, -1)           # [b*v, h*w]

    fxfycxcy = fxfycxcy.reshape(b * v, 4)               # [b*v, 4]
    c2w = c2w.reshape(b * v, 4, 4)                      # [b*v, 4, 4]

    x = (idx_x + 0.5 - fxfycxcy[:, 2:3]) / fxfycxcy[:, 0:1]     # [b*v, h*w]
    y = (idx_y + 0.5 - fxfycxcy[:, 3:4]) / fxfycxcy[:, 1:2]     # [b*v, h*w]
    z = torch.ones_like(x)                                      # [b*v, h*w]

    ray_d = torch.stack([x, y, z], dim=1)                       # [b*v, 3, h*w]
    ray_d = torch.bmm(c2w[:, :3, :3], ray_d)                    # [b*v, 3, h*w]
    ray_d = ray_d / torch.norm(ray_d, dim=1, keepdim=True)      # [b*v, 3, h*w]

    ray_o = c2w[:, :3, 3:4].expand(b * v, -1, h*w)              # [b*v, 3, h*w]

    ray_o = ray_o.reshape(b, v, 3, h, w)                        # [b, v, 3, h, w]
    ray_d = ray_d.reshape(b, v, 3, h, w)                        # [b, v, 3, h, w]

    return ray_o, ray_d

def compute_rays(fxfycxcy, c2w, h, w):
    """Transform target before computing loss
    Args:
        fxfycxcy (torch.tensor): [b, v, 4]
        c2w (torch.tensor): [b, v, 4, 4]
    Returns:
        ray_o: (b, v, 3, h, w)
        ray_d: (b, v, 3, h, w)
    """
    b, v = fxfycxcy.size(0), fxfycxcy.size(1)

    # Efficient meshgrid equivalent using broadcasting
    idx_x = torch.arange(w, device=c2w.device)[None, :].expand(h, -1)  # [h, w]
    idx_y = torch.arange(h, device=c2w.device)[:, None].expand(-1, w)  # [h, w]

    # Reshape for batched matrix multiplication
    idx_x = idx_x.flatten().expand(b * v, -1)           # [b*v, h*w]
    idx_y = idx_y.flatten().expand(b * v, -1)           # [b*v, h*w]

    fxfycxcy = fxfycxcy.reshape(b * v, 4)               # [b*v, 4]
    c2w = c2w.reshape(b * v, 4, 4)                      # [b*v, 4, 4]

    x = (idx_x + 0.5 - fxfycxcy[:, 2:3]) / fxfycxcy[:, 0:1]     # [b*v, h*w]
    y = (idx_y + 0.5 - fxfycxcy[:, 3:4]) / fxfycxcy[:, 1:2]     # [b*v, h*w]
    z = torch.ones_like(x)                                      # [b*v, h*w]

    ray_d = torch.stack([x, y, z], dim=1)                       # [b*v, 3, h*w]
    ray_d = torch.bmm(c2w[:, :3, :3], ray_d)                    # [b*v, 3, h*w]
    ray_d = ray_d / torch.norm(ray_d, dim=1, keepdim=True)      # [b*v, 3, h*w]

    ray_o = c2w[:, :3, 3:4].expand(b * v, -1, h*w)              # [b*v, 3, h*w]

    ray_o = ray_o.reshape(b, v, 3, h, w)                        # [b, v, 3, h, w]
    ray_d = ray_d.reshape(b, v, 3, h, w)                        # [b, v, 3, h, w]

    ray_o = rearrange(ray_o, 'b v c h w -> b (v h w) c')
    ray_d = rearrange(ray_d, 'b v c h w -> b (v h w) c')

    return ray_o, ray_d


def position_grid_to_embed(pos_grid: torch.Tensor, embed_dim: int, omega_0: float = 100) -> torch.Tensor:
    """
    Convert 2D position grid (HxWx2) to sinusoidal embeddings (HxWxC)

    Args:
        pos_grid: Tensor of shape (H, W, 2) containing 2D coordinates
        embed_dim: Output channel dimension for embeddings

    Returns:
        Tensor of shape (H, W, embed_dim) with positional embeddings
    """
    H, W, grid_dim = pos_grid.shape
    assert grid_dim == 2
    pos_flat = pos_grid.reshape(-1, grid_dim)  # Flatten to (H*W, 2)

    # Process x and y coordinates separately
    emb_x = make_sincos_pos_embed(embed_dim // 2, pos_flat[:, 0], omega_0=omega_0)  # [1, H*W, D/2]
    emb_y = make_sincos_pos_embed(embed_dim // 2, pos_flat[:, 1], omega_0=omega_0)  # [1, H*W, D/2]

    # Combine and reshape
    emb = torch.cat([emb_x, emb_y], dim=-1)  # [1, H*W, D]

    return emb.view(H, W, embed_dim)  # [H, W, D]


def make_sincos_pos_embed(embed_dim: int, pos: torch.Tensor, omega_0: float = 100) -> torch.Tensor:
    """
    This function generates a 1D positional embedding from a given grid using sine and cosine functions.

    Args:
    - embed_dim: The embedding dimension.
    - pos: The position to generate the embedding from.

    Returns:
    - emb: The generated 1D positional embedding.
    """
    assert embed_dim % 2 == 0
    device = pos.device
    omega = torch.arange(embed_dim // 2, dtype=torch.float32 if device.type == "mps" else torch.double, device=device)
    omega /= embed_dim / 2.0
    omega = 1.0 / omega_0**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = torch.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = torch.sin(out)  # (M, D/2)
    emb_cos = torch.cos(out)  # (M, D/2)

    emb = torch.cat([emb_sin, emb_cos], dim=1)  # (M, D)
    return emb.float()


# Inspired by https://github.com/microsoft/moge


def create_uv_grid(
    width: int, height: int, aspect_ratio: float = None, dtype: torch.dtype = None, device: torch.device = None
) -> torch.Tensor:
    """
    Create a normalized UV grid of shape (width, height, 2).

    The grid spans horizontally and vertically according to an aspect ratio,
    ensuring the top-left corner is at (-x_span, -y_span) and the bottom-right
    corner is at (x_span, y_span), normalized by the diagonal of the plane.

    Args:
        width (int): Number of points horizontally.
        height (int): Number of points vertically.
        aspect_ratio (float, optional): Width-to-height ratio. Defaults to width/height.
        dtype (torch.dtype, optional): Data type of the resulting tensor.
        device (torch.device, optional): Device on which the tensor is created.

    Returns:
        torch.Tensor: A (width, height, 2) tensor of UV coordinates.
    """
    # Derive aspect ratio if not explicitly provided
    if aspect_ratio is None:
        aspect_ratio = float(width) / float(height)

    # Compute normalized spans for X and Y
    diag_factor = (aspect_ratio**2 + 1.0) ** 0.5
    span_x = aspect_ratio / diag_factor
    span_y = 1.0 / diag_factor

    # Establish the linspace boundaries
    left_x = -span_x * (width - 1) / width
    right_x = span_x * (width - 1) / width
    top_y = -span_y * (height - 1) / height
    bottom_y = span_y * (height - 1) / height

    # Generate 1D coordinates
    x_coords = torch.linspace(left_x, right_x, steps=width, dtype=dtype, device=device)
    y_coords = torch.linspace(top_y, bottom_y, steps=height, dtype=dtype, device=device)

    # Create 2D meshgrid (width x height) and stack into UV
    uu, vv = torch.meshgrid(x_coords, y_coords, indexing="xy")
    uv_grid = torch.stack((uu, vv), dim=-1)

    return uv_grid
