import torch
from torch import Tensor
from mvp_cuda._C import spherical_harmonics_opacity_fwd, spherical_harmonics_opacity_bwd

def spherical_harmonics_opacity(
    degrees_to_use: int,
    dirs: Tensor,  # [..., 3]
    coeffs: Tensor  # [..., K, 1]
) -> Tensor:
    """Computes spherical harmonics for opacity.

    Args:
        degrees_to_use: The degree to be used.
        dirs: Directions. [..., 3]
        coeffs: Coefficients. [..., K, 1]

    Returns:
        Spherical harmonics. [..., 1]
    """
    assert (degrees_to_use + 1) ** 2 <= coeffs.shape[-2], coeffs.shape
    batch_dims = dirs.shape[:-1]
    assert dirs.shape == batch_dims + (3,), dirs.shape
    assert (
        (len(coeffs.shape) == len(batch_dims) + 2)
        and coeffs.shape[:-2] == batch_dims
        and coeffs.shape[-1] == 1
    ), coeffs.shape

    return _SphericalHarmonicsOpacity.apply(
        degrees_to_use, dirs.contiguous(), coeffs.contiguous()
    )
    
    
class _SphericalHarmonicsOpacity(torch.autograd.Function):
    """Spherical Harmonics Opacity"""

    @staticmethod
    def forward(
        ctx, sh_degree: int, dirs: Tensor, coeffs: Tensor
    ) -> Tensor:
        
        opacities = torch.empty(
            dirs.shape[:-1] + (1,), 
            dtype=dirs.dtype, 
            device=dirs.device
        )
        
        spherical_harmonics_opacity_fwd(
            sh_degree,
            dirs,
            coeffs,
            opacities
        )
        ctx.save_for_backward(dirs, coeffs)
        ctx.sh_degree = sh_degree
        return opacities

    @staticmethod
    def backward(ctx, v_opacities: Tensor):
        dirs, coeffs = ctx.saved_tensors
        sh_degree = ctx.sh_degree
        
        v_coeffs = torch.zeros_like(coeffs)
        compute_v_dirs = ctx.needs_input_grad[1]
        v_dirs = None
        if compute_v_dirs:
            v_dirs = torch.zeros_like(dirs)
        
        spherical_harmonics_opacity_bwd(
            sh_degree,
            dirs,
            coeffs,
            v_opacities,
            v_coeffs,
            v_dirs
        )
        
        return None, v_dirs, v_coeffs