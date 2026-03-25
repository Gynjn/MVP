#pragma once

#include <torch/extension.h> 
#include <vector>

namespace at {
class Tensor;
}

namespace mvp_cuda {
void launch_spherical_harmonics_opacity_fwd_kernel(
    // inputs
    const uint32_t degrees_to_use,
    const at::Tensor dirs,                // [..., 3]
    const at::Tensor coeffs,              // [..., K, 1]
    // outputs
    at::Tensor opacities // [..., 1]
);

void launch_spherical_harmonics_opacity_bwd_kernel(
    // inputs
    const uint32_t degrees_to_use,
    const at::Tensor dirs,                // [..., 3]
    const at::Tensor coeffs,              // [..., K, 1]
    const at::Tensor v_opacities,            // [..., 1]
    // outputs
    at::Tensor v_coeffs,            // [..., K, 1]
    at::optional<at::Tensor> v_dirs // [..., 3]
);

} // namespace 'mvp_cuda'