#include <torch/extension.h>
#include "spherical_harmonics.h"


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("spherical_harmonics_opacity_fwd", 
          &mvp_cuda::launch_spherical_harmonics_opacity_fwd_kernel);

    m.def("spherical_harmonics_opacity_bwd", 
          &mvp_cuda::launch_spherical_harmonics_opacity_bwd_kernel);
}