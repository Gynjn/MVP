import os
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

sources = [
    os.path.join("csrc", "bindings.cpp"),
    os.path.join("csrc", "spherical_harmonics_kernel.cu")
]

extra_compile_args = {
    "cxx": ["-O3"],
    "nvcc": ["-O3", "--use_fast_math"]
}

include_dirs = [
    os.path.join(ROOT_DIR, "csrc"),
    os.path.join(ROOT_DIR, "csrc", "third_party", "glm") 
]

setup(
    name="mvp_cuda", 
    version="0.1.0",
    author="Hyungbin Cho",
    description="CUDA optimization for MVP Opacity SH calculation",
    
    ext_modules=[
        CUDAExtension(
            name="mvp_cuda._C", 
            sources=sources,
            include_dirs=include_dirs,
            extra_compile_args=extra_compile_args,
        )
    ],
    
    cmdclass={
        "build_ext": BuildExtension
    }
)