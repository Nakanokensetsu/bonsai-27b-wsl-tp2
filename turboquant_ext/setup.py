# Standalone build of vllm#53060's _turboquant_C for the pip-installed vLLM 0.32.0
# (sm_120 only). Output: _turboquant_C.cpython-312-x86_64-linux-gnu.so -> copy into
# site-packages/vllm/.
import os
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

nvcc = [
    "-O3", "--use_fast_math", "-std=c++17", "--expt-relaxed-constexpr",
    "-U__CUDA_NO_HALF_OPERATORS__", "-U__CUDA_NO_HALF_CONVERSIONS__",
    "-U__CUDA_NO_BFLOAT16_CONVERSIONS__", "-U__CUDA_NO_HALF2_OPERATORS__",
    "-gencode", "arch=compute_120,code=sm_120",
    "-isystem", os.path.join(os.environ["CUDA_HOME"], "include"),  # glibc 2.43 rsqrt noexcept clash (same as plugin JIT)
    "--threads", "1", "-lineinfo", "-Xptxas", "-v",
]
setup(
    name="turboquant_c_local",
    ext_modules=[CUDAExtension(
        "_turboquant_C",
        ["main.cpp", "kernel.cu", "mma_kernel.cu"],
        extra_compile_args={"cxx": ["-O3", "-std=c++17"], "nvcc": nvcc},
    )],
    cmdclass={"build_ext": BuildExtension.with_options(use_ninja=True)},
)
