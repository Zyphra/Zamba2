import warnings
import os
from pathlib import Path
from setuptools import setup, find_packages
from packaging.version import parse, Version
import subprocess
import torch
from torch.utils.cpp_extension import (
    BuildExtension,
    CUDAExtension,
    CUDA_HOME,
)

# Keep only the CUDA build logic and extension setup
def get_cuda_bare_metal_version(cuda_dir):
    raw_output = subprocess.check_output(
        [cuda_dir + "/bin/nvcc", "-V"], universal_newlines=True
    )
    output = raw_output.split()
    release_idx = output.index("release") + 1
    bare_metal_version = parse(output[release_idx].split(",")[0])
    return raw_output, bare_metal_version

def check_if_cuda_home_none(global_option: str) -> None:
    if CUDA_HOME is not None:
        return
    warnings.warn(
        f"{global_option} was requested, but nvcc was not found. Are you sure your environment has nvcc available? "
        "If you're installing within a container from https://hub.docker.com/r/pytorch/pytorch, "
        "only images whose names contain 'devel' will provide nvcc."
    )

def append_nvcc_threads(nvcc_extra_args):
    return nvcc_extra_args + ["--threads", "4"]

ext_modules = []
if CUDA_HOME is not None:
    _, bare_metal_version = get_cuda_bare_metal_version(CUDA_HOME)
    if bare_metal_version < Version("11.6"):
        raise RuntimeError(
            "CUDA 11.6 or higher is required. "
            "Note: make sure nvcc has a supported version by running nvcc -V."
        )

    cc_flag = []
    cc_flag.extend([
        "-gencode", "arch=compute_70,code=sm_70",
        "-gencode", "arch=compute_80,code=sm_80"
    ])
    if bare_metal_version >= Version("11.8"):
        cc_flag.extend(["-gencode", "arch=compute_90,code=sm_90"])

    ext_modules.append(
        CUDAExtension(
            name="selective_scan_cuda",
            sources=[
                "csrc/selective_scan/selective_scan.cpp",
                "csrc/selective_scan/selective_scan_fwd_fp32.cu",
                "csrc/selective_scan/selective_scan_fwd_fp16.cu", 
                "csrc/selective_scan/selective_scan_fwd_bf16.cu",
                "csrc/selective_scan/selective_scan_bwd_fp32_real.cu",
                "csrc/selective_scan/selective_scan_bwd_fp32_complex.cu",
                "csrc/selective_scan/selective_scan_bwd_fp16_real.cu",
                "csrc/selective_scan/selective_scan_bwd_fp16_complex.cu",
                "csrc/selective_scan/selective_scan_bwd_bf16_real.cu",
                "csrc/selective_scan/selective_scan_bwd_bf16_complex.cu",
            ],
            extra_compile_args={
                "cxx": ["-O3", "-std=c++17"],
                "nvcc": append_nvcc_threads([
                    "-O3", "-std=c++17",
                    "-U__CUDA_NO_HALF_OPERATORS__",
                    "-U__CUDA_NO_HALF_CONVERSIONS__", 
                    "-U__CUDA_NO_BFLOAT16_OPERATORS__",
                    "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
                    "-U__CUDA_NO_BFLOAT162_OPERATORS__",
                    "-U__CUDA_NO_BFLOAT162_CONVERSIONS__",
                    "--expt-relaxed-constexpr",
                    "--expt-extended-lambda",
                    "--use_fast_math",
                    "--ptxas-options=-v",
                    "-lineinfo"
                ] + cc_flag),
            },
            include_dirs=[Path(os.path.dirname(os.path.abspath(__file__))) / "csrc" / "selective_scan"],
        )
    )

setup(
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension},
)
