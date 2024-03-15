import paddle
from paddle.utils.cpp_extension import CppExtension, CUDAExtension, setup


if __name__ == "__main__":
    sources = [
        "src/similar.cc",
        "src/similar.cu",
        "src/weighting.cc",
        "src/weighting.cu",
    ]
    flags = None

    if paddle.device.is_compiled_with_cuda():
        extension = CUDAExtension
        flags = {
            "cxx": ["-DPADDLE_WITH_CUDA"],
            "nvcc": ["-arch=sm_80"],
        }
    else:
        sources = filter(lambda x: x.endswith("cu"), sources)
        extension = CppExtension

    extension = extension(sources=sources, extra_compile_args=flags)
    setup(name="local_attn", ext_modules=extension)
