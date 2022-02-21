from torch.utils.cpp_extension import load
nerf_lib = load(
    'nerf_lib', ['nerf_lib.cpp', 'global_to_local.cu'],
    verbose=True, extra_cflags=['-w'])
help(nerf_lib)
