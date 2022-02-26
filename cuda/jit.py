from torch.utils.cpp_extension import load
include_dirs = ['/usr/local/magma/include']
nerf_lib = load(
    'nerf_lib',
    ['nerf_lib.cpp', 'global_to_local.cu', 'streams.cu', 'multimatmul.cu'],
    extra_include_paths=include_dirs,
    extra_cflags=['-w', '-D_GLIBCXX_USE_CXX11_ABI=0'], verbose=True)
# help(nerf_lib)
print(dir(nerf_lib))
