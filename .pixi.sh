export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export CPATH="${CONDA_PREFIX}/targets/x86_64-linux/include:${CPATH:-}"
# cuda-nvcc exports arch 10.1, which torch 2.9 rejects when JIT-building gsplat.
export TORCH_CUDA_ARCH_LIST="12.0+PTX"
