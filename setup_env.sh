(
    module purge
    module load cuda/12.4

    uv sync --extra cu124 --extra buildtools

    VENV_PATH=$(pwd)/.venv

    VENV_CUDNN_INCLUDE_PATH="$VENV_PATH/lib/python3.10/site-packages/nvidia/cudnn/include"
    export CPLUS_INCLUDE_PATH="$VENV_CUDNN_INCLUDE_PATH${CPLUS_INCLUDE_PATH:+:$CPLUS_INCLUDE_PATH}"

    VENV_CUDNN_LIB_PATH="$VENV_PATH/lib/python3.10/site-packages/nvidia/cudnn/lib"
    PYTORCH_LIB_PATH="$VENV_PATH/lib/python3.10/site-packages/torch/lib"
    export LD_LIBRARY_PATH=$PYTORCH_LIB_PATH:$VENV_CUDNN_LIB_PATH:$LD_LIBRARY_PATH
    export CUDA_HOME="/ceph/apps/ubuntu-24/packages/cuda/12.4.0_550.54.14"

    MAX_JOBS=4 uv sync --extra cu124 --extra extcu124 --no-build-isolation
)

cd evaluations && uv sync
