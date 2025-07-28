#!/bin/bash
#SBATCH --job-name=sit-training
#SBATCH --partition=gpu_lowp  # Specify the partition name
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=8         # Adjust based on your needs
#SBATCH --gres=gpu:h100:2               # Number of GPUs per node
#SBATCH --mem=48G                  # Adjust based on your needs
#SBATCH --time=24:00:00            # Adjust based on your needs
#SBATCH --output=/nfs/ghome/live/martorellat/guided-diffusion/logs/%j/log.out
#SBATCH --error=/nfs/ghome/live/martorellat/guided-diffusion/logs/%j/log.err

# Exit on errors
set -o errexit

# Parse command line arguments
LOCAL_MODE=false
for arg in "$@"; do
    case $arg in
        --local)
            LOCAL_MODE=true
            shift
            ;;
        *)
            # Unknown option, ignore
            ;;
    esac
done

# Set environment variables for wandb logging (optional)
export WANDB_KEY="71b54366f0dcf364f47a59ed91fd5e5db58a0928"
export ENTITY="tommaso_research"
export PROJECT="sit_training"
export EXPERIMENT_NAME="cifar10_cond_distributional_logsnr"

export OPENAI_LOGDIR="/ceph/scratch/martorellat/guided_diffusion/logs_$EXPERIMENT_NAME"
export OPENAI_BLOBDIR="/ceph/scratch/martorellat/guided_diffusion/blobs_$EXPERIMENT_NAME"

# --- Training ---
# See the README for more options: https://github.com/willisma/SiT#training-sit [1]

DATA_PATH="/nfs/ghome/live/martorellat/data/cifar_train" # Specify the path to your ImageNet training data
POPULATION_SIZE=4

# Set number of processes per node based on local mode
if [ "$LOCAL_MODE" = true ]; then
    # Launch the training using torchrun
    uv run python -m debugpy --listen 0.0.0.0:5678 --wait-for-client ./scripts/image_train.py \
        --data_dir $DATA_PATH \
        --image_size 32 \
        --num_classes 10 \
        --num_channels 128 \
        --num_res_blocks 3 \
        --class_cond True \
        --learn_sigma True \
        --lr 1e-4 \
        --batch_size $((256 / $POPULATION_SIZE)) \
        --dropout 0.3 \
        --diffusion_steps 4000 \
        --noise_schedule cosine \
        --predict_xstart True \
        --use_distributional True \
        --distributional_loss_weighting KINGMA_SNR \
        --distributional_population_size $POPULATION_SIZE

else
    echo "Running in SLURM mode with $SLURM_GPUS_ON_NODE GPUs"
    mpiexec -n $SLURM_GPUS_ON_NODE uv run ./scripts/image_train.py \
        --data_dir $DATA_PATH \
        --image_size 32 \
        --num_classes 10 \
        --num_channels 128 \
        --num_res_blocks 3 \
        --class_cond True \
        --learn_sigma True \
        --lr 1e-4 \
        --batch_size $(((256 / $POPULATION_SIZE) / $SLURM_GPUS_ON_NODE)) \
        --dropout 0.3 \
        --diffusion_steps 4000 \
        --noise_schedule cosine \
        --predict_xstart True \
        --use_distributional True \
        --distributional_loss_weighting KINGMA_SNR \
        --distributional_population_size $POPULATION_SIZE

fi
