#!/bin/bash
#SBATCH --job-name=fid-evaluation
#SBATCH --partition=gpu_lowp  # Specify the partition name
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4         # Adjust based on your needs
#SBATCH --gres=gpu:h100:1               # Number of GPUs per node
#SBATCH --mem=48G                  # Adjust based on your needs
#SBATCH --time=24:00:00            # Adjust based on your needs
#SBATCH --output=/nfs/ghome/live/martorellat/guided-diffusion/logs/%j/log.out
#SBATCH --error=/nfs/ghome/live/martorellat/guided-diffusion/logs/%j/log.err

module purge
module load cuda/12.4

VENV_PATH=$(pwd)/.venv
INNER_VENV_PATH=$(pwd)/evaluations/.venv

VENV_CUDNN_INCLUDE_PATH="$VENV_PATH/lib/python3.10/site-packages/nvidia/cudnn/include"
export CPLUS_INCLUDE_PATH="$VENV_CUDNN_INCLUDE_PATH${CPLUS_INCLUDE_PATH:+:$CPLUS_INCLUDE_PATH}"

VENV_CUDNN_LIB_PATH="$VENV_PATH/lib/python3.10/site-packages/nvidia/cudnn/lib"
PYTORCH_LIB_PATH="$VENV_PATH/lib/python3.10/site-packages/torch/lib"

source $VENV_PATH/bin/activate

# Exit on errors
set -o errexit

EXPERIMENT_NAME="cifar10_cond_distributional_noweighting_without_sigma_better_channels_smaller_lr"
CHECKPOINT_STEP="300000"

echo "Checkpoint: $EXPERIMENT_NAME"
echo "Checkpoint step: $CHECKPOINT_STEP"

SAMPLING_STEPS=(5 10 20 30 50)
CFG_SCALES=(0.0)
SAMPLING_MODES=("DDIM" "iDDPM")
NUM_FID_SAMPLES=50000

export OPENAI_LOGDIR="/ceph/scratch/martorellat/guided_diffusion/logs_$EXPERIMENT_NAME"
export OPENAI_BLOBDIR="/ceph/scratch/martorellat/guided_diffusion/blobs_$EXPERIMENT_NAME"

for N_STEPS in "${SAMPLING_STEPS[@]}"
do
    for CFG_SCALE in "${CFG_SCALES[@]}"
    do
        echo "Running sampling with $N_STEPS steps and CFG scale $CFG_SCALE"
        export OPENAI_SAMPLESDIR="/ceph/scratch/martorellat/guided_diffusion/samples_$EXPERIMENT_NAME/$SAMPLING_MODE-steps-$N_STEPS"
        # Set number of processes per node based on local mode
        USE_DDIM="False"
        N_STEPS_FORMATTED=$N_STEPS
        if [ "$SAMPLING_MODE" = "DDIM" ]; then
            USE_DDIM="True"
            N_STEPS_FORMATTED="$N_STEPS"
        fi
        (
            export LD_LIBRARY_PATH=$PYTORCH_LIB_PATH:$VENV_CUDNN_LIB_PATH:$LD_LIBRARY_PATH
            export CUDA_HOME="/ceph/apps/ubuntu-24/packages/cuda/12.4.0_550.54.14"
            
            python \
                scripts/image_sample.py \
                --data_dir "/nfs/ghome/live/martorellat/data/cifar_train" \
                --model_path "/ceph/scratch/martorellat/guided_diffusion/blobs_$EXPERIMENT_NAME/ema_0.9999_$CHECKPOINT_STEP.pt" \
                --image_size 32 \
                --num_classes 10 \
                --num_channels 192 \
                --num_res_blocks 3 \
                --class_cond True \
                --batch_size 1024 \
                --diffusion_steps 4000 \
                --noise_schedule cosine \
                --use_distributional True \
                --distributional_num_eps_channels 1 \
                --num_head_channels 64 \
                --use_fp16 True \
                --use_ddim $USE_DDIM \
                --timestep_respacing $N_STEPS_FORMATTED \
                --num_samples $NUM_FID_SAMPLES \
                --exp_name $EXPERIMENT_NAME \
                --use_same_noise_in_sampling $USE_SAME_NOISE
        )

        # Important to cd because we have another venv inside evaulations
        # and uv picks it automatically if we are in the right folder
        cd evaluations
        source $INNER_VENV_PATH/bin/activate
        python \
            evaluator.py \
            /nfs/ghome/live/martorellat/data/images_train.npz \
            /ceph/scratch/martorellat/guided_diffusion/samples_$EXPERIMENT_NAME/$SAMPLING_MODE-steps-$N_STEPS/samples_${NUM_FID_SAMPLES}x32x32x3.npz
        cd ..
        source $VENV_PATH/bin/activate
        

    done
done

# Plot FID
uv run python \
    scripts_extra/plot_fid.py \
    --samples_dir "/ceph/scratch/martorellat/guided_diffusion/samples_$EXPERIMENT_NAME" \
    --plot_out "/ceph/scratch/martorellat/guided_diffusion/samples_$EXPERIMENT_NAME" \
    --exp_name "$EXPERIMENT_NAME" \
    --sampling_steps "${SAMPLING_STEPS[@]}" \
