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

# # Exit on errors
set -o errexit

EXPERIMENT_NAME="cifar10_cond_baseline"
CHECKPOINT_STEP="300000"

echo "Checkpoint: $EXPERIMENT_NAME"
echo "Checkpoint step: $CHECKPOINT_STEP"

SAMPLING_STEPS=(2 3 4 5 10 30 50 100 250 500 1000)
CFG_SCALES=(0.0)
SAMPLING_MODE="DDIM"
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
            N_STEPS_FORMATTED="ddim$N_STEPS"
        fi
        uv run python \
            scripts/image_sample.py \
            --data_dir "/nfs/ghome/live/martorellat/data/cifar_train" \
            --model_path "/ceph/scratch/martorellat/guided_diffusion/blobs_$EXPERIMENT_NAME/ema_0.9999_$CHECKPOINT_STEP.pt" \
            --image_size 32 \
            --num_classes 10 \
            --num_channels 128 \
            --num_res_blocks 3 \
            --class_cond True \
            --learn_sigma True \
            --batch_size 1024 \
            --diffusion_steps 4000 \
            --noise_schedule cosine \
            --predict_xstart True \
            --use_ddim $USE_DDIM \
            --timestep_respacing $N_STEPS_FORMATTED \
            --num_samples $NUM_FID_SAMPLES \
            --exp_name $EXPERIMENT_NAME

        # Important to cd because we have another venv inside evaulations
        # and uv picks it automatically if we are in the right folder
        cd evaluations
        uv run python \
            evaluator.py \
            /nfs/ghome/live/martorellat/data/images_train.npz \
            /ceph/scratch/martorellat/guided_diffusion/samples_$EXPERIMENT_NAME/$SAMPLING_MODE-steps-$N_STEPS/samples_${NUM_FID_SAMPLES}x32x32x3.npz
        cd ..
        

    done
done

# Plot FID
uv run python \
    scripts_extra/plot_fid.py \
    --samples_dir "/ceph/scratch/martorellat/guided_diffusion/samples_$EXPERIMENT_NAME" \
    --plot_out "/ceph/scratch/martorellat/guided_diffusion/samples_$EXPERIMENT_NAME" \
    --exp_name "$EXPERIMENT_NAME" \
    --sampling_steps "${SAMPLING_STEPS[@]}" \