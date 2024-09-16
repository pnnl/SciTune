#!/bin/bash

#SBATCH --output="slurm/llava-%j.out"
#SBATCH -A steel_thread
#SBATCH -p a100_shared
#SBATCH --time=04-00:00:00
#SBATCH --gres=gpu:1
#SBATCH --job-name="llava-scicap"

CONDA_ENV="llava-v2"
LLaVA_DIR="/qfs/projects/steel_thread/hora620/DevHub/scientific-instruction-tuning/LLaVA"
HF_DIR="/qfs/projects/steel_thread/hora620/DevHub/scientific-instruction-tuning/transformers"

export HF_HOME="/rcfs/projects/steel_thread/hora620/hf"
export TORCH_EXTENSIONS_DIR="/rcfs/projects/steel_thread/hora620/pytorch"

#SCRIPT_PATH="llava/train"
SCRIPT_PATH="../LLaVA/llava/train"

if [[ -n "${SLURM_JOB_ID}" ]]; then
    echo "HOST: $(hostname -i)"
    echo "USER: ${USER}"
    
    module purge
    module load gcc/9.1.0
    module load openmpi/4.1.1
    module load cuda/11.1
    module load python/miniconda3.9
	source /share/apps/python/miniconda3.9/etc/profile.d/conda.sh
    
    module list
    nvidia-smi
fi
conda run -n ${CONDA_ENV} --no-capture-output \
    torchrun --nnodes=1 --nproc_per_node=1 --master_port=25001 \
        ${SCRIPT_PATH}/train_mem.py \
        --model_name_or_path /rcfs/projects/steel_thread/models/LLAVA-2/13B \
        --data_path /rcfs/projects/steel_thread/hora620/hf/hub/datasets--CrowdAILab--scicap/snapshots/203770e81e7ff9facdd4a1b35048a3e3abf5ebcf/llava_scicap_sample_333472_v2.json \
        --image_folder /rcfs/projects/steel_thread/hora620/hf/hub/datasets--CrowdAILab--scicap/snapshots/203770e81e7ff9facdd4a1b35048a3e3abf5ebcf/share-task-img-mask/arxiv/train \
        --vision_tower openai/clip-vit-large-patch14 \
        --tune_mm_mlp_adapter True \
        --mm_vision_select_layer -2 \
        --mm_use_im_start_end \
        --bf16 True \
        --output_dir /rcfs/projects/steel_thread/models/LLAVA-3-pretrain-scitune-333472-v2-13B \
        --num_train_epochs 1 \
        --per_device_train_batch_size 8 \
        --per_device_eval_batch_size 4 \
        --gradient_accumulation_steps 2 \
        --evaluation_strategy "no" \
        --save_strategy "steps" \
        --save_steps 100 \
        --save_total_limit 1 \
        --learning_rate 2e-3 \
        --weight_decay 0. \
        --warmup_ratio 0.03 \
        --lr_scheduler_type "cosine" \
        --logging_steps 1 \
        --tf32 True \
        --model_max_length 2048 \
        --gradient_checkpointing True \
        --lazy_preprocess True \
        --report_to wandb