#!/bin/bash

#SBATCH --output="slurm/llava-%j.out"
#SBATCH -A steel_thread
#SBATCH -p a100_80
#SBATCH --time=04-00:00:00
#SBATCH --gres=gpu:8
#SBATCH --job-name="llava-scicap-scienceqa"

CONDA_ENV="llava-v2"
LLaVA_DIR="/qfs/projects/steel_thread/hora620/DevHub/scientific-instruction-tuning/LLaVA"
HF_DIR="/qfs/projects/steel_thread/hora620/DevHub/scientific-instruction-tuning/transformers"

export HF_HOME="/rcfs/projects/steel_thread/hora620/hf"
export TORCH_EXTENSIONS_DIR="/rcfs/projects/steel_thread/hora620/pytorch"

if [[ -n "${SLURM_JOB_ID}" ]]; then
    echo "HOST: $(hostname -i)"
    echo "USER: ${USER}"
    
    module purge
    module load gcc/9.1.0
    module load openmpi/4.1.1
    module load cuda/11.4
    module load python/miniconda3.9
	source /share/apps/python/miniconda3.9/etc/profile.d/conda.sh
    
    module list
    nvidia-smi
fi

# ## Extract projector features
# conda run -n ${CONDA_ENV} --no-capture-output \
#     python scripts/extract_mm_projector.py \
#     --model_name_or_path /rcfs/projects/steel_thread/models/LLAVA-7b-pretrain-scitune \
#     --output /rcfs/projects/steel_thread/models/LLAVA-7b-pretrain-scitune/mm_projector/LLAVA-7b-pretrain-scitune.bin

# conda run -n ${CONDA_ENV} --no-capture-output \

LLAVA_SCITUNE_MODEL_DIR="/rcfs/projects/steel_thread/models/LLAVA-2-pretrain-scitune-333472-v2-13B"
MM_PROJECTOR_PATH="${LLAVA_SCITUNE_MODEL_DIR}/mm_projector/checkpoint-2600.bin"
LLAVA_SCITUNE_SCIENCEQA_MODEL_DIR="${LLAVA_SCITUNE_MODEL_DIR}/scienceqa_v0"
mkdir -p ${LLAVA_SCITUNE_SCIENCEQA_MODEL_DIR}

#SCRIPT_PATH="llava/train"
SCRIPT_PATH="../LLaVA/llava/train"
# PROMPT_VERSION="llava_llama_2"
PROMPT_VERSION="v0"

conda run -n ${CONDA_ENV} --no-capture-output \
    torchrun --nnodes=1 --nproc_per_node=8 --master_port=25001 \
        ${SCRIPT_PATH}/train_mem.py \
        --model_name_or_path /rcfs/projects/steel_thread/models/LLAVA/13B \
        --version ${PROMPT_VERSION} \
        --data_path /qfs/projects/steel_thread/hora620/DevHub/scientific-instruction-tuning/ScienceQA/data/scienceqa/llava_train_QCM-LEPA.json \
        --image_folder /qfs/projects/steel_thread/hora620/DevHub/scientific-instruction-tuning/ScienceQA/data/scienceqa/images/train \
        --vision_tower openai/clip-vit-large-patch14 \
        --pretrain_mm_mlp_adapter ${MM_PROJECTOR_PATH} \
        --mm_vision_select_layer -2 \
        --bf16 True \
        --output_dir ${LLAVA_SCITUNE_SCIENCEQA_MODEL_DIR} \
        --num_train_epochs 12 \
        --per_device_train_batch_size 4 \
        --per_device_eval_batch_size 4 \
        --gradient_accumulation_steps 2 \
        --evaluation_strategy "no" \
        --save_strategy "steps" \
        --save_steps 500 \
        --save_total_limit 5 \
        --learning_rate 2e-5 \
        --weight_decay 0. \
        --warmup_ratio 0.03 \
        --lr_scheduler_type "cosine" \
        --logging_steps 1 \
        --tf32 True \
        --fsdp "full_shard auto_wrap" \
        --fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer' \
        --model_max_length 2048 \
        --gradient_checkpointing True \
        --lazy_preprocess True \
        --report_to wandb