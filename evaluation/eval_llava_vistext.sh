#!/bin/bash

#SBATCH --output="slurm/llava-%j.out"
#SBATCH -A steel_thread
#SBATCH -p a100_80_shared
#SBATCH --time=04-00:00:00
#SBATCH --gres=gpu:1
#SBATCH --job-name="llava-vistext"

CONDA_ENV="llava"
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

LLAMA_MODEL_DIR="/rcfs/projects/steel_thread/models/LLAVA/13B"
LLAVA_SCITUNE_MODEL_DIR="/rcfs/projects/steel_thread/models/LLAVA-7b-pretrain-scitune-333472-v2-13B"
MM_PROJECTOR_PATH="${LLAVA_SCITUNE_MODEL_DIR}/mm_projector/checkpoint-2600.bin"

## single GPU
conda run -n ${CONDA_ENV} --no-capture-output \
    python -m llava.eval.model_vqa_science \
        --model-name ${LLAMA_MODEL_DIR} \
        --vision-tower openai/clip-vit-large-patch14 \
        --mm-projector ${MM_PROJECTOR_PATH} \
        --question-file /rcfs/projects/steel_thread/hora620/hf/hub/vistext/data/llava_vistext_validation_caption_L1_1202.json \
        --image-folder /rcfs/projects/steel_thread/hora620/hf/hub/vistext/data/images \
        --answers-file "${LLAVA_SCITUNE_MODEL_DIR}/vistext/llava_val_vistext_prediction_caption_L1_1202.jsonl" \
        --answer-prompter \
        --conv-mode simple