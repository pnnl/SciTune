#!/bin/bash

#SBATCH --output="slurm/llava-%j.out"
#SBATCH -A steel_thread
#SBATCH -p a100_80_shared
#SBATCH --time=04-00:00:00
#SBATCH --gres=gpu:1
#SBATCH --job-name="llava-scicap-scienceqa"

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

LLAVA_SCITUNE_MODEL_DIR="/rcfs/projects/steel_thread/models/LLAVA-2-pretrain-scitune-333472-v2-13B"
LLAVA_SCITUNE_SCIENCEQA_MODEL_DIR="${LLAVA_SCITUNE_MODEL_DIR}/scienceqa_v0/checkpoint-2000/"

## single GPU
conda run -n ${CONDA_ENV} --no-capture-output \
    python -m llava.eval.model_vqa_science \
        --model-name ${LLAVA_SCITUNE_SCIENCEQA_MODEL_DIR} \
        --question-file /qfs/projects/steel_thread/hora620/DevHub/scientific-instruction-tuning/ScienceQA/data/scienceqa/llava_test_QCM-LEPA.json \
        --image-folder /qfs/projects/steel_thread/hora620/DevHub/scientific-instruction-tuning/ScienceQA/data/scienceqa/images/test \
        --answers-file "${LLAVA_SCITUNE_MODEL_DIR}/scienceqa_v0/llava_test_QCM-LEPA_prediction.jsonl" \
        --answer-prompter \
        --conv-mode simple









# ## multiple GPUs
# CHUNKS=8
# for IDX in {0..7}; do
#     CUDA_VISIBLE_DEVICES=$IDX
#     conda run -n ${CONDA_ENV} --no-capture-output \
#         python -m llava.eval.model_vqa_science \
#             --model-name ${LLAVA_SCITUNE_SCIENCEQA_MODEL_DIR} \
#             --question-file /qfs/projects/steel_thread/hora620/DevHub/scientific-instruction-tuning/ScienceQA/data/scienceqa/llava_test_QCM-LEPA.json \
#             --image-folder /qfs/projects/steel_thread/hora620/DevHub/scientific-instruction-tuning/ScienceQA/data/scienceqa/images/test \
#             --answers-file "${LLAVA_SCITUNE_MODEL_DIR}/scienceqa/llava_test_QCM-LEPA_prediction-chunk${CHUNKS}_${IDX}.jsonl" \
#             --num-chunks $CHUNKS \
#             --chunk-idx $IDX \
#             --answer-prompter \
#             --conv-mode simple &
# done

# output_file="${LLAVA_SCITUNE_MODEL_DIR}/scienceqa/llava_test_QCM-LEPA_prediction.jsonl"
# for idx in $(seq 0 $((CHUNKS-1))); do
#   cat "${LLAVA_SCITUNE_MODEL_DIR}/scienceqa/llava_test_QCM-LEPA_prediction-chunk${idx}.jsonl" >> "$output_file"
# done