#!/bin/bash

CONDA_ENV="base"
: ${LLAMA_MODEL_DIR:="/opt/scitune/models/llama/13B"}
: ${MM_PROJECTOR_PATH:="/opt/scitune/models/scitune-scicap/mm_projector/checkpoint-2600.bin"} 
: ${QUESTION_FILE:="/opt/scitune/dataset/mathvista/scitune_instructions/scitune_mathvista_validation.json"}  ## Generated scitune validation instructions from the mathvista dataset
: ${IMAGE_FOLDER:="/opt/scitune/dataset/mathvista/images/"}
: ${ANSWERS_FILE:="/opt/scitune/dataset/mathvista/scitune_mathvista_prediction.jsonl"}

## single GPU
conda run -n ${CONDA_ENV} --no-capture-output \
    python -m llava.eval.model_vqa_science \
        --model-name ${LLAMA_MODEL_DIR}\
        --vision-tower openai/clip-vit-large-patch14 \
        --mm-projector ${MM_PROJECTOR_PATH} \
        --question-file ${QUESTION_FILE} \
        --image-folder ${IMAGE_FOLDER}\
        --answers-file ${ANSWERS_FILE} \
        --answer-prompter \
        --conv-mode simple
