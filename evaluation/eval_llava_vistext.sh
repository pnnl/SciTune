#!/bin/bash


CONDA_ENV="base"

: ${LLAMA_MODEL_DIR="/opt/scitune_data/evaluation/13B"}
: ${MM_PROJECTOR_PATH="/opt/scitune_data/evaluation/mm_projector/checkpoint-2600.bin"}
: ${QUESTION_FILE:="/opt/scitune_data/evaluation/llava_vistext_validation_caption_L1_1202.json"}
: ${IMAGE_FOLDER:="/opt/scitune_data/evaluation/vistext"}
: ${ANSWERS_FILE:="/opt/scitune_data/evaluation/llava_val_vistext_prediction_caption_L1_1202.jsonl"}

## single GPU
conda run -n ${CONDA_ENV} --no-capture-output \
    python -m llava.eval.model_vqa_science \
        --model-name ${LLAMA_MODEL_DIR} \
        --vision-tower openai/clip-vit-large-patch14 \
        --mm-projector ${MM_PROJECTOR_PATH} \
        --question-file ${QUESTION_FILE} \
        --image-folder ${IMAGE_FOLDER} \
        --answers-file ${ANSWERS_FILE} \
        --answer-prompter \
        --conv-mode simple
