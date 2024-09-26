#!/bin/bash

CONDA_ENV="base"

: ${LLAMA_MODEL_DIR:="/opt/scitune/models/llama/13B"}
: ${MM_PROJECTOR_PATH:="/opt/scitune/models/scitune-scicap/mm_projector/checkpoint-*.bin"}  ## Please define the scitune-scicap checkpoint location
: ${QUESTION_FILE:="/opt/scitune/dataset/scicap/scitune_instructions/scitune_scicap_validation.json"}  ## Generated scitune validation instructions from the scicap dataset
: ${IMAGE_FOLDER:="/opt/scitune/dataset/scicap/snapshots/203770e81e7ff9facdd4a1b35048a3e3abf5ebcf/share-task-img-mask/arxiv/val"}
: ${ANSWERS_FILE:="/opt/scitune/dataset/scicap/scitune_scicap_prediction.jsonl"}
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
