#!/bin/bash

CONDA_ENV="base"

: ${SCITUNE_SCIENCEQA_MODEL_DIR:="/opt/scitune/models/scitune-scienceqa/"} ## SciTune-ScienceQA Model
: ${QUESTION_FILE:="/opt/scitune/dataset/scienceqa/scienceqa_test_QCM-LEPA.json"}  # Generated in the preprocessing stage
: ${IMAGE_FOLDER:="/opt/scitune/dataset/scienceqa/images/test"} # ScienceQA image folder
: ${ANSWERS_FILE:="/opt/scitune/dataset/scienceqa/scitune_scienceqa_prediction.jsonl"}

## single GPU
conda run -n ${CONDA_ENV} --no-capture-output \
    python -m llava.eval.model_vqa_science \
        --model-name ${SCITUNE_SCIENCEQA_MODEL_DIR} \
        --question-file ${QUESTION_FILE} \
        --image-folder ${IMAGE_FOLDER} \
        --answers-file ${ANSWERS_FILE} \
        --answer-prompter \
        --conv-mode simple
