#!/bin/bash

CONDA_ENV="base"

: ${MM_PROJECTOR_PATH:="/opt/scitune/models/scitune-scicap/mm_projector/checkpoint-*.bin"} ## Please define the scitune-scicap checkpoint location
: ${SCITUNE_SCIENCEQA_MODEL_DIR:="/opt/scitune/models/scitune-scienceqa/"}
mkdir -p ${SCITUNE_SCIENCEQA_MODEL_DIR}

: ${SCRIPT_PATH:="/opt/scitune/training/llava/train"}
: ${PROMPT_VERSION:="v0"}
: ${LLAMA_MODEL_DIR:="/opt/scitune/models/llama/13B"} ## Base LLaMA Model weights
: ${DATA_PATH:="/opt/scitune/dataset/scienceqa/scienceqa_train_QCM-LEPA.json"} # Generated in the preprocessing stage
: ${IMAGE_FOLDER:="/opt/scitune/dataset/scienceqa/images/train"} # ScienceQA image folder

conda run -n ${CONDA_ENV} --no-capture-output \
    torchrun --nnodes=1 --nproc_per_node=1 --master_port=25001 \
        ${SCRIPT_PATH}/train_mem.py \
        --model_name_or_path ${LLAMA_MODEL_DIR} \
        --version ${PROMPT_VERSION} \
        --data_path ${DATA_PATH} \
        --image_folder ${IMAGE_FOLDER} \
        --vision_tower openai/clip-vit-large-patch14 \
        --pretrain_mm_mlp_adapter ${MM_PROJECTOR_PATH} \
        --mm_vision_select_layer -2 \
        --bf16 True \
        --output_dir ${SCITUNE_SCIENCEQA_MODEL_DIR} \
        --num_train_epochs 1 \
        --per_device_train_batch_size 4 \
        --per_device_eval_batch_size 4 \
        --gradient_accumulation_steps 2 \
        --evaluation_strategy "no" \
        --save_strategy "steps" \
        --save_steps 1 \
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

