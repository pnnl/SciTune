#!/bin/bash

CONDA_ENV="base"

: ${MM_PROJECTOR_PATH:="/opt/scitune/models/LLAVA-3-pretrain-scitune-333472-v2-13B/mm_projector/checkpoint-2600.bin"}
: ${LLAVA_SCITUNE_SCIENCEQA_MODEL_DIR:="/opt/scitune/models/scienceqa_v0"}
mkdir -p ${LLAVA_SCITUNE_SCIENCEQA_MODEL_DIR}

: ${SCRIPT_PATH:="/opt/scitune/training/llava/train"}
: ${PROMPT_VERSION:="v0"}
: ${MODEL_PATH:="/opt/scitune/models/llama/13B"}
: ${DATA_PATH:="/opt/scitune/dataset/llava_train_QCM-LEPA.json"} # Write instaructions to download the dataset in the README.md (Get instructions from Sameera)
: ${IMAGE_FOLDER:="/opt/scitune/dataset/train"} # Write instaructions to download the dataset in the README.md step (Get instructions from Sameera)

conda run -n ${CONDA_ENV} --no-capture-output \
    torchrun --nnodes=1 --nproc_per_node=1 --master_port=25001 \
        ${SCRIPT_PATH}/train_mem.py \
        --model_name_or_path ${MODEL_PATH} \
        --version ${PROMPT_VERSION} \
        --data_path ${DATA_PATH} \
        --image_folder ${IMAGE_FOLDER} \
        --vision_tower openai/clip-vit-large-patch14 \
        --pretrain_mm_mlp_adapter ${MM_PROJECTOR_PATH} \
        --mm_vision_select_layer -2 \
        --bf16 True \
        --output_dir ${LLAVA_SCITUNE_SCIENCEQA_MODEL_DIR} \
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

