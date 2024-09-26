#!/bin/bash
CONDA_ENV="base"

: ${SCRIPT_PATH:="/opt/scitune/training/llava/train"}
: ${LLAMA_MODEL_DIR:="/opt/scitune/models/llama/13B"} ## Base LLaMA Model weights
: ${DATA_PATH:="/opt/scitune/dataset/scicap/scitune_instructions/scitune_scicap_train.json"}  ## Generated scitune training instructions from the scicap dataset
: ${IMAGE_FOLDER:="/opt/scitune/dataset/scicap/snapshots/203770e81e7ff9facdd4a1b35048a3e3abf5ebcf/share-task-img-mask/arxiv/train"} ## SciCap Image Folder 
: ${OUTPUT_DIR:="/opt/scitune/models/scitune-scicap"}

conda run -n ${CONDA_ENV} --no-capture-output \
    torchrun --nnodes=1 --nproc_per_node=1 --master_port=25001 \
        ${SCRIPT_PATH}/train_mem.py \
        --model_name_or_path ${LLAMA_MODEL_DIR} \
        --data_path ${DATA_PATH} \
        --image_folder ${IMAGE_FOLDER} \
        --vision_tower openai/clip-vit-large-patch14 \
        --tune_mm_mlp_adapter True \
        --mm_vision_select_layer -2 \
        --mm_use_im_start_end \
        --bf16 True \
        --output_dir ${OUTPUT_DIR} \
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
