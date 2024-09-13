#!/bin/bash
CONTROLLER_PORT=${CONTROLLER_PORT:-10000}
WORKER_PORT=${WORKER_PORT:-40000}
DATA_DIR=${DATA_DIR:-/tmp/data}
MODEL=${MODEL:-scienceqa/scitune-llava-checkpoint-4500}
SCIENCE_TASK=scienceqa
python -m llava.serve.model_worker --host 0.0.0.0 --controller http://localhost:$CONTROLLER_PORT --port $WORKER_PORT --worker http://localhost:$WORKER_PORT --model-path $DATA_DIR/models/$MODEL --science-task $SCIENCE_TASK --load-4bit



