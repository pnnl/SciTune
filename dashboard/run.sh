#!/bin/bash
CONTROLLER_PORT=${CONTROLLER_PORT:-10000}
python -m llava.serve.gradio_web_server --controller http://localhost:$CONTROLLER_PORT --model-list-mode reload