#!/bin/bash
CONTROLLER_PORT=${CONTROLLER_PORT:-10000}
python -m llava.serve.controller --host 0.0.0.0 --port $CONTROLLER_PORT