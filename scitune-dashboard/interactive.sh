#!/bin/bash
docker run --rm -it --gpus all --network host -v "/home/ubuntu/scitune_data":"/opt/scitune/data" --entrypoint bash -e DATA_DIR="/opt/scitune/data" dashboard_test