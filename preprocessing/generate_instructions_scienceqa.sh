#!/bin/bash

python scripts/convert_sqa_to_llava \
    convert_to_llava \
    --base-dir /opt/scitune/dataset/scienceqa \
    --split $1