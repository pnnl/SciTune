python scripts/convert_sqa_to_llava \
    convert_to_llava \
    --base-dir /opt/scitune/dataset/scienceqa \
    --split train

python scripts/convert_sqa_to_llava \
    convert_to_llava \
    --base-dir /opt/scitune/dataset/scienceqa \
    --split val

python scripts/convert_sqa_to_llava \
    convert_to_llava \
    --base-dir /opt/scitune/dataset/scienceqa \
    --split minival

python scripts/convert_sqa_to_llava \
    convert_to_llava \
    --base-dir /opt/scitune/dataset/scienceqa \
    --split test

python scripts/convert_sqa_to_llava \
    convert_to_llava \
    --base-dir /opt/scitune/dataset/scienceqa \
    --split minitest