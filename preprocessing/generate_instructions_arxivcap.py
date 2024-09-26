from PIL import Image
from tqdm import tqdm
import json
import os
import random
import pandas as pd
from datasets import load_dataset
from functools import reduce
import re
import sys 
import warnings
from template import *

## Download the ArxivCap dataset from HuggingFace
## https://huggingface.co/datasets/MMInstruction/ArxivCap

try:
    data_base_dir="/opt/scitune/dataset/arxivcap"
    ## Randomly select a subset of dataset for faster lading
    data = load_dataset(
        "parquet",
        data_files=f"{data_base_dir}/*.parquet" 
    )
except:
    warnings.warn("Dataset does not exist. Please refer to the README instructions to download the data\n", UserWarning)
    sys.exit()

list_data_dict=[]
for record in data['train']:
    for image_meta in record['caption_images']:
        image_meta['title']=record['title']
        image_meta['abstract']=record['abstract']
        image_meta['categories']=record['meta']['meta_from_kaggle']['categories']
        list_data_dict.append(image_meta)



target_format=[]
data_record_index=0
print(list_data_dict[0].keys())
mac_record_index=len(list_data_dict)

target_format=[]

os.makedirs(f"{data_base_dir}/scitune_instructions", exist_ok=True)

for idx, data_dict in tqdm(enumerate(list_data_dict), total=len(list_data_dict)):
    for sub_idx, sub_figs in enumerate(data_dict['cil_pairs']):
        out_dict = {}
        out_dict['id'] = f"{idx}_{sub_idx}"

        img_dir = sub_figs['image_file'].split("/")[0]
        os.makedirs(f"{data_base_dir}/scitune_instructions/images/{img_dir}", exist_ok=True)
        sub_figs['image'].save(f"{data_base_dir}/scitune_instructions/images/{sub_figs['image_file']}")

        out_dict['image_path'] = f"{data_base_dir}/scitune_instructions/images/{sub_figs['image_file']}"

        human_text = random.choice(detail_describe_instructions)
        if len(sub_figs['sub_caption']) > 0:
            gpt_text = sub_figs['sub_caption']
        else:
            gpt_text = data_dict['caption']

        out_dict["conversation"] = [
            {"from": "human", "value": human_text},
            {"from": "gpt", "value": gpt_text}
        ]

        target_format.append(out_dict)

        with open(f"{data_base_dir}/scitune_instructions/{idx}_{sub_idx}.json", "w") as wf:
            json.dump(out_dict, wf)

print(f'Number of samples: {len(target_format)}')
