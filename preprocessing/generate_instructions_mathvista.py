from functools import reduce
from template import *
from PIL import Image
import pandas as pd
import warnings
import random
import json
import sys
import os
import re



## Download the MathVista dataset
## https://huggingface.co/datasets/AI4Math/MathVista

split=sys.argv[1]

data_base_dir="/opt/scitune/dataset/mathvista"
data_path=f'{data_base_dir}/data/{split}.json'

if not os.path.isfile(data_path):
    warnings.warn("Dataset does not exist. Please refer to the README instructions to download the data\n", UserWarning)
    sys.exit()
data = json.load(open(data_path, "r"))

print("# Images: ",len(data))

target_format=[]
data_record_index=0
for _index,_record in data.items():
    input_query=query_data[_index]
    output=_record['answer']
    target_format.append({
            "id": _index,
            "image": _record['image'],
            "conversations": [
                {'from': 'human', 'value': f"{input_query}\n<image>"},
                {'from': 'gpt', 'value': f"{output}"},
            ],
        })
    data_record_index+=1
    # if data_record_index>500:
    #     break;

print(f'Number of samples: {len(target_format)}')
os.makedirs(f"{data_base_dir}/scitune_instructions", exist_ok=True)
with open(os.path.join(f"{data_base_dir}/scitune_instructions", f"scitune_mathvista_{split}.json"), "w") as f:
        json.dump(target_format, f, indent=2)
