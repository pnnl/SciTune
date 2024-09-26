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



## Download the VisText dataset
## https://github.com/mitvis/vistext

data_base_dir="/opt/scitune/dataset/vistext"
data_path=f'{data_base_dir}/data/data_validation.json'

if not os.path.isfile(data_path):
    warnings.warn("Dataset does not exist. Please refer to the README instructions to download the data\n", UserWarning)
    sys.exit()
list_data_dict = json.load(open(data_path, "r"))
list_data_df = pd.DataFrame(list_data_dict['images'])

print("# Images: ",len(list_data_dict))

target_format=[]
data_record_index=0
mac_record_index=len(list_data_dict)
while data_record_index<mac_record_index:
    source=list_data_dict[data_record_index]
    input=random.choice(detail_describe_instructions)

    output=source['caption_L1']

    target_format.append({
        "id": source['caption_id'],
        "image": source['img_id']+".png",
        "conversations": [
            {'from': 'human', 'value': f"{input}\n<image>"},
            {'from': 'gpt', 'value': f"{output}"},
        ],
    })
    data_record_index+=1
    # if data_record_index>500:
    #     break;

print(f'Number of samples: {len(target_format)}')
os.makedirs(f"{data_base_dir}/scitune_instructions", exist_ok=True)
with open(os.path.join(f"{data_base_dir}/scitune_instructions", f"scitune_vistext_validation.json"), "w") as f:
        json.dump(target_format, f, indent=2)
