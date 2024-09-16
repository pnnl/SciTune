from PIL import Image
import json
import os
import random
import pandas as pd
from functools import reduce
import re

from template import *

## Download the SciCap dataset from HuggingFace
## https://huggingface.co/datasets/CrowdAILab/scicap

data_base_dir='/rcfs/projects/steel_thread/hora620/hf/hub/datasets--CrowdAILab--scicap/snapshots/203770e81e7ff9facdd4a1b35048a3e3abf5ebcf/'
data_path='/rcfs/projects/steel_thread/hora620/hf/hub/datasets--CrowdAILab--scicap/snapshots/203770e81e7ff9facdd4a1b35048a3e3abf5ebcf/val.json'
image_folder='/rcfs/projects/steel_thread/hora620/hf/hub/datasets--CrowdAILab--scicap/snapshots/203770e81e7ff9facdd4a1b35048a3e3abf5ebcf/share-task-img-mask/arxiv/val'
list_data_dict = json.load(open(data_path, "r"))
list_data_df=pd.DataFrame(list_data_dict['images'])

print("# Images: ",len(list_data_dict['images']))
print("# Annotations: ",len(list_data_dict['annotations']))
print("Figure Types: ",list_data_df['figure_type'].unique())

target_format=[]
data_record_index=0
mac_record_index=len(list_data_dict['images'])
while data_record_index<mac_record_index:
    source=list_data_dict['images'][data_record_index]
    annot_source=list_data_dict['annotations'][data_record_index]
    ##print(source)
    input=random.choice(detail_describe_instructions)

    paragraph_text="\n".join(annot_source.get('paragraph',[]))
    mention_text = reduce(lambda a,b:a+b, annot_source.get('mention',[[]]))
    mention_text="\n".join(mention_text)

    ocr_text='\t'.join(source.get('ocr',[]))

    output=source.get('figure_type','Image')+" "+annot_source['caption_no_index']
    output+="\n"+ocr_text
    output+="\n"+paragraph_text
    output+="\n"+mention_text

    pattern = re.compile(r"\b(Fig(?:ure)?)\b", re.IGNORECASE)
    output = re.sub(pattern, r"%s"%source.get('figure_type','Image'), output)

    target_format.append({
        "id": source['id'],
        "image": source['file_name'],
        "conversations": [
            {'from': 'human', 'value': f"{input}\n<image>"},
            {'from': 'gpt', 'value': f"{output}"},
        ],
    })
    data_record_index+=1
    # if data_record_index>500:
    #     break;

print(f'Number of samples: {len(target_format)}')
with open(os.path.join(data_base_dir, f"scitune_scicap_training_{data_record_index}.json"), "w") as f:
        json.dump(target_format, f, indent=2)