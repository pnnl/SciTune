from qafacteval import QAFactEval
import os, json
import pandas as pd

data_path="/rcfs/projects/steel_thread/models/LLAVA-7b-pretrain-scitune-333472-v2-13B/scicap/llava_val_scicap_prediction_501_v2.jsonl"
scitune_preds=pd.read_json(data_path,lines=True)

data_path='/rcfs/projects/steel_thread/hora620/hf/hub/datasets--CrowdAILab--scicap/snapshots/203770e81e7ff9facdd4a1b35048a3e3abf5ebcf/'
with open(os.path.join(data_path, f"llava_scicap_val_501_v2.json"), "r") as f:
    gt_scicap=json.load(f)
gt_scicap=pd.DataFrame(gt_scicap)

scitune_preds_with_gt_scicap=pd.merge(scitune_preds,gt_scicap,left_on='question_id',right_on='id')
scitune_preds_with_gt_scicap['gt_text']=scitune_preds_with_gt_scicap['conversations'].apply(lambda x: x[1]['value'])
scitune_preds_with_gt_scicap['scitune_text']=scitune_preds_with_gt_scicap['text'].apply(lambda x:"\n".join(x.split('\n')[1:]))
scitune_preds_with_gt_scicap['scitune_gt_text']=scitune_preds_with_gt_scicap['gt_text'].apply(lambda x:"\n".join(x.split('\n')[1:]))

kwargs = {"cuda_device": -1, "use_lerc_quip": True, \
        "verbose": True, "generation_batch_size": 32, \
        "answering_batch_size": 32, "lerc_batch_size": 8}

model_folder = "/qfs/projects/steel_thread/hora620/DevHub/scientific-instruction-tuning/LLaVA/QAFactEval/models" # path to models downloaded with download_models.sh
metric = QAFactEval(
    lerc_quip_path=f"{model_folder}/quip-512-mocha",
    generation_model_path=f"{model_folder}/generation/model.tar.gz",
    answering_model_dir=f"{model_folder}/answering",
    lerc_model_path=f"{model_folder}/lerc/model.tar.gz",
    lerc_pretrained_model_path=f"{model_folder}/lerc/pretraining.tar.gz",
    **kwargs
)

referenced_texts=scitune_preds_with_gt_scicap['scitune_text'].values
candidate_texts=scitune_preds_with_gt_scicap['scitune_gt_text'].values

print("Evaluating QAFactEval..")
all_results=[]
index=0
while index < len(referenced_texts):
      print(f'Fetching record {index}')
      referenced_text=[referenced_texts[index]]
      candidate_text=[[candidate_texts[index]]]
      print(referenced_text,candidate_text)

      referenced_text=["The ratio of the corrected flux to the original flux for the data in the top panel of Graph Plot"]
      candidate_text=[["Since j m,n increases monotonically with n, we must take n = 1 to minimize T "]]

      results = metric.score_batch_qafacteval(referenced_text, candidate_text, return_qa_pairs=True)
      all_results.append(results)
      index+=1
      break;
print("Done QAFactEval.")
# score = results[0][0]['qa-eval']['lerc_quip']
print(all_results)

# data_base_dir='/qfs/projects/steel_thread/hora620/DevHub/scientific-instruction-tuning/Grounded-Segment-Anything/results/scicap_val/'
# with open(os.path.join(data_base_dir, f"mentions_llava_scicap_val_501_v2.json"), "w") as f:
#         json.dump(results, f, indent=2)