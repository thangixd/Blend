import os

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import torch

from utils.pipeline_initializer import initialize_pipeline
from utils.generators import further_rephrase_bx2_questions

pipe = initialize_pipeline("../../models/llama", torch.bfloat16)

# Specific setting for Llama-3-8B-Instruct for batching
pipe.tokenizer.pad_token_id = pipe.model.config.eos_token_id
pipe.tokenizer.padding_side = "left"

# Adjust names and sources
dataset_names = ["chembl", "adventure", "public", "chicago", "fetaqa"]

for name in dataset_names:
    further_rephrase_bx2_questions(
        name,
        pipe
    )
