import os

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch

from utils.pipeline_initializer import initialize_pipeline
from utils.generators import generate_contexts, generate_questions, label_questions

# Adjust names and sources
dataset_name = "chicago"
data_src_path = "../../data_src/tables/pneuma_chicago_10K"
contexts_name = f"contexts_{dataset_name}"
benchmark_name = f"bx_{dataset_name}"

pipe = initialize_pipeline("../../archive/models/llama", torch.bfloat16)

# Specific setting for Llama-3-8B-Instruct for batching
pipe.tokenizer.pad_token_id = pipe.model.config.eos_token_id
pipe.tokenizer.padding_side = "left"

with open("questions.txt") as file:
    questions = [question.strip() for question in file.readlines()]

generate_contexts(contexts_name, data_src_path, questions, pipe)
generate_questions(benchmark_name, contexts_name, pipe)
label_questions(benchmark_name, contexts_name, pipe)
