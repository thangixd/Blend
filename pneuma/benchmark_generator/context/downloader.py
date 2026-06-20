from huggingface_hub import hf_hub_download
from huggingface_hub import login

login("")  # TODO: Set HuggingFace access token

# Download meta-llama/Meta-Llama-3-8B-Instruct
files = [
    "config.json",
    "generation_config.json",
    "model-00001-of-00004.safetensors",
    "model-00002-of-00004.safetensors",
    "model-00003-of-00004.safetensors",
    "model-00004-of-00004.safetensors",
    "model.safetensors.index.json",
    "special_tokens_map.json",
    "tokenizer_config.json",
    "tokenizer.json",
]
repo_id = "meta-llama/Meta-Llama-3-8B-Instruct"
for file in files:
    hf_hub_download(repo_id=repo_id, filename=file, local_dir="./llama")
