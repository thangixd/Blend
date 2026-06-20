import os
import gdown
import urllib.request
from tqdm import tqdm

script_dir = os.path.dirname(os.path.abspath(__file__))

# Download content benchmarks
url = "https://drive.google.com/drive/folders/1Cg69BWaD2vjLsvzNvOUAhqUMN4DFF6hy"
gdown.download_folder(url, output=os.path.join(script_dir, "content"))

# Download context benchmarks
contexts = [
    "pneuma_chicago_10K_questions_annotated.jsonl",
    "pneuma_public_bi_questions_annotated.jsonl",
    "pneuma_chembl_10K_questions_annotated.jsonl",
    "pneuma_fetaqa_questions_annotated.jsonl",
    "pneuma_adventure_works_questions_annotated.jsonl",
]

for context in tqdm(contexts):
    urllib.request.urlretrieve(
        f"https://storage.googleapis.com/pneuma_open/{context}",
        filename=os.path.join(script_dir, f"context/{context}"),
    )
