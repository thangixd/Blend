# Pneuma-Summarizer

We provide the scripts to generate all summaries used in our experiments, all of which has this naming convention: `generate_content_[].py`. You may download all the generated summaries (except for the DBReader variant) using `summaries/downloader.ipynb`. Alternatively, you can generate the summaries manually by running the following scripts (you may also costumize the arguments for these scripts):

```bash
pip install -r requirements.txt
pip install -r ../../benchmark_generator/context/requirements.txt

nohup python -u generate_content_dbreader.py >> generate_content_dbreader.out &
nohup python -u generate_content_schema_narrations.py >> generate_content_schema_narrations.out &
nohup python -u generate_content_sample_rows.py >> generate_content_sample_rows.out &
nohup python -u generate_content_schema_concat.py >> generate_content_schema_concat.out &
```

Then, run `convert_summaries_to_blocks.py` to account for the limited context window of the embedding model, which for the model we use (`BGE-BASE`) is 512 tokens.

Note: Ensure that you have already downloaded the necessary models using `../models/downloader.ipynb`.
