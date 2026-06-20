# Pneuma-Retriever

To run the experiment, do the following steps:

1. Download the necessary datasets, summaries, models, and benchmarks in the `pneuma_summarizer/summaries`, `data_src`, and `models`.

2. Install the requirements with the following script:
```bash
pip install -r requirements.txt
pip install -r ../benchmark_generator/context/requirements.txt
```

3. Index the summaries using `index_vector.py` and `index_keyword.py`.

4. Produce embeddings for the summaries with `produce_question_embeddings.py`.

5. Run the benchmarks on a baseline. For example, you can run hybrid search with `nohup python -u hybrid_search.py >> hybrid_search.out &`. You may need to adjust some parts in the code (e.g., the re-ranker to be used).
