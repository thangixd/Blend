import json
import os


def main():
    results_path = "Mixed/"
    batch_names = os.listdir(results_path)
    batch_names.sort(key=lambda x: (len(x), x))

    all_scores = {}

    for batch in batch_names:
        batch_score = {
            "data_ingestion": [],
            "generate_index": [],
            "query_index": [],
        }
        scores_path = os.listdir(os.path.join(results_path, batch))
        for score_path in scores_path:
            with open(os.path.join(results_path, batch, score_path)) as f:
                score = json.load(f)["results"]
                if "read_tables" in score and "summarize" in score:
                    batch_score["data_ingestion"].append(
                        score["read_tables"]["time"] + score["summarize"]["time"]
                    )
                if "generate_index" in score:
                    batch_score["generate_index"].append(
                        score["generate_index"]["time"]
                    )
                if "query_index" in score:
                    batch_score["query_index"].append(score["query_index"]["time"])

        for key in batch_score:
            batch_score[key].sort()
        batch_score["query_index"] = batch_score["query_index"][:10]
        all_scores[batch] = batch_score

    for key in all_scores:
        print(key)
        print("Ingestion")
        print(all_scores[key]["data_ingestion"][0])
        print("Generate index")
        print(all_scores[key]["generate_index"][0])
        print("Query index")
        for score in all_scores[key]["query_index"]:
            print(score)
        print("====================================")

    # print(json.dumps(all_scores, indent=4))


if __name__ == "__main__":
    main()
