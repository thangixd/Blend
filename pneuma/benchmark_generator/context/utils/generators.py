import random

from tqdm import tqdm
from .csv_data_source import CsvDataSource
from .prompting_interface import prompt_pipeline
from .prompts import (
    get_generate_context_prompt,
    get_generate_bx1_prompt,
    get_generate_bx2_prompt,
    get_labeling_prompt,
    get_bx2_extra_rephrase_prompt,
)
from .jsonl import write_jsonl, read_jsonl
from transformers.pipelines.text_generation import TextGenerationPipeline


def generate_contexts(
    contexts_name: str,
    data_src_path: str,
    questions: list[str],
    pipe: TextGenerationPipeline,
    generation_params={},
):
    csv_data_source = CsvDataSource(data_src_path)
    try:
        contexts = read_jsonl(f"{contexts_name}.jsonl")
    except:
        contexts: list[dict[str, str]] = []

    for table in tqdm(iter(csv_data_source), desc="Processing tables"):
        conversations = []
        for i in tqdm(range(len(questions)), desc="Iterating questions"):
            # Skip if the contexts of a table is already generated
            if table[0] in [context["table"] for context in contexts]:
                continue
            prompt = get_generate_context_prompt(table[1], questions[i], table[2])
            conversations.append([{"role": "user", "content": prompt}])

        if len(conversations) > 0:
            outputs = prompt_pipeline(
                pipe,
                conversations,
                batch_size=2,
                context_length=8192,
                top_p=None,
                temperature=None,
                **generation_params,
            )

            for output_idx, output in enumerate(outputs):
                context = output[-1]["content"]
                contexts.append(
                    {
                        "id": f"{table[0]}_SEP_contexts-{output_idx}",
                        "table": table[0],
                        "context_question": questions[output_idx],
                        "context": context,
                    }
                )
                write_jsonl(contexts, f"{contexts_name}.jsonl")


def generate_questions(
    benchmark_name: str,
    contexts_name: str,
    pipe: TextGenerationPipeline,
    generation_params={},
):
    try:
        benchmark = read_jsonl(f"{benchmark_name}.jsonl")
    except:
        benchmark: list[dict[str, str]] = []
    contexts = read_jsonl(f"{contexts_name}.jsonl")
    context_questions = [context["context_question"] for context in contexts[:51]]
    total_sample_size = 1020
    num_samples_per_group = total_sample_size // len(context_questions)

    for question_id, context_question in enumerate(tqdm(
        context_questions, "Iterating specific context questions"
    )):
        specific_contexts = [
            context
            for context in contexts
            if context["context_question"] == context_question
        ]
        random.seed(question_id)
        sampled_contexts = random.sample(specific_contexts, k=num_samples_per_group)

        conversations_bx1 = []
        context_ids: list[str] = []
        tables: list[str] = []
        for context in sampled_contexts:
            if context["id"] in [row["context_id"] for row in benchmark]:
                break
            prompt = get_generate_bx1_prompt(context["context"])
            conversations_bx1.append([{"role": "user", "content": prompt}])
            context_ids.append(context["id"])
            tables.append(context["table"])

        if len(conversations_bx1) > 0:
            outputs = prompt_pipeline(
                pipe,
                conversations_bx1,
                batch_size=2,
                context_length=8192,
                top_p=None,
                temperature=None,
                **generation_params,
            )

            for output_idx, output in enumerate(outputs):
                question_bx1 = output[-1]["content"].split("Question: ")[-1]
                benchmark.append(
                    {
                        "context_id": context_ids[output_idx],
                        "question_bx1": question_bx1,
                        "question_bx2": "",
                        "answer_tables": [tables[output_idx]],
                    }
                )
                write_jsonl(benchmark, f"{benchmark_name}.jsonl")

        conversations_bx2: list[str] = []

        base_bx2_index = question_id*len(sampled_contexts)
        for item in benchmark[base_bx2_index:base_bx2_index+len(sampled_contexts)]:
            # Skip if the bx2 question already generated
            if len(item["question_bx2"]) > 0:
                continue
            prompt = get_generate_bx2_prompt(item["question_bx1"])
            conversations_bx2.append([{"role": "user", "content": prompt}])

        if len(conversations_bx2) > 0:
            outputs = prompt_pipeline(
                pipe,
                conversations_bx2,
                batch_size=2,
                context_length=8192,
                top_p=None,
                temperature=None,
                **generation_params,
            )

            for output_idx, output in enumerate(outputs):
                question_bx2 = output[-1]["content"]
                benchmark[base_bx2_index+output_idx]["question_bx2"] = question_bx2
                write_jsonl(benchmark, f"{benchmark_name}.jsonl")

def further_rephrase_bx2_questions(
    dataset_name: str,
    pipe: TextGenerationPipeline,
    generation_params={},
):
    print(f"Processing dataset {dataset_name}")
    benchmark = read_jsonl(
        f"../../data_src/benchmarks/context/{dataset_name}/bx_{dataset_name}.jsonl"
    )
    prompts = [
        [
            {
                "role": "user",
                "content": get_bx2_extra_rephrase_prompt(row["question_bx2"]),
            }
        ]
        for row in benchmark
    ]

    for i in tqdm(range(0, len(prompts), 2)):
        outputs = prompt_pipeline(
            pipe,
            prompts[i:i+2],
            batch_size=2,
            context_length=8192,
            top_p=None,
            temperature=None,
            **generation_params,
        )
        for output_idx, output in enumerate(outputs):
            rephrased_bx2_question = output[-1]["content"]
            benchmark[i+output_idx]["question_bx2"] = rephrased_bx2_question
            write_jsonl(benchmark, f"bx_{dataset_name}_final.jsonl")

def label_questions(
    benchmark_name: str,
    contexts_name: str,
    pipe: TextGenerationPipeline,
    generation_params={},
):
    benchmark = read_jsonl(f"{benchmark_name}.jsonl")
    contexts = read_jsonl(f"{contexts_name}.jsonl")

    for i in tqdm(range(len(benchmark)), "Processing rows of benchmark"):
        question_bx1 = benchmark[i]["question_bx1"]
        answer_tables: list[str] = benchmark[i]["answer_tables"]
        context_id = benchmark[i]["context_id"]

        context_elicitation_question = [
            context for context in contexts if context["id"] == context_id
        ][0]["context_question"]

        # Contexts that were generated from the same context elicitation question
        specific_contexts = [
            context
            for context in contexts
            if context["context_question"] == context_elicitation_question
        ]

        conversations = []
        tables: list[str] = []
        for context in tqdm(specific_contexts, "Processing specific contexts"):
            prompt = get_labeling_prompt(context["context"], question_bx1)
            conversations.append([{"role": "user", "content": prompt}])
            tables.append(context["table"])

        outputs = prompt_pipeline(
            pipe,
            conversations,
            batch_size=2,
            context_length=8192,
            max_new_tokens=4,
            top_p=None,
            temperature=None,
            **generation_params,
        )

        for output_idx, output in enumerate(outputs):
            answer = output[-1]["content"].strip().lower()
            if answer.startswith("yes") or answer.startswith("**yes"):
                answer_tables.append(tables[output_idx])

        benchmark[i]["answer_tables"] = answer_tables
        write_jsonl(benchmark, f"{benchmark_name}.jsonl")
