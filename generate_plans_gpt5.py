"""
Generate planning output using OpenAI API (GPT-5).
Equivalent to portkey.py main block but uses OpenAI instead of Gemini.
"""

import csv
import json
import os
from typing import List

from openai import OpenAI

from search_agent.tongyi_utils.prompts import PROMPT_PLANNER
from tqdm import tqdm

def read_tsv(file_path: str) -> List[List[str]]:
    with open(file_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        return [row for row in reader]


def append_jsonl(file_path: str, data: List[dict]) -> None:
    with open(file_path, "a", newline="", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")


def main() -> None:
    print('api_key', os.getenv("OPENAI_API_KEY"), 'organization', os.getenv("OPENAI_ORGANIZATION"))
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"), organization=os.getenv("OPENAI_ORGANIZATION"))
    model_name = "gpt-5-mini"

    # Match portkey params
    max_tokens = 4096
    temperature = 0.7
    top_p = 0.95

    NUM_QUERIES = 2
    data = read_tsv("topics-qrels/queries_first100.tsv")
    output_file = "gpt5_plans.jsonl"

    for query in tqdm(data[:NUM_QUERIES], desc="Generating plans"):
        query_id = query[0]
        query_text = query[1]

        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": PROMPT_PLANNER},
                {"role": "user", "content": f"User Question: {query_text}"},
            ],
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
        )

        output = response.choices[0].message.content or ""

        print("-" * 100)
        print(f"Query ID: {query_id}")
        print(f"Query Text: {query_text}")
        print(f"Output: {output}")

        append_jsonl(output_file, [
            {"query_id": query_id, "query_text": query_text, "output": output}
        ])


if __name__ == "__main__":
    main()
