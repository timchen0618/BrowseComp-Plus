import re
import csv
from pathlib import Path


def extract_all_eval_stats(text: str):
    pattern = re.compile(
        r"""
        Processed\s+(?P<num_evaluations>\d+)\s+evaluations.*?
        Accuracy:\s+(?P<accuracy>\d+(?:\.\d+)?)%.*?
        Recall:\s+(?P<recall>\d+(?:\.\d+)?)%.*?
        Average\s+Tool\s+Calls:.*?'search':\s*(?P<num_searches>\d+(?:\.\d+)?).*?
        Summary\s+saved\s+to\s+(?P<summary_path>\S+evaluation_summary\.json)
        """,
        re.DOTALL | re.VERBOSE,
    )

    rows = []
    for m in pattern.finditer(text):
        summary_path = m.group("summary_path")
        p = Path(summary_path)
        run_name = "/".join(p.parts[-4:-1])

        rows.append({
            "run_name": run_name,
            "num_evaluations": int(m.group("num_evaluations")),
            "accuracy": float(m.group("accuracy")),
            "recall": float(m.group("recall")),
            "num_searches": float(m.group("num_searches")),
        })
    return rows


def save_to_csv(input_path: str, output_path: str):
    text = Path(input_path).read_text(encoding="utf-8")
    rows = extract_all_eval_stats(text)

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "run_name",
                "num_evaluations",
                "accuracy",
                "recall",
                "num_searches",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved {len(rows)} rows to {output_path}")


save_to_csv("sbatch_outputs/eval.out", "eval_summary.csv")