import time
from enum import Enum

import pandas as pd
from google.genai import types
from pydantic import BaseModel
from sklearn.model_selection import train_test_split

import random
import httpx

BASE = "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data"

PROMPT_V1 = """You classify customer messages sent to a bank's support team.
Choose the single best category for the message.
If the message is not about banking, choose out_of_scope."""


def load_data():
    """Same split as P1 (support-ticket-triage): stratified 85/15, seed 42."""
    train = pd.read_csv(f"{BASE}/train.csv")
    test = pd.read_csv(f"{BASE}/test.csv")
    train_df, val_df = train_test_split(
        train, test_size=0.15, random_state=42, stratify=train["category"]
    )
    return train_df, val_df, test


def make_schema(categories):
    """Response schema: one field whose value must be one of the categories."""
    Category = Enum("Category", {name: name for name in categories})

    class Classification(BaseModel):
        category: Category

    return Classification


def classify(client, model, text, prompt, schema):
    """Classify one message. Returns the prediction plus latency and token usage."""
    start = time.perf_counter()
    response = client.models.generate_content(
        model=model,
        contents=text,
        config=types.GenerateContentConfig(
            system_instruction=prompt,
            response_mime_type="application/json",
            response_schema=schema,
        ),
    )
    latency = time.perf_counter() - start

    usage = response.usage_metadata
    return {
        "pred": response.parsed.category.value if response.parsed else None,
        "latency_s": round(latency, 2),
        "tokens_in": usage.prompt_token_count,
        "tokens_out": usage.candidates_token_count,
    }

def build_few_shot_prompt(train_df, per_category=3, seed=42):
    """PROMPT_V1 plus a few real training examples for every category."""
    examples = train_df.groupby("category").sample(per_category, random_state=seed)

    lines = []
    for category, group in examples.groupby("category"):
        lines.append(f"- {category}: " + " | ".join(group["text"]))

    return PROMPT_V1 + "\n\nReal example messages for each category:\n" + "\n".join(lines)


def run_eval(client, model, prompt, schema, texts, results_path, requests_per_minute=15):
    """Classify every text, saving after each one. Re-running resumes; failed rows are retried."""
    from pathlib import Path
    from google.genai import errors

    results_path = Path(results_path)
    results_path.parent.mkdir(exist_ok=True)
    delay = 60 / requests_per_minute + 0.5

    rows = pd.read_csv(results_path).to_dict("records") if results_path.exists() else []
    rows = [r for r in rows if pd.notna(r.get("pred"))]   # drop failures so they get retried
    done = {r["text"] for r in rows}

    for i, text in enumerate(texts):
        if text in done:
            continue

        for attempt in range(3):
            try:
                result = classify(client, model, text, prompt, schema)
                break
            except errors.APIError as e:
                if e.code == 429 and attempt < 2:
                    wait = 30 * (attempt + 1)
                    print(f"\nRate limited, waiting {wait}s...")
                    time.sleep(wait)
                else:
                    result = {"pred": None, "error": str(e)}
                    break

        rows.append({"text": text, **result})
        pd.DataFrame(rows).to_csv(results_path, index=False)
        print(f"{i + 1}/{len(texts)}", end="\r")
        time.sleep(delay)

    print(f"\nDone: {len(rows)} results in {results_path}")
    return pd.read_csv(results_path)