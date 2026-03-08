"""
Run the inference pipeline: load data, run models via Modal, save results.
Adjust hyperparameters below before running.
"""

import os
from pathlib import Path

import modal
import pandas as pd
from tqdm import tqdm

from modal_inference import app, ModelRunner


# =============================================================================
# HYPERPARAMETERS (edit these before running)
# =============================================================================
MODEL_CONFIGS = [
    ("meta-llama/Llama-3.1-8B-Instruct", "Llama"),
    ("google/gemma-7b-it", "Gemma"),
    ("Qwen/Qwen2.5-7B-Instruct", "Qwen"),
]
BATCH_SIZE = 16
MAX_TOKENS = 200
MIN_TOKENS = 100

# Paths (resolved relative to this script, not cwd)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DATA_PATH = os.path.join(PROJECT_ROOT, "Data", "nlp-queries-dataset-v3.xlsx")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
RESULTS_PATH = os.path.join(RESULTS_DIR, "results.xlsx")

# Dataset config: (base_prompt_column, sheet_name) per sheet
# Sheets: Advench, Context Shift, Polite Shift, Both Shift
DATASET_CONFIG = [
    ("goal", "case-a"),
    ("context_shifted_target", "case-b"),
    ("politeness_shifted_target", "case-c"),
    ("both_shifted_target", "case-d"),
]
LANGUAGES = ["en", "es", "zh", "ja"]


def load_data(data_path: str) -> tuple[list[pd.DataFrame], list[str], list[str]]:
    """Load Excel sheets and return (dataframes, sheet names, base columns)."""
    xl = pd.ExcelFile(data_path)
    base_cols = [c for c, _ in DATASET_CONFIG]
    names = [n for _, n in DATASET_CONFIG]
    dfs = []
    for name in names:
        if name not in xl.sheet_names:
            raise ValueError(f"Sheet '{name}' not found in {data_path}. Available: {xl.sheet_names}")
        dfs.append(pd.read_excel(data_path, sheet_name=name))
    return dfs, names, base_cols


def _prompt_col_for_lang(base_col: str, lang: str) -> str:
    """Resolve prompt column for language: base_col for en, base_col_es for es, etc."""
    if lang == "en":
        return base_col
    return f"{base_col}_{lang}"


def run_inference():
    Path(RESULTS_DIR).mkdir(parents=True, exist_ok=True)

    dfs, names, base_cols = load_data(DATA_PATH)

    modal.enable_output()
    with app.run():
        for model_id, model_name in MODEL_CONFIGS:
            print(f"\n{'='*60}")
            print(f"Model: {model_name} ({model_id})")
            print("="*60)
            runner = ModelRunner(model_name=model_id)

            for i, (df, base_col, name) in enumerate(zip(dfs, base_cols, names)):
                for lang in LANGUAGES:
                    prompt_col = _prompt_col_for_lang(base_col, lang)
                    if prompt_col not in df.columns:
                        print(f"  Warning: '{prompt_col}' not in sheet '{name}', skipping.")
                        continue

                    prompts = df[prompt_col].astype(str).tolist()
                    out_col = f"{model_name}_{lang}"
                    outputs = []

                    with tqdm(total=len(prompts), desc=f"{name} ({out_col})", unit="prompt") as pbar:
                        for j in range(0, len(prompts), BATCH_SIZE):
                            batch = prompts[j : j + BATCH_SIZE]
                            batch_out = runner.generate_batch.remote(
                                batch,
                                max_new_tokens=MAX_TOKENS,
                                min_tokens=MIN_TOKENS,
                            )
                            outputs.extend(batch_out)
                            pbar.update(len(batch))

                    dfs[i][out_col] = outputs
                    print(f"  {name} {out_col}: {len(outputs)} responses")

    with pd.ExcelWriter(RESULTS_PATH, engine="openpyxl") as writer:
        for df, name in zip(dfs, names):
            df.to_excel(writer, sheet_name=name, index=False)

    print(f"\nResults saved to {RESULTS_PATH}")


if __name__ == "__main__":
    run_inference()
