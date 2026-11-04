"""
build_persuade.py -- step 1 of the constructed dataset (v2): prompts + human essays.

Source: PERSUADE 2.0 corpus (Crossley et al., 2024), ~25k argumentative essays written by
US students in grades 6-12 for 15 prompts, each with an expert holistic score (1-6).
https://github.com/scrosseye/persuade_corpus_2.0   (download the "Training set" CSV into data/persuade/)

Outputs
  data/persuade/prompts.csv       one row per prompt: prompt_id, prompt_name, task, instructions,
                                  source_text, modal_grade, n_essays, selected
  data/persuade/human_sample.csv  --per_prompt human essays per selected prompt
                                  (essay_id, prompt_id, prompt_name, task, holistic_score,
                                   grade_level, word_count, text)

Prompt selection: every "Independent" prompt, plus the source-based prompts whose source
passages we have (the two Kaggle prompts in data/kaggle/train_prompts.csv). Source-based
prompts without their source articles are excluded: an LLM would have to invent the article
content, which would make the machine essays systematically different from the human ones.

Ethics / data minimisation: demographic columns (gender, race/ethnicity, ELL, disability,
economic disadvantage) are NOT copied into any output file.

    python new_data/build_persuade.py --per_prompt 60
"""
import argparse
import glob
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from common import normalise  # noqa: E402

KEEP = ["essay_id", "prompt_id", "prompt_name", "task", "holistic_score", "grade_level", "word_count", "text"]


def find_col(df, *names):
    for n in names:
        if n in df.columns:
            return n
    raise KeyError(f"none of {names} in columns {list(df.columns)}")


def load_persuade(path):
    df = pd.read_csv(path, low_memory=False)
    id_col = find_col(df, "essay_id_comp", "essay_id")
    df = df.drop_duplicates(subset=[id_col]).copy()     # discourse-level file -> one row per essay
    out = pd.DataFrame({
        "essay_id": df[id_col],
        "prompt_name": df[find_col(df, "prompt_name")].str.strip(),
        "task": df[find_col(df, "task")].str.strip(),
        "assignment": df[find_col(df, "assignment")],
        "holistic_score": df[find_col(df, "holistic_essay_score", "holistic_score")],
        "grade_level": pd.to_numeric(df.get("grade_level"), errors="coerce"),
        "text": df[find_col(df, "full_text", "text")].map(normalise),
    })
    out["word_count"] = out["text"].str.split().str.len()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--persuade_csv", default=None, help="default: the first *.csv in data/persuade/")
    ap.add_argument("--kaggle_prompts", default="data/kaggle/train_prompts.csv")
    ap.add_argument("--per_prompt", type=int, default=60, help="human essays sampled per prompt")
    ap.add_argument("--min_words", type=int, default=150, help="human essays shorter than this are not sampled")
    ap.add_argument("--include_unsourced", action="store_true", help="also keep source-based prompts without sources")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    path = args.persuade_csv or sorted(p for p in glob.glob("data/persuade/*.csv")
                                       if not os.path.basename(p) in ("prompts.csv", "human_sample.csv"))[0]
    print("Reading", path)
    essays = load_persuade(path)
    print(f"{len(essays)} unique essays, {essays.prompt_name.nunique()} prompts")

    kp = pd.read_csv(args.kaggle_prompts)
    sources = dict(zip(kp.prompt_name.str.strip(), kp.source_text))

    rows = []
    for i, (name, g) in enumerate(sorted(essays.groupby("prompt_name"), key=lambda x: x[0])):
        task = g.task.mode()[0]
        has_src = name in sources
        rows.append(dict(prompt_id=i, prompt_name=name, task=task,
                         instructions=g.assignment.mode()[0],
                         source_text=sources.get(name, ""),
                         modal_grade=int(g.grade_level.mode()[0]) if g.grade_level.notna().any() else -1,
                         n_essays=len(g),
                         selected=(task.lower().startswith("indep") or has_src or args.include_unsourced)))
    prompts = pd.DataFrame(rows)
    os.makedirs("data/persuade", exist_ok=True)
    prompts.to_csv("data/persuade/prompts.csv", index=False)
    print(prompts[["prompt_id", "prompt_name", "task", "modal_grade", "n_essays", "selected"]].to_string(index=False))

    sel = prompts[prompts.selected]
    pid = dict(zip(sel.prompt_name, sel.prompt_id))
    pool = essays[essays.prompt_name.isin(pid) & (essays.word_count >= args.min_words)].copy()
    pool["prompt_id"] = pool.prompt_name.map(pid)
    sample = pd.concat([g.sample(n=min(args.per_prompt, len(g)), random_state=args.seed)
                        for _, g in pool.groupby("prompt_id")])
    sample[KEEP].to_csv("data/persuade/human_sample.csv", index=False)
    print(f"\nSampled {len(sample)} human essays from {len(sel)} prompts -> data/persuade/human_sample.csv")
    print(sample.groupby("prompt_name").holistic_score.describe()[["count", "mean"]].round(2))


if __name__ == "__main__":
    main()
