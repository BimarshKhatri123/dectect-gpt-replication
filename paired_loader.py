"""
paired_loader.py -- loads pre-paired human/machine text into the
{"original": [...], "sampled": [...]} dict that run.py's detectors expect
(original = human-written, sampled = machine-generated).

Two entry points
  load_paired_csv(path)  generic: any CSV with `human` and `machine` columns, one row per
                         pair (data/hc3/hc3_paired.csv, data/new/persuade_paired_*.csv).
                         Used via  python run.py --dataset paired:<path>
  load_paired()          legacy loader for the Assignment-2 pilot (2 Kaggle prompts,
                         human = Kaggle essays, machine = Qwen2.5-1.5B essays).
                         Used via  python run.py --dataset kaggle_paired
"""
import random

import pandas as pd


def _clean(t):
    return " ".join(str(t).split())


def load_paired_csv(path):
    """Rows are already shuffled, balanced and length-matched by the builder scripts."""
    df = pd.read_csv(path)
    missing = {"human", "machine"} - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing column(s) {missing}")
    return {"original": [_clean(t) for t in df["human"]],
            "sampled": [_clean(t) for t in df["machine"]]}


def load_paired(human_csv="data/kaggle/train_essays.csv",
                machine_csv="data/kaggle/machine_essays.csv",
                max_words=250, seed=0):
    human = pd.read_csv(human_csv)
    human = human[human["generated"] == 0]
    machine = pd.read_csv(machine_csv)

    random.seed(seed)
    per_prompt = {}
    for pid in sorted(machine["prompt_id"].unique()):
        h = [_clean(t) for t in human[human["prompt_id"] == pid]["text"]]
        m = [_clean(t) for t in machine[machine["prompt_id"] == pid]["text"]]
        h = list(dict.fromkeys(h))
        m = list(dict.fromkeys(m))
        random.shuffle(h)
        random.shuffle(m)
        k = min(len(h), len(m))                 # balanced classes within each prompt
        per_prompt[pid] = list(zip(h[:k], m[:k]))

    # interleave prompts so that the first n_samples pairs cover every prompt
    pairs = []
    longest = max(len(v) for v in per_prompt.values())
    for i in range(longest):
        for pid in per_prompt:
            if i < len(per_prompt[pid]):
                pairs.append(per_prompt[pid][i])

    original, sampled = [], []
    for h, m in pairs:
        hw, mw = h.split(), m.split()
        n = min(len(hw), len(mw), max_words)    # same length for both: no length shortcut
        original.append(" ".join(hw[:n]))
        sampled.append(" ".join(mw[:n]))
    return {"original": original, "sampled": sampled}
