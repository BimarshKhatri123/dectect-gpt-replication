"""
pair_persuade.py -- step 3 of the constructed dataset: pair human and machine essays.

For each generator and each prompt, every machine essay is paired with a DIFFERENT human essay
written for the same prompt (human essays are split between generators so no human essay is
used twice). Both sides are cut to the same number of words (<= 250, which also keeps both
under the 512-token limit of T5), and prompts are interleaved so any prefix is prompt-balanced.

Outputs
  data/new/persuade_paired_<generator>.csv   input to run.py (--dataset paired:<file>) and llm_judge.py
      pair_id, prompt_id, prompt_name, task, generator, human_essay_id, holistic_score,
      machine_id, human, machine, n_words
  data/new/academic_essays_v2.csv            the released dataset in long format (one text per row)

    python new_data/pair_persuade.py
"""
import argparse
import glob
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from common import interleave, trim_pair  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--human", default="data/persuade/human_sample.csv")
    ap.add_argument("--machine_glob", default="data/persuade/machine_*.csv")
    ap.add_argument("--prompts", default="data/persuade/prompts.csv")
    ap.add_argument("--max_words", type=int, default=250)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    human = pd.read_csv(args.human).sample(frac=1, random_state=args.seed)
    tasks = dict(pd.read_csv(args.prompts)[["prompt_id", "task"]].values)
    machine_files = sorted(glob.glob(args.machine_glob))
    if not machine_files:
        sys.exit(f"no files match {args.machine_glob}; run make_machine_essays.py first")

    used = {}          # prompt_id -> number of human essays already consumed
    long_rows = []
    os.makedirs("data/new", exist_ok=True)
    for path in machine_files:
        mach = pd.read_csv(path).sample(frac=1, random_state=args.seed)
        gen = os.path.basename(path)[len("machine_"):-len(".csv")]
        groups = {}
        for pid, m in mach.groupby("prompt_id"):
            h = human[human.prompt_id == pid].iloc[used.get(pid, 0):]
            k = min(len(h), len(m))
            used[pid] = used.get(pid, 0) + k
            rows = []
            for (_, hr), (_, mr) in zip(h.iloc[:k].iterrows(), m.iloc[:k].iterrows()):
                ht, mt, n = trim_pair(hr.text, mr.text, args.max_words)
                rows.append(dict(prompt_id=pid, prompt_name=hr.prompt_name, task=tasks.get(pid, ""),
                                 generator=mr.source_model, human_essay_id=hr.essay_id,
                                 holistic_score=hr.holistic_score, machine_id=mr.id,
                                 human=ht, machine=mt, n_words=n))
            groups[pid] = rows
            if k < len(m):
                print(f"  WARNING {gen} prompt {pid}: only {k} human essays left for {len(m)} machine essays")
        pairs = pd.DataFrame(interleave(groups, seed=args.seed))
        pairs.insert(0, "pair_id", [f"{gen}_{i:04d}" for i in range(len(pairs))])
        out = f"data/new/persuade_paired_{gen}.csv"
        pairs.to_csv(out, index=False)
        print(f"{out}: {len(pairs)} pairs, {pairs.prompt_id.nunique()} prompts, mean {pairs.n_words.mean():.0f} words")

        for r in pairs.itertuples():
            base = dict(pair_id=r.pair_id, prompt_id=r.prompt_id, prompt_name=r.prompt_name, task=r.task, n_words=r.n_words)
            long_rows.append(dict(base, label="human", source="PERSUADE 2.0", source_id=r.human_essay_id,
                                  holistic_score=r.holistic_score, text=r.human))
            long_rows.append(dict(base, label="machine", source=r.generator, source_id=r.machine_id,
                                  holistic_score="", text=r.machine))

    release = pd.DataFrame(long_rows)
    release.to_csv("data/new/academic_essays_v2.csv", index=False)
    print(f"\nRelease file data/new/academic_essays_v2.csv: {len(release)} texts "
          f"({(release.label == 'human').sum()} human / {(release.label == 'machine').sum()} machine)")


if __name__ == "__main__":
    main()
