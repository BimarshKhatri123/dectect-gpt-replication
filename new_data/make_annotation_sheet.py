"""
make_annotation_sheet.py -- blind human annotation of a sample of the constructed dataset.

Draws --n texts (half human, half machine, from different pairs, mixed generators), shuffles
them and writes
  data/annotation/sheet.xlsx (or .csv)   item_id, prompt_name, text, guess, quality   <- fill in
  data/annotation/key.csv                item_id, pair_id, label, generator           <- do not open
The annotator writes `guess` = H (human) or A (AI) and `quality` = 1..6 for every row.
analysis/aggregate.py then reports annotator accuracy and Cohen's kappa against the gold
labels and against the LLM judges.

    python new_data/make_annotation_sheet.py --n 40
"""
import argparse
import glob
import os

import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs_glob", default="data/new/persuade_paired_*.csv")
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--first", type=int, default=200, help="sample from the pairs that are also judged/evaluated")
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    files = sorted(glob.glob(args.pairs_glob))
    per_file = args.n // (2 * len(files))
    items = []
    for f in files:
        p = pd.read_csv(f).head(args.first).sample(n=2 * per_file, random_state=args.seed)
        for k, (_, r) in enumerate(p.iterrows()):           # alternate sides so each pair contributes one text
            lab = "human" if k % 2 == 0 else "machine"
            items.append(dict(pair_id=r.pair_id, label=lab, generator=r.generator if lab == "machine" else "",
                              prompt_name=r.prompt_name, text=r[lab]))
    df = pd.DataFrame(items).sample(frac=1, random_state=args.seed).reset_index(drop=True)
    df.insert(0, "item_id", [f"item_{i:02d}" for i in range(len(df))])

    os.makedirs("data/annotation", exist_ok=True)
    df[["item_id", "pair_id", "label", "generator"]].to_csv("data/annotation/key.csv", index=False)
    sheet = df[["item_id", "prompt_name", "text"]].assign(guess="", quality="")
    try:
        sheet.to_excel("data/annotation/sheet.xlsx", index=False)
        print("Wrote data/annotation/sheet.xlsx")
    except ImportError:
        sheet.to_csv("data/annotation/sheet.csv", index=False)
        print("Wrote data/annotation/sheet.csv")
    print(f"{len(df)} items ({(df.label == 'human').sum()} human / {(df.label == 'machine').sum()} machine)")


if __name__ == "__main__":
    main()
