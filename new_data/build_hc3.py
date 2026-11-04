"""
build_hc3.py -- existing new dataset #1: HC3 (Guo et al., 2023), human vs ChatGPT answers
to the SAME question, in five domains. https://huggingface.co/datasets/Hello-SimpleAI/HC3

Output data/hc3/hc3_paired.csv  (pair_id, domain, question, human, machine, n_words)
One human answer and one ChatGPT answer per question, both normalised and cut to the same
length, domains interleaved so any prefix of the file is domain-balanced.

    python new_data/build_hc3.py --cache_dir ../cache
"""
import argparse
import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from common import interleave, normalise, trim_pair  # noqa: E402

DOMAINS = ["wiki_csai", "open_qa", "finance", "medicine", "reddit_eli5"]
# ChatGPT answers that are refusals / meta-comments are not useful "machine text"
BAD_PHRASES = ["as an ai", "language model", "i'm sorry", "i am sorry", "i'm not able", "i cannot"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache_dir", default="../cache")
    ap.add_argument("--out", default="data/hc3/hc3_paired.csv")
    ap.add_argument("--min_words", type=int, default=100, help="both answers need at least this many words")
    ap.add_argument("--max_words", type=int, default=250)
    ap.add_argument("--per_domain", type=int, default=200)
    ap.add_argument("--domains", default=",".join(DOMAINS))
    args = ap.parse_args()

    from huggingface_hub import hf_hub_download
    groups, stats = {}, []
    for dom in args.domains.split(","):
        path = hf_hub_download("Hello-SimpleAI/HC3", f"{dom}.jsonl", repo_type="dataset", cache_dir=args.cache_dir)
        rows, n_q = [], 0
        with open(path, encoding="utf-8") as f:
            for line in f:
                ex = json.loads(line)
                n_q += 1
                hs = [normalise(h) for h in ex.get("human_answers", []) if h]
                ms = [normalise(m) for m in ex.get("chatgpt_answers", []) if m]
                hs = [h for h in hs if len(h.split()) >= args.min_words]
                ms = [m for m in ms if len(m.split()) >= args.min_words
                      and not any(p in m.lower() for p in BAD_PHRASES)]
                if not hs or not ms:
                    continue
                h, m, n = trim_pair(hs[0], ms[0], args.max_words)
                rows.append(dict(domain=dom, question=normalise(ex["question"]), human=h, machine=m, n_words=n))
        groups[dom] = rows
        stats.append((dom, n_q, len(rows)))

    # shuffle within each domain, interleave, then keep at most per_domain pairs per domain
    pairs = interleave(groups, seed=0)
    counts, kept = {}, []
    for r in pairs:
        if counts.get(r["domain"], 0) < args.per_domain:
            counts[r["domain"]] = counts.get(r["domain"], 0) + 1
            kept.append(r)
    df = pd.DataFrame(kept)
    df.insert(0, "pair_id", [f"hc3_{i:04d}" for i in range(len(df))])
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df.to_csv(args.out, index=False)

    print("domain        questions  usable_pairs  kept")
    for dom, n_q, n_ok in stats:
        print(f"{dom:13s} {n_q:9d} {n_ok:13d} {counts.get(dom, 0):5d}")
    print(f"Saved {len(df)} pairs to {args.out}; mean words {df.n_words.mean():.0f}")


if __name__ == "__main__":
    main()
