"""
make_machine_essays.py  --  build the machine-generated half of the new dataset.

Reads the Kaggle prompt file (train_prompts.csv), asks an LLM to write student
essays for each prompt, cleans them, and saves a labelled CSV.

Two ways to run it
  A) Local model on your GPU (default, fully scripted):
       python make_machine_essays.py --per_prompt 50
  B) Text you generated yourself in a chat model (ChatGPT / Claude / Gemini):
       python make_machine_essays.py --print_prompt 0        # copy this into the chat
       # save each reply as data\\manual\\p0_01.txt, p0_02.txt, ... p1_01.txt ...
       python make_machine_essays.py --manual_dir data\\manual --source_model_name "gpt-4o"

Output: data/kaggle/machine_essays.csv
Columns: id, prompt_id, prompt_name, text, label, source_model, temperature, word_count
"""
import argparse
import glob
import os
import re
import sys

import pandas as pd

MIN_WORDS = 150  # drop essays shorter than this (failed generations)


def build_prompt(row, use_source, source_words):
    # Remove the "time management" sentence: it is exam instructions, not part of the task.
    instr = re.sub(r"Manage your time carefully.*?revise and edit your response\.\s*", "",
                   row["instructions"], flags=re.S)
    instr = re.sub(r"Write your (essay|response) in the space provided\.", "", instr).strip()
    parts = ["You are a US high school student (grades 6-12) completing a writing test."]
    if use_source:
        src = " ".join(str(row["source_text"]).split()[:source_words])
        parts.append("Here are the source passages you may use:\n\n" + src)
    parts.append("Task: " + instr)
    parts.append("Write the essay now, about 350 words, as plain paragraphs. "
                 "Do not include a title, headings, bullet points or any commentary.")
    return "\n\n".join(parts)


def clean(text):
    text = re.sub(r"[#*_`>]+", "", text)            # strip markdown
    text = re.sub(r"^(title|essay)\s*:.*?\n", "", text.strip(), flags=re.I)
    return " ".join(text.split())                   # same newline handling as run.py


def save(rows, out):
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(out, index=False)
    return df


def run_local(args, prompts):
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    ver = tuple(int(x) for x in transformers.__version__.split(".")[:2])
    if ver < (4, 37):
        sys.exit(f"transformers {transformers.__version__} is too old for {args.model}. "
                 "Make a SEPARATE venv for generation (see the guide) so run.py is not affected.")
    if not torch.cuda.is_available():
        sys.exit("No CUDA GPU found. Use the manual route (--manual_dir) or a smaller model.")

    print(f"Loading {args.model} ...")
    tok = AutoTokenizer.from_pretrained(args.model, cache_dir=args.cache_dir)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float16, cache_dir=args.cache_dir).cuda().eval()

    rows = []
    if os.path.exists(args.out) and not args.overwrite:
        rows = pd.read_csv(args.out).to_dict("records")
        print(f"Resuming: {len(rows)} essays already saved in {args.out}")

    torch.manual_seed(args.seed)
    for _, p in prompts.iterrows():
        pid = int(p["prompt_id"])
        have = sum(1 for r in rows if int(r["prompt_id"]) == pid)
        prompt = build_prompt(p, not args.no_source, args.source_words)
        chat = tok.apply_chat_template([{"role": "user", "content": prompt}],
                                       tokenize=False, add_generation_prompt=True)
        inputs = tok(chat, return_tensors="pt").to("cuda")
        while have < args.per_prompt:
            n = min(args.batch, args.per_prompt - have)
            with torch.no_grad():
                out = model.generate(**inputs, do_sample=True, temperature=args.temperature,
                                     top_p=args.top_p, max_new_tokens=args.max_new_tokens,
                                     num_return_sequences=n, pad_token_id=tok.eos_token_id)
            for seq in out:
                text = clean(tok.decode(seq[inputs["input_ids"].shape[1]:], skip_special_tokens=True))
                if len(text.split()) < MIN_WORDS:
                    continue
                rows.append(dict(id=f"m{pid}_{have:03d}", prompt_id=pid, prompt_name=p["prompt_name"],
                                 text=text, label="machine", source_model=args.model,
                                 temperature=args.temperature, word_count=len(text.split())))
                have += 1
            save(rows, args.out)   # save after every batch so a crash loses nothing
            print(f"  prompt {pid}: {have}/{args.per_prompt}")
    return rows


def run_manual(args, prompts):
    names = dict(zip(prompts["prompt_id"], prompts["prompt_name"]))
    rows = []
    for path in sorted(glob.glob(os.path.join(args.manual_dir, "p*_*.txt"))):
        m = re.match(r"p(\d+)_", os.path.basename(path))
        if not m or int(m.group(1)) not in names:
            print("skipping", path)
            continue
        pid = int(m.group(1))
        text = clean(open(path, encoding="utf-8").read())
        if len(text.split()) < MIN_WORDS:
            print("too short, skipped:", path)
            continue
        rows.append(dict(id=f"m{pid}_{len(rows):03d}", prompt_id=pid, prompt_name=names[pid], text=text,
                         label="machine", source_model=args.source_model_name,
                         temperature="", word_count=len(text.split())))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", default="data/kaggle/train_prompts.csv")
    ap.add_argument("--out", default="data/kaggle/machine_essays.csv")
    ap.add_argument("--per_prompt", type=int, default=50)
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--cache_dir", default=None)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--max_new_tokens", type=int, default=500)
    ap.add_argument("--batch", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--source_words", type=int, default=1500)
    ap.add_argument("--no_source", action="store_true", help="do not show the source passages")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--print_prompt", type=int, default=None, help="print the prompt for this prompt_id and exit")
    ap.add_argument("--manual_dir", default=None)
    ap.add_argument("--source_model_name", default="manual-chat-model")
    args = ap.parse_args()

    prompts = pd.read_csv(args.prompts)
    if args.print_prompt is not None:
        row = prompts[prompts["prompt_id"] == args.print_prompt].iloc[0]
        print(build_prompt(row, not args.no_source, args.source_words))
        return

    rows = run_manual(args, prompts) if args.manual_dir else run_local(args, prompts)
    df = save(rows, args.out)
    print(f"\nSaved {len(df)} machine essays to {args.out}")
    print(df.groupby("prompt_id")["word_count"].describe()[["count", "mean", "min", "max"]])


if __name__ == "__main__":
    main()