"""
make_machine_essays.py -- step 2 of the constructed dataset: the machine-written essays.

For every selected prompt, an instruction-tuned LLM is given the SAME assignment text the
students saw (plus the source passages for source-based prompts) and asked to write the essay
as a student of the prompt's typical grade. Outputs are cleaned and saved with full
provenance (generator, temperature, top_p, seed).

Used twice in this project
  pilot (Assignment 2):  --prompts data/kaggle/train_prompts.csv --per_prompt 50 \
                         --out data/kaggle/machine_essays.csv
  v2 (final report):     --prompts data/persuade/prompts.csv --per_prompt 30 \
                         --model Qwen/Qwen2.5-1.5B-Instruct  (and HuggingFaceTB/SmolLM2-1.7B-Instruct)

Output columns: id, prompt_id, prompt_name, text, label, source_model, temperature, top_p,
seed, word_count

Manual route (text produced in a chat UI) is kept for completeness:
    python new_data/make_machine_essays.py --print_prompt 0
    python new_data/make_machine_essays.py --manual_dir data/manual --source_model_name gpt-4o
"""
import argparse
import glob
import os
import re
import sys

import pandas as pd

MIN_WORDS = 150  # shorter outputs are failed generations
TITLE_RE = re.compile(r"^(?!dear\b)[^.!?]{5,80}?:\s", re.I)   # "The Benefits of X: ..." (but not "Dear Principal:")
PLACEHOLDER_RE = re.compile(r"\[[^\]]{2,30}\]")                # "[Your Name]", "[School Name]"


def short_name(model):
    return model.split("/")[-1].lower().replace("-instruct", "")


def build_prompt(row, use_source=True, source_words=1500):
    # Remove exam-logistics sentences: they are instructions about time, not about the essay.
    instr = re.sub(r"Manage your time carefully.*?revise and edit your response\.\s*", "",
                   str(row["instructions"]), flags=re.S)
    instr = re.sub(r"Write your (essay|response) in the space provided\.", "", instr).strip()
    grade = int(row["modal_grade"]) if "modal_grade" in row and int(row["modal_grade"]) > 0 else None
    who = f"a US student in grade {grade}" if grade else "a US high school student (grades 6-12)"
    parts = [f"You are {who} completing a writing assignment."]
    src = row.get("source_text", "")
    if use_source and isinstance(src, str) and src.strip():
        parts.append("Here are the source passages you may use:\n\n" + " ".join(src.split()[:source_words]))
    parts.append("Task: " + instr)
    parts.append("Write the essay now, about 400 words, as plain paragraphs. Do not include a title, "
                 "headings, bullet points, placeholders such as [Your Name], or any commentary.")
    return "\n\n".join(parts)


def clean(text):
    """Return the cleaned essay, or None if it should be discarded."""
    text = re.sub(r"[#*_`>]+", "", text)                                   # markdown
    text = re.sub(r"^(title|essay)\s*:.*?\n", "", text.strip(), flags=re.I)  # "Title: ..." line
    text = PLACEHOLDER_RE.sub("", text)
    text = " ".join(text.split())                                          # same newline handling as run.py
    if TITLE_RE.match(text) or len(text.split()) < MIN_WORDS:
        return None
    return text


def save(rows, out):
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(out, index=False)
    return df


def run_local(args, prompts):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

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
    tag = short_name(args.model)
    for _, p in prompts.iterrows():
        pid = int(p["prompt_id"])
        have = sum(1 for r in rows if int(r["prompt_id"]) == pid)
        chat = tok.apply_chat_template([{"role": "user", "content": build_prompt(p, not args.no_source, args.source_words)}],
                                       tokenize=False, add_generation_prompt=True)
        inputs = tok(chat, return_tensors="pt").to("cuda")
        tries = 0
        while have < args.per_prompt and tries < 4 * args.per_prompt:
            n = min(args.batch, args.per_prompt - have)
            with torch.no_grad():
                out = model.generate(**inputs, do_sample=True, temperature=args.temperature,
                                     top_p=args.top_p, max_new_tokens=args.max_new_tokens,
                                     num_return_sequences=n, pad_token_id=tok.eos_token_id)
            for seq in out:
                tries += 1
                text = clean(tok.decode(seq[inputs["input_ids"].shape[1]:], skip_special_tokens=True))
                if text is None:
                    continue
                rows.append(dict(id=f"{tag}_p{pid:02d}_{have:03d}", prompt_id=pid, prompt_name=p["prompt_name"],
                                 text=text, label="machine", source_model=args.model,
                                 temperature=args.temperature, top_p=args.top_p, seed=args.seed,
                                 word_count=len(text.split())))
                have += 1
            save(rows, args.out)   # save after every batch so a crash loses nothing
        print(f"  prompt {pid} ({p['prompt_name']}): {have}/{args.per_prompt} kept from {tries} generations")
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
        if text is None:
            print("rejected by cleaning:", path)
            continue
        rows.append(dict(id=f"manual_p{pid:02d}_{len(rows):03d}", prompt_id=pid, prompt_name=names[pid], text=text,
                         label="machine", source_model=args.source_model_name, temperature="", top_p="",
                         seed="", word_count=len(text.split())))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", default="data/persuade/prompts.csv")
    ap.add_argument("--out", default=None, help="default: data/persuade/machine_<model>.csv")
    ap.add_argument("--per_prompt", type=int, default=30)
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--cache_dir", default="../cache")
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--max_new_tokens", type=int, default=600)
    ap.add_argument("--batch", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--source_words", type=int, default=1500)
    ap.add_argument("--no_source", action="store_true", help="do not show the source passages")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--print_prompt", type=int, default=None, help="print the prompt for this prompt_id and exit")
    ap.add_argument("--manual_dir", default=None)
    ap.add_argument("--source_model_name", default="manual-chat-model")
    args = ap.parse_args()
    if args.out is None:
        args.out = f"data/persuade/machine_{short_name(args.model)}.csv"

    prompts = pd.read_csv(args.prompts).fillna({"source_text": ""})
    if "selected" in prompts.columns:
        prompts = prompts[prompts["selected"]]
    if args.print_prompt is not None:
        print(build_prompt(prompts[prompts["prompt_id"] == args.print_prompt].iloc[0], not args.no_source, args.source_words))
        return

    rows = run_manual(args, prompts) if args.manual_dir else run_local(args, prompts)
    df = save(rows, args.out)
    print(f"\nSaved {len(df)} machine essays to {args.out}")
    print(df.groupby("prompt_id")["word_count"].describe()[["count", "mean", "min", "max"]])


if __name__ == "__main__":
    main()
