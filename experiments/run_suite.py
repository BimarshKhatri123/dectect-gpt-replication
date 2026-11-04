"""
run_suite.py -- runs every experiment in the final report, in named suites.

    env\\Scripts\\python experiments\\run_suite.py --suite prefetch    # download models/data (no GPU)
    env\\Scripts\\python experiments\\run_suite.py --suite smoke       # ~15 min end-to-end check
    env\\Scripts\\python experiments\\run_suite.py --suite replication # night 1  (Sec. 3 of the report)
    env\\Scripts\\python experiments\\run_suite.py --suite build_data  # PERSUADE + HC3 datasets (Sec. 4)
    env\\Scripts\\python experiments\\run_suite.py --suite new_data    # night 2  (Sec. 5)
    env\\Scripts\\python experiments\\run_suite.py --suite judge       # LLM-as-judge (Sec. 5)
    env\\Scripts\\python experiments\\run_suite.py --list              # print every command

Run from the repository root. Each step writes logs/<step>.log (UTF-8); finished steps are
recorded in logs/status.json and skipped next time (use --force to re-run), and a failing
step is logged and the suite carries on with the next one.
"""
import argparse
import json
import os
import subprocess
import sys
import time

PY = sys.executable
CACHE = os.environ.get("DETECTGPT_CACHE", "../cache")

# Paper configuration for the "GPT-2" column of Table 1 (Mitchell et al., 2023):
# GPT-2 XL (1.5B) source/scoring model, T5-3B mask filler, 2-word spans, 15% of words masked
# (pct_words_masked 0.3 * span/(span+2*buffer) = 0.3 * 2/4 = 0.15), temperature-1 sampling.
# On an 8 GB laptop GPU T5-3B runs at ~9 s per perturbed text (VRAM spills to system RAM), so
# the main runs use T5-large -- the mask filler the paper itself uses for its perturbation-count
# study (Fig. 8). T5-3B is kept for a small check (suite "t5_3b_check").
MASK = os.environ.get("DETECTGPT_MASK", "t5-large")
PAPER = ["--base_model_name", "gpt2-xl", "--mask_filling_model_name", MASK,
         "--pct_words_masked", "0.3", "--span_length", "2", "--cache_dir", CACHE,
         "--batch_size", "5", "--chunk_size", "20",
         # speed settings measured on the RTX 4060 Laptop (8 GB): both models in half precision stay
         # on the GPU together, and perturbed texts are scored 8 at a time (checked against the
         # one-at-a-time scores inside run.py)
         # (T5 stays fp32: bf16 was slower for T5 in our benchmark). Perturbations are drawn once
         # for the largest k and re-used, and cached across scoring models.
         "--base_half", "--keep_both_on_gpu", "--ll_batch_size", "8",
         "--reuse_perturbations", "--perturbation_cache", "perturbation_cache"]


def detectgpt(name, dataset, n, perts, extra=(), key=None, out="final"):
    name = f"{name}_n{n}"          # the sample size is part of the step name and results folder
    cmd = [PY, "run.py", "--output_name", f"{out}/{name}", "--dataset", dataset,
           "--n_samples", str(n), "--n_perturbation_list", perts] + PAPER
    if key:
        cmd += ["--dataset_key", key]
    return name, cmd + list(extra)


def swap(cmd, flag, value):
    """Replace the value that follows `flag` in a command list."""
    cmd = list(cmd)
    cmd[cmd.index(flag) + 1] = value
    return cmd


def suites(n_rep, n_new, n_abl):
    S = {}
    S["prefetch"] = [("prefetch", [PY, "experiments/prefetch.py", "--cache_dir", CACHE])]

    tiny = ["--n_samples", "10", "--n_perturbation_list", "2", "--batch_size", "5", "--cache_dir", CACHE]
    S["smoke"] = [
        ("smoke_paper_config_pubmed", [PY, "run.py", "--output_name", "smoke/paper_cfg", "--dataset", "pubmed",
                                       "--base_model_name", "gpt2-xl", "--mask_filling_model_name", "t5-3b", "--half"] + tiny),
        ("smoke_xsum", [PY, "run.py", "--output_name", "smoke/xsum", "--dataset", "xsum", "--dataset_key", "document",
                        "--base_model_name", "gpt2", "--mask_filling_model_name", "t5-small"] + tiny),
        ("smoke_squad", [PY, "run.py", "--output_name", "smoke/squad", "--dataset", "squad", "--dataset_key", "context",
                         "--base_model_name", "gpt2", "--mask_filling_model_name", "t5-small"] + tiny),
        ("smoke_writing", [PY, "run.py", "--output_name", "smoke/writing", "--dataset", "writing",
                           "--base_model_name", "gpt2", "--mask_filling_model_name", "t5-small"] + tiny),
        ("smoke_build_hc3", [PY, "new_data/build_hc3.py", "--cache_dir", CACHE]),
        ("smoke_hc3_paired", [PY, "run.py", "--output_name", "smoke/hc3", "--dataset", "paired:data/hc3/hc3_paired.csv",
                              "--base_model_name", "gpt2", "--mask_filling_model_name", "t5-small"] + tiny),
        ("smoke_judge_qwen", [PY, "llm_judge.py", "--pairs", "data/hc3/hc3_paired.csv", "--n", "3",
                              "--judge", "Qwen/Qwen2.5-3B-Instruct", "--out_dir", "results/smoke/llm_judge", "--cache_dir", CACHE]),
        ("smoke_judge_phi", [PY, "llm_judge.py", "--pairs", "data/hc3/hc3_paired.csv", "--n", "3",
                             "--judge", "microsoft/Phi-3.5-mini-instruct", "--out_dir", "results/smoke/llm_judge", "--cache_dir", CACHE]),
    ]

    # ---- Section 3: replication on the original datasets (night 1) -------------------
    S["replication"] = [
        detectgpt("rep_xsum", "xsum", n_rep, "1,10,100", key="document"),
        detectgpt("rep_squad", "squad", n_rep, "1,10,100", key="context"),
        detectgpt("rep_writing", "writing", n_rep, "1,10,100"),
        detectgpt("rep_pubmed", "pubmed", n_rep, "1,10,100"),
    ]
    # ablations that explain the gap between our pilot runs and the paper (XSum)
    abl = []
    for mf in ["t5-base", "t5-small"]:
        name, cmd = detectgpt(f"abl_mask_{mf}", "xsum", n_abl, "1,10,100", key="document")
        abl.append((name, swap(cmd, "--mask_filling_model_name", mf)))
    name, cmd = detectgpt("abl_base_gpt2small", "xsum", n_abl, "1,10,100", key="document")
    abl.append((name, swap(cmd, "--base_model_name", "gpt2")))
    S["replication"] += abl

    # WMT16 English/German (paper Fig. 4, paper_scripts/supervised.sh): multilingual mGPT scoring
    # model as in the paper; mT5-large instead of mT5-XL (3.7B, too large for 8 GB), mirroring
    # T5-large for T5-3B above
    S["wmt"] = [("install_wmt_deps", [PY, "-m", "pip", "install", "sentencepiece", "protobuf"]),  # mT5 tokenizer
                ("prefetch_wmt", [PY, "experiments/prefetch.py", "--cache_dir", CACHE,
                                  "--models", "sberbank-ai/mGPT,google/mt5-large", "--datasets", "wmt16"])]
    for lang in ["english", "german"]:
        name, cmd = detectgpt(f"rep_{lang}", lang, n_rep, "1,10,100")
        cmd = swap(swap(cmd, "--base_model_name", "sberbank-ai/mGPT"), "--mask_filling_model_name", "google/mt5-large")
        S["wmt"].append((name, cmd))

    # timing benchmark (decides n and k for the long runs) and a small T5-3B data point
    S["bench"] = [detectgpt("bench_t5large", "xsum", 10, "10", key="document", out="bench")]
    S["bench2"] = [detectgpt("bench2_fast", "xsum", 10, "10", key="document", out="bench")]
    S["bench3"] = [detectgpt("bench3_reuse", "xsum", 10, "1,10", key="document", out="bench")]
    n3, c3 = detectgpt("t5_3b_xsum", "xsum", 30, "1,10", key="document")
    S["t5_3b_check"] = [(n3, swap(c3, "--mask_filling_model_name", "t5-3b") + ["--half"])]

    # ---- Section 4: build the new datasets (needs data/persuade/*.csv) ----------------
    S["build_data"] = [
        ("build_hc3", [PY, "new_data/build_hc3.py", "--cache_dir", CACHE]),
        ("build_persuade", [PY, "new_data/build_persuade.py", "--per_prompt", "60"]),
        ("gen_qwen", [PY, "new_data/make_machine_essays.py", "--model", "Qwen/Qwen2.5-1.5B-Instruct",
                      "--per_prompt", "30", "--cache_dir", CACHE]),
        ("gen_smollm2", [PY, "new_data/make_machine_essays.py", "--model", "HuggingFaceTB/SmolLM2-1.7B-Instruct",
                         "--per_prompt", "30", "--cache_dir", CACHE]),
        ("pair_persuade", [PY, "new_data/pair_persuade.py"]),
        ("annotation_sheet", [PY, "new_data/make_annotation_sheet.py", "--n", "40"]),
    ]

    # ---- Section 5: DetectGPT on new data (night 2) -----------------------------------
    qwen, smol = "data/new/persuade_paired_qwen2.5-1.5b.csv", "data/new/persuade_paired_smollm2-1.7b.csv"
    new = [
        detectgpt("new_kaggle_gpt2xl", "kaggle", n_new, "1,10,100"),           # paper protocol on student essays
        detectgpt("new_hc3_scorer_gpt2xl", "paired:data/hc3/hc3_paired.csv", n_new, "1,10,100"),
        detectgpt("new_persuade_qwen_scorer_gpt2xl", "paired:" + qwen, n_new, "1,10,100"),
        detectgpt("new_persuade_smol_scorer_gpt2xl", "paired:" + smol, n_new, "1,10,100"),
    ]
    # white-box: score each subset with the model that actually wrote it
    for name, path, model in [("new_persuade_qwen_scorer_qwen", qwen, "Qwen/Qwen2.5-1.5B-Instruct"),
                              ("new_persuade_smol_scorer_smol", smol, "HuggingFaceTB/SmolLM2-1.7B-Instruct")]:
        # white-box scoring re-uses the cached perturbations, but Qwen/SmolLM2 have 150k/49k-token
        # vocabularies, so scoring 8 texts at once overflowed the 8 GB GPU (~3 min per pair);
        # these runs use 100 pairs and score 2 texts at a time
        n_, cmd = detectgpt(name, "paired:" + path, min(n_new, 100), "1,10,100")
        cmd = swap(swap(cmd, "--base_model_name", model), "--ll_batch_size", "2")
        new.append((n_, cmd))
    S["new_data"] = new

    # ---- Section 5: LLM-as-judge -------------------------------------------------------
    S["judge"] = [(f"judge_{os.path.basename(p)[:-4]}_{j.split('/')[-1]}",
                   [PY, "llm_judge.py", "--pairs", p, "--judge", j, "--n", str(n_new), "--cache_dir", CACHE])
                  for p in [qwen, smol, "data/hc3/hc3_paired.csv"]
                  for j in ["Qwen/Qwen2.5-3B-Instruct", "microsoft/Phi-3.5-mini-instruct"]]
    # ---- everything, in priority order (one command; safe to re-run after any interruption) ----
    import re
    by = {re.sub(r"_n\d+$", "", name): (name, cmd) for suite in ("replication", "build_data", "new_data", "judge", "t5_3b_check", "wmt")
          for name, cmd in S[suite]}
    order = ["rep_xsum",
             "build_hc3", "build_persuade", "gen_qwen", "gen_smollm2", "pair_persuade", "annotation_sheet",
             "new_persuade_qwen_scorer_gpt2xl",
             "rep_squad"]
    order += [n for n, _ in S["judge"]]
    order += ["new_hc3_scorer_gpt2xl", "rep_writing", "new_kaggle_gpt2xl", "rep_pubmed",
              "new_persuade_smol_scorer_gpt2xl",
              "install_wmt_deps", "prefetch_wmt", "rep_english", "rep_german",
              "abl_mask_t5-small", "abl_mask_t5-base",
              "new_persuade_qwen_scorer_qwen", "new_persuade_smol_scorer_smol"]
    # (abl_base_gpt2small and t5_3b_xsum remain runnable on their own; left out of "all" for time)
    assert set(order) <= set(by), set(order) - set(by)
    S["all"] = [by[n] for n in order]
    return S


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default=None)
    ap.add_argument("--only", default=None, help="comma-separated step names to run")
    ap.add_argument("--n_rep", type=int, default=200, help="pairs per replication run")
    ap.add_argument("--n_new", type=int, default=200, help="pairs per new-data run")
    ap.add_argument("--n_abl", type=int, default=100, help="pairs per ablation run")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()

    S = suites(args.n_rep, args.n_new, args.n_abl)
    if args.list:
        for s, steps in S.items():
            print(f"\n[{s}]")
            for name, cmd in steps:
                print(f"  {name}: {' '.join(cmd[1:])}")
        return
    steps = S[args.suite]
    if args.only:
        keep = set(args.only.split(","))
        steps = [s for s in steps if s[0] in keep]

    # Windows puts an idle laptop into standby, which freezes a running experiment for hours.
    # ES_CONTINUOUS | ES_SYSTEM_REQUIRED | ES_DISPLAY_REQUIRED keeps the machine awake while this
    # script runs. The display flag matters on "Modern Standby" laptops, which drop into low power
    # as soon as the screen turns off even when the system flag is set.
    if os.name == "nt":
        import ctypes
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001 | 0x00000002)
        print("Keeping the computer awake while experiments run.", flush=True)

    os.makedirs("logs", exist_ok=True)
    status_path = "logs/status.json"
    status = json.load(open(status_path)) if os.path.exists(status_path) else {}
    env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1")
    for name, cmd in steps:
        if status.get(name, {}).get("ok") and not args.force:
            print(f"[skip] {name} (already done)")
            continue
        print(f"[run ] {name}  ->  logs/{name}.log", flush=True)
        t0 = time.time()
        with open(f"logs/{name}.log", "w", encoding="utf-8") as log:
            log.write(" ".join(cmd) + "\n\n")
            log.flush()
            rc = subprocess.call(cmd, stdout=log, stderr=subprocess.STDOUT, env=env)
        status[name] = dict(ok=(rc == 0), returncode=rc, minutes=round((time.time() - t0) / 60, 1),
                            finished=time.strftime("%Y-%m-%d %H:%M"))
        json.dump(status, open(status_path, "w"), indent=2)
        print(f"[{'ok' if rc == 0 else 'FAIL'}] {name} in {status[name]['minutes']} min", flush=True)
    print("\nSuite finished. Summary in logs/status.json")


if __name__ == "__main__":
    main()
