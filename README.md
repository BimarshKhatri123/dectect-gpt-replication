# Replicating DetectGPT and testing it on machine-generated student essays

COMP8240 (Applications of Data Science, Macquarie University) final project. Bimarsh Khatri, 49140124.

This repository is a fork of the official DetectGPT code
([eric-mitchell/detect-gpt](https://github.com/eric-mitchell/detect-gpt), Mitchell et al., ICML 2023) with:

1. **Replication** of the paper's GPT-2 XL results on XSum, SQuAD, WritingPrompts and PubMedQA, plus ablations
   that explain the remaining gap (mask-filling model size, number of perturbations).
2. **New data**: Kaggle LLM-Detect student essays, HC3 (human vs ChatGPT answers), and a **new dataset of 600
   student/LLM essay pairs** built from the PERSUADE 2.0 corpus with Qwen2.5-1.5B-Instruct and SmolLM2-1.7B-Instruct.
3. **LLM-as-judge**: Qwen2.5-3B-Instruct and Phi-3.5-mini-instruct used as zero-shot detectors and as essay-quality raters.

The report is in [`report/main.pdf`](report/main.pdf) (ACL format, LaTeX source in `report/`).

## Repository layout

| Path | What it is |
|---|---|
| `run.py` | The authors' experiment script. Every change is marked `# NEW` (paired-CSV datasets, fp16/batched scoring, per-text perturbation cache, Windows fixes). With none of the new flags it behaves like the original. |
| `custom_datasets.py` | Original loaders + `load_kaggle` (UTF-8 fix for WritingPrompts on Windows). |
| `paired_loader.py` | Loads any CSV with `human`/`machine` columns: `python run.py --dataset paired:<file.csv>`. |
| `llm_judge.py` | LLM-as-judge (detection probability and 1–6 quality score from next-token probabilities). |
| `new_data/` | Scripts that build the new datasets (see below). |
| `experiments/run_suite.py` | Runs every experiment in the report; one log per run in `logs/`, resumable. |
| `experiments/prefetch.py` | Downloads all models and datasets into the cache. |
| `analysis/aggregate.py` | Builds every table and figure (bootstrap CIs, TPR at fixed FPR) from `results/`. |
| `results/final/` | Result files of the final runs (AUROC, predictions, ROC curves). `results/llm_judge/` holds judge scores. Earlier `results/*` folders are the Assignment 1–2 pilot runs. |
| `data/new/academic_essays_v2.csv` | **The new dataset** (long format, 1,200 texts, full provenance). |
| `data/new/persuade_paired_*.csv` | The same data as pairs, the input to `run.py` and `llm_judge.py`. |
| `data/hc3/hc3_paired.csv` | HC3 pairs used in the report. |
| `paper_scripts/` | The original authors' run scripts (unchanged). |
| `pilot/` | Assignment-2 versions of the generation and cleaning scripts (superseded by `new_data/`). |
| `report/` | LaTeX source (ACL template), figures and the compiled report. |
| `analysis/out/` | Every number in the report as CSV (AUROC with 95% CIs, TPR at fixed FPR, judge tables). |

Result files that would contain Kaggle competition essays are not committed (the competition data may not be
redistributed); their numbers are in `analysis/out/all_runs.csv`, and `run_suite.py` regenerates them.

## Setup

Tested on Windows 11, Python 3.10, NVIDIA RTX 4060 Laptop (8 GB), 16 GB RAM.

```bash
python -m venv env
env\Scripts\activate            # Linux/macOS: source env/bin/activate
pip install -r requirements.txt  # install the CUDA build of torch first if needed: https://pytorch.org
python experiments/prefetch.py --cache_dir ../cache   # ~45 GB of models and datasets
```

External data that cannot be redistributed here:

* **WritingPrompts**: download from [Kaggle](https://www.kaggle.com/datasets/ratthachat/writing-prompts), put `valid.wp_source` and `valid.wp_target` in `data/writingPrompts/`.
* **Kaggle LLM-Detect**: download `train_essays.csv` and `train_prompts.csv` from the [competition](https://www.kaggle.com/competitions/llm-detect-ai-generated-text/data) into `data/kaggle/`.
* **PERSUADE 2.0**: download the "Training set" CSV linked from [scrosseye/persuade_corpus_2.0](https://github.com/scrosseye/persuade_corpus_2.0) into `data/persuade/` (only needed to *rebuild* the new dataset, which is already included).

## Reproducing the report

```bash
python experiments/run_suite.py --list            # show every command
python experiments/run_suite.py --suite all       # everything in the report (~39 GPU-hours on an RTX 4060 Laptop; per-run times in Appendix B of the report)
python analysis/aggregate.py                      # tables -> analysis/out/, figures -> report/figures/
```

Individual parts: `--suite replication`, `--suite build_data`, `--suite new_data`, `--suite judge`. A quick end-to-end
check (~20 min) is `--suite smoke`. A single run looks like:

```bash
python run.py --dataset xsum --dataset_key document --n_samples 200 --n_perturbation_list 1,10,100 \
  --base_model_name gpt2-xl --mask_filling_model_name t5-large --pct_words_masked 0.3 --span_length 2 \
  --base_half --ll_batch_size 8 --reuse_perturbations --perturbation_cache perturbation_cache --cache_dir ../cache
```

## How the new dataset was built (`new_data/`)

1. `build_persuade.py`: selects the 8 independent PERSUADE prompts plus the 2 text-dependent prompts whose source
   texts are available, and samples 60 student essays (≥150 words) per prompt. Demographic fields are dropped.
2. `make_machine_essays.py`: each generator writes 30 essays per prompt from the same assignment text the students
   saw (plus source passages), as "a US student in grade *g*"; temperature 0.8, top-p 0.95, seed 0. Markdown, titles
   and placeholders are removed; outputs under 150 words are discarded.
3. `pair_persuade.py`: pairs every machine essay with a different student essay on the same prompt, truncates both
   to the same length (≤250 words) and writes the paired and long-format CSVs.
4. `make_annotation_sheet.py`: a blind 40-item sheet for human annotation (optional).

`build_hc3.py` builds the HC3 pairs (one human and one ChatGPT answer per question, both normalised and length-matched).

## Licences and credit

Code: MIT (original DetectGPT licence retained). PERSUADE-derived data: CC BY-NC-SA 4.0 (Crossley et al., 2024),
non-commercial research use only. HC3-derived data: CC BY-SA 4.0 (Guo et al., 2023). Machine essays were generated
with Qwen2.5-1.5B-Instruct and SmolLM2-1.7B-Instruct (both Apache-2.0) and are labelled as machine-generated.
