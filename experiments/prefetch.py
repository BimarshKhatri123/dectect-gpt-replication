"""
prefetch.py -- download every model and dataset used in the report into the cache folder,
so the overnight runs never stall on a download. Only one weight format is fetched per model
(safetensors when available) to save disk space. Needs internet, no GPU.

    env\\Scripts\\python experiments\\prefetch.py --cache_dir ../cache
"""
import argparse
import fnmatch
import time

MODELS = [
    "gpt2", "gpt2-xl",                                          # source / scoring models
    "t5-small", "t5-base", "t5-large", "t5-3b",                 # mask-filling models
    "roberta-base-openai-detector", "roberta-large-openai-detector",   # supervised baselines
    "Qwen/Qwen2.5-1.5B-Instruct", "HuggingFaceTB/SmolLM2-1.7B-Instruct",  # generators (new data)
    "Qwen/Qwen2.5-3B-Instruct", "microsoft/Phi-3.5-mini-instruct",       # LLM judges
    "sberbank-ai/mGPT", "google/mt5-large",                     # WMT16 En/De replication
]
COMMON = ["*.json", "*.txt", "*.model", "tokenizer*", "spiece.model", "merges.txt", "vocab*"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache_dir", default="../cache")
    ap.add_argument("--models", default=None, help="comma-separated subset of models (default: all)")
    ap.add_argument("--datasets", default="xsum,squad,pubmed_qa", help="comma-separated datasets to fetch")
    args = ap.parse_args()

    from huggingface_hub import HfApi, snapshot_download
    api = HfApi()
    for m in (args.models.split(",") if args.models else MODELS):
        files = api.list_repo_files(m)
        weights = ["*.safetensors"] if any(fnmatch.fnmatch(f, "*.safetensors") for f in files) else ["pytorch_model*.bin"]
        print(f"-> {m}  ({weights[0]})", flush=True)
        for attempt in range(1, 9):          # home Wi-Fi drops; partial files resume on retry
            try:
                snapshot_download(m, cache_dir=args.cache_dir, allow_patterns=COMMON + weights)
                break
            except Exception as e:           # noqa: BLE001
                print(f"   attempt {attempt} failed ({type(e).__name__}); retrying in 20 s", flush=True)
                time.sleep(20)
        else:
            raise RuntimeError(f"could not download {m}")

    import datasets
    configs = {"xsum": None, "squad": None, "pubmed_qa": "pqa_labeled", "wmt16": "de-en"}
    for name in filter(None, args.datasets.split(",")):
        cfg = configs[name]
        print(f"-> dataset {name}", flush=True)
        d = datasets.load_dataset(name, cfg, split="train", cache_dir=args.cache_dir)
        print(f"   {len(d)} rows")
    print("All downloads finished.")


if __name__ == "__main__":
    main()
