"""
aggregate.py -- turn every finished run into the tables and figures used in the report.

    python analysis/aggregate.py          # reads results/, writes analysis/out/ and report/figures/

Outputs (analysis/out/)
  all_runs.csv                one row per (run, method): AUROC + 95% bootstrap CI, PR-AUC,
                              TPR at 1% / 5% false-positive rate, run time
  replication_vs_paper.csv    our AUROC next to Table 1 of Mitchell et al. (2023)
  persuade_by_quality.csv     AUROC when the human side is restricted by expert essay score
  judge_summary.csv           LLM-as-judge AUROC, accuracy, order bias, quality vs expert score
  judge_agreement.csv         Cohen's kappa between the two judges
  annotation.csv              human annotator vs gold / judges (only if the sheet is filled in)
Figures (report/figures/): fig_perturbations.pdf, fig_mask_ablation.pdf, fig_quality.pdf
"""
import glob
import json
import os
import re

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score, roc_auc_score, roc_curve

OUT, FIG = "analysis/out", "report/figures"
METHOD_FILES = {
    "likelihood_threshold_results.json": "log p(x)",
    "rank_threshold_results.json": "Rank",
    "logrank_threshold_results.json": "LogRank",
    "entropy_threshold_results.json": "Entropy",
    "roberta-base-openai-detector_results.json": "RoBERTa-base",
    "roberta-large-openai-detector_results.json": "RoBERTa-large",
}
# Table 1 of Mitchell et al. (2023), GPT-2 (1.5B) column, 500 samples, T5-3B, 100 perturbations
PAPER_GPT2 = {
    "xsum": {"log p(x)": 0.86, "Rank": 0.79, "LogRank": 0.89, "Entropy": 0.60, "DetectGPT": 0.99},
    "squad": {"log p(x)": 0.91, "Rank": 0.83, "LogRank": 0.94, "Entropy": 0.58, "DetectGPT": 0.99},
    "writing": {"log p(x)": 0.97, "Rank": 0.87, "LogRank": 0.98, "Entropy": 0.37, "DetectGPT": 0.99},
}


def boot_ci(real, samples, n_boot=2000, seed=0):
    """95% percentile bootstrap CI of AUROC, resampling human and machine scores separately."""
    rng = np.random.default_rng(seed)
    real, samples = np.asarray(real, float), np.asarray(samples, float)
    y = np.r_[np.zeros(len(real)), np.ones(len(samples))]
    stats = [roc_auc_score(y, np.r_[rng.choice(real, len(real)), rng.choice(samples, len(samples))])
             for _ in range(n_boot)]
    return np.percentile(stats, [2.5, 97.5])


def tpr_at(real, samples, fpr_target):
    fpr, tpr, _ = roc_curve(np.r_[np.zeros(len(real)), np.ones(len(samples))], np.r_[real, samples])
    return float(np.interp(fpr_target, fpr, tpr))


def method_name(fname):
    if fname in METHOD_FILES:
        return METHOD_FILES[fname]
    m = re.match(r"perturbation_(\d+)_([dz])_results\.json", fname)
    return f"DetectGPT-{m.group(2)} ({m.group(1)})" if m else None


def run_folders():
    for args_path in glob.glob("results/**/args.json", recursive=True):
        folder = os.path.dirname(args_path)
        if any(s in folder for s in ("smoke", "bench")):
            continue
        yield folder, json.load(open(args_path))


def collect():
    rows = []
    for folder, a in run_folders():
        rt = os.path.join(folder, "runtime.json")
        runtime = json.load(open(rt))["seconds"] / 60 if os.path.exists(rt) else np.nan
        run = (a.get("output_name") or "pilot").replace("final/", "")
        ds = a["dataset"]
        ds = os.path.splitext(os.path.basename(ds.split(":", 1)[1]))[0] if ds.startswith("paired:") else ds
        for f in sorted(glob.glob(os.path.join(folder, "*_results.json"))):
            m = method_name(os.path.basename(f))
            if m is None:
                continue
            r = json.load(open(f))
            h, s = r["predictions"]["real"], r["predictions"]["samples"]
            lo, hi = boot_ci(h, s)
            rows.append(dict(run=run, dataset=ds, base_model=a["base_model_name"], mask_model=a["mask_filling_model_name"],
                             n=len(h), method=m, auroc=r["metrics"]["roc_auc"], ci_low=lo, ci_high=hi,
                             pr_auc=r["pr_metrics"]["pr_auc"], tpr_at_1fpr=tpr_at(h, s, .01), tpr_at_5fpr=tpr_at(h, s, .05),
                             runtime_min=runtime, folder=folder))
    return pd.DataFrame(rows)


def best_run(df, prefix):
    """Largest-n finished run whose name starts with prefix (e.g. rep_xsum_n200 over rep_xsum)."""
    d = df[df.run.str.match(prefix + r"(_n\d+)?$")]
    return d[d.n == d.n.max()] if len(d) else d


def replication_table(df):
    out = []
    for ds, paper in PAPER_GPT2.items():
        d = best_run(df, f"rep_{ds}")
        for m, pv in paper.items():
            o = d[d.method == ("DetectGPT-z (100)" if m == "DetectGPT" else m)]
            if len(o):
                o = o.iloc[0]
                out.append(dict(dataset=ds, method=m, n=o.n, paper=pv, ours=o.auroc, ci_low=o.ci_low, ci_high=o.ci_high,
                                diff=o.auroc - pv))
    return pd.DataFrame(out)


def quality_table(df):
    """AUROC on PERSUADE runs when the human side is restricted by expert holistic score."""
    rows = []
    for run in sorted(df[df.run.str.startswith("new_persuade")].run.unique()):
        d0 = df[df.run == run].iloc[0]
        pairs_file = f"data/new/persuade_paired_{'qwen2.5-1.5b' if '_qwen_' in run else 'smollm2-1.7b'}.csv"
        pairs = pd.read_csv(pairs_file).head(d0.n)
        score = pairs.holistic_score.values
        for method in ["DetectGPT-z (100)", "LogRank", "log p(x)", "RoBERTa-large"]:
            f = df[(df.run == run) & (df.method == method)]
            if not len(f):
                continue
            fname = {"DetectGPT-z (100)": "perturbation_100_z_results.json", "LogRank": "logrank_threshold_results.json",
                     "log p(x)": "likelihood_threshold_results.json",
                     "RoBERTa-large": "roberta-large-openai-detector_results.json"}[method]
            r = json.load(open(os.path.join(f.iloc[0].folder, fname)))
            h, s = np.array(r["predictions"]["real"]), np.array(r["predictions"]["samples"])
            for name, mask in [("all", score > 0), ("score<=2", score <= 2), ("score>=4", score >= 4)]:
                y = np.r_[np.zeros(mask.sum()), np.ones(len(s))]
                rows.append(dict(run=run, method=method, human_subset=name, n_human=int(mask.sum()),
                                 auroc=roc_auc_score(y, np.r_[h[mask], s])))
    return pd.DataFrame(rows)


def judge_tables():
    rows, by_data = [], {}
    for f in sorted(glob.glob("results/llm_judge/*__*.csv")):
        data, judge = os.path.basename(f)[:-4].split("__")
        df = pd.read_csv(f)
        by_data.setdefault(data, {})[judge] = df
        y = (df.label == "machine").astype(int)
        lo, hi = boot_ci(df.p_ai[y == 0], df.p_ai[y == 1])
        row = dict(dataset=data, judge=judge, n_texts=len(df), auroc=roc_auc_score(y, df.p_ai), ci_low=lo, ci_high=hi,
                   accuracy=((df.p_ai > 0.5) == y).mean(), frac_called_ai=(df.p_ai > 0.5).mean(),
                   auroc_order1=roc_auc_score(y, df.p_ai_order1), auroc_order2=roc_auc_score(y, df.p_ai_order2),
                   order_agreement=((df.p_ai_order1 > .5) == (df.p_ai_order2 > .5)).mean())
        if "quality" in df:
            h, m = df[df.label == "human"], df[df.label == "machine"]
            row.update(quality_human=h.quality.mean(), quality_machine=m.quality.mean(),
                       spearman_vs_expert=h[["quality", "holistic_score"]].corr("spearman").iloc[0, 1],
                       qwk_vs_expert=cohen_kappa_score(h.holistic_score.astype(int), h.quality_argmax, weights="quadratic"))
        rows.append(row)
    agree = []
    for data, judges in by_data.items():
        if len(judges) == 2:
            (j1, a), (j2, b) = judges.items()
            m = a.merge(b, on=["pair_id", "label"], suffixes=("_1", "_2"))
            row = dict(dataset=data, judges=f"{j1} vs {j2}", n=len(m),
                       kappa_detect=cohen_kappa_score(m.p_ai_1 > .5, m.p_ai_2 > .5))
            if "quality_argmax_1" in m:
                row["qwk_quality"] = cohen_kappa_score(m.quality_argmax_1, m.quality_argmax_2, weights="quadratic")
            agree.append(row)
    return pd.DataFrame(rows), pd.DataFrame(agree), by_data


def annotation_table(by_data):
    sheet = next((p for p in ["data/annotation/sheet.xlsx", "data/annotation/sheet.csv"] if os.path.exists(p)), None)
    if sheet is None or not os.path.exists("data/annotation/key.csv"):
        return None
    s = pd.read_excel(sheet) if sheet.endswith("xlsx") else pd.read_csv(sheet)
    s = s[s.guess.astype(str).str.strip().str.upper().str[:1].isin(["H", "A"])]
    if s.empty:
        return None
    k = pd.read_csv("data/annotation/key.csv").merge(s[["item_id", "guess", "quality"]], on="item_id")
    k["says_ai"] = k.guess.astype(str).str.strip().str.upper().str.startswith("A")
    gold = k.label == "machine"
    rows = [dict(comparison="annotator vs gold", n=len(k), accuracy=(k.says_ai == gold).mean(),
                 kappa=cohen_kappa_score(gold, k.says_ai))]
    for data, judges in by_data.items():
        for judge, jd in judges.items():
            m = k.merge(jd[["pair_id", "label", "p_ai"]], on=["pair_id", "label"])
            if len(m):
                rows.append(dict(comparison=f"annotator vs {judge} ({data})", n=len(m),
                                 accuracy=((m.p_ai > .5) == m.says_ai).mean(), kappa=cohen_kappa_score(m.says_ai, m.p_ai > .5)))
    return pd.DataFrame(rows)


def figures(df, qt):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 7, "figure.dpi": 300, "axes.spines.top": False, "axes.spines.right": False})
    os.makedirs(FIG, exist_ok=True)
    colors = {"xsum": "#2a6fb0", "squad": "#d9822b", "writing": "#3a9b5c", "pubmed": "#8a5ab0",
              "persuade_paired_qwen2.5-1.5b": "#c0392b", "persuade_paired_smollm2-1.7b": "#7f8c8d"}

    # 1) AUROC vs number of perturbations
    fig, ax = plt.subplots(figsize=(3.0, 1.9))
    for prefix in ["rep_xsum", "rep_squad", "rep_writing", "rep_pubmed", "new_persuade_qwen_scorer_gpt2xl",
                   "new_persuade_smol_scorer_gpt2xl"]:
        d = best_run(df, prefix)
        d = d[d.method.str.startswith("DetectGPT-z")]
        if not len(d):
            continue
        d = d.assign(k=d.method.str.extract(r"\((\d+)\)")[0].astype(int)).sort_values("k")
        ds = d.dataset.iloc[0]
        label = {"persuade_paired_qwen2.5-1.5b": "PERSUADE+Qwen", "persuade_paired_smollm2-1.7b": "PERSUADE+SmolLM2"}.get(ds, ds)
        ax.errorbar(d.k, d.auroc, yerr=[d.auroc - d.ci_low, d.ci_high - d.auroc], marker="o", ms=2.5, lw=1, capsize=1.5,
                    color=colors.get(ds), label=label, ls="--" if ds.startswith("persuade") else "-")
    ax.set_xscale("log"); ax.set_xticks([1, 10, 100]); ax.set_xticklabels(["1", "10", "100"])
    ax.set_xlabel("perturbations $k$"); ax.set_ylabel("AUROC"); ax.legend(frameon=False, fontsize=5.5, loc="lower right")
    fig.tight_layout(); fig.savefig(f"{FIG}/fig_perturbations.pdf"); plt.close(fig)

    # 2) mask-filling model size (XSum, k=100)
    abl = df[(df.method == "DetectGPT-z (100)") & df.run.str.match(r"(abl_mask_.*|rep_xsum(_n\d+)?)$")]
    if len(abl) > 1:
        size = {"t5-small": 60, "t5-base": 220, "t5-large": 770, "t5-3b": 2850}
        abl = abl.assign(params=abl.mask_model.map(size)).sort_values(["params", "n"]).drop_duplicates("params", keep="first")
        fig, ax = plt.subplots(figsize=(3.0, 1.7))
        ax.errorbar(abl.params, abl.auroc, yerr=[abl.auroc - abl.ci_low, abl.ci_high - abl.auroc], marker="o", ms=3,
                    capsize=1.5, color="#2a6fb0", lw=1)
        for _, r in abl.iterrows():
            ax.annotate(f"{r.mask_model} (n={r.n})", (r.params, r.auroc), textcoords="offset points", xytext=(4, -8), fontsize=5.5)
        ax.axhline(0.99, color="grey", lw=.7, ls=":"); ax.text(70, 0.985, "paper (T5-3B)", fontsize=5.5, color="grey", va="top")
        ax.set_xscale("log"); ax.set_xlabel("mask-filling model parameters (M)"); ax.set_ylabel("AUROC (XSum)")
        fig.tight_layout(); fig.savefig(f"{FIG}/fig_mask_ablation.pdf"); plt.close(fig)

    # 3) AUROC by human-essay quality (PERSUADE)
    if len(qt):
        q = qt[qt.run.str.contains("scorer_gpt2xl")]
        fig, ax = plt.subplots(figsize=(3.0, 1.8))
        methods = ["DetectGPT-z (100)", "LogRank", "log p(x)", "RoBERTa-large"]
        subsets = ["score<=2", "all", "score>=4"]
        w = 0.2
        for i, m in enumerate(methods):
            vals = [q[(q.method == m) & (q.human_subset == s)].auroc.mean() for s in subsets]
            ax.bar(np.arange(3) + (i - 1.5) * w, vals, w, label=m.replace("-z (100)", ""),
                   color=["#c0392b", "#2a6fb0", "#7fb3e0", "#bbbbbb"][i])
        ax.set_xticks(range(3)); ax.set_xticklabels(["weak human essays\n(score $\\leq$2)", "all", "strong human essays\n(score $\\geq$4)"])
        ax.set_ylim(0.4, 1.0); ax.set_ylabel("AUROC"); ax.axhline(0.5, color="grey", lw=.6, ls=":")
        ax.legend(frameon=False, fontsize=5.5, ncol=2, loc="upper center", bbox_to_anchor=(0.5, 1.22))
        fig.tight_layout(); fig.savefig(f"{FIG}/fig_quality.pdf"); plt.close(fig)


def main():
    os.makedirs(OUT, exist_ok=True)
    df = collect()
    df.sort_values(["run", "method"]).to_csv(f"{OUT}/all_runs.csv", index=False)
    print(f"{len(df)} (run, method) rows from {df.folder.nunique()} runs; "
          f"total run time {df.drop_duplicates('folder').runtime_min.sum() / 60:.1f} GPU-hours")
    rt = replication_table(df)
    rt.to_csv(f"{OUT}/replication_vs_paper.csv", index=False)
    print(rt.round(3).to_string(index=False))
    qt = quality_table(df)
    qt.to_csv(f"{OUT}/persuade_by_quality.csv", index=False)
    print(qt.round(3).to_string(index=False))
    jt, agree, by_data = judge_tables()
    if len(jt):
        jt.to_csv(f"{OUT}/judge_summary.csv", index=False)
        agree.to_csv(f"{OUT}/judge_agreement.csv", index=False)
        print(jt.round(3).to_string(index=False)); print(agree.round(3).to_string(index=False))
    ann = annotation_table(by_data)
    if ann is not None:
        ann.to_csv(f"{OUT}/annotation.csv", index=False); print(ann.round(3).to_string(index=False))
    figures(df, qt)


if __name__ == "__main__":
    main()
