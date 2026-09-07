# -*- coding: utf-8 -*-
"""
run_pipeline.py
-----------------
Runs the Inter/Intra-Domain ambiguity detection pipeline (ambiguity_framework.py)
over a sample of the PURE requirements dataset (Pure_Annotate_Dataset.csv) and
the small demo test set (testData.csv), producing:
    - sample_output/ambiguity_report_full.csv   (every candidate word instance)
    - sample_output/ambiguity_report_flagged.csv (only words flagged ambiguous)
    - sample_output/summary_ambiguity_type_counts.png
    - sample_output/summary_domain_distribution.png
    - sample_output/summary.txt (headline numbers for the report / slides)

Usage:
    python3 run_pipeline.py --dataset Pure_Annotate_Dataset.csv --sample_size 400
"""

import argparse
import os
import random

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from ambiguity_framework import AmbiguityPipeline


def load_sentences(dataset_path: str, sample_size: int, seed: int = 42) -> list:
    df = pd.read_csv(dataset_path, encoding="latin-1")
    col = 'sentence' if 'sentence' in df.columns else ('requirement_sentence' if 'requirement_sentence' in df.columns else df.columns[0])
    sentences = df[col].dropna().astype(str).tolist()
    # keep sentences of reasonable length; drop near-duplicates/empties
    sentences = [s.strip() for s in sentences if 20 <= len(s.strip()) <= 300]
    random.seed(seed)
    if sample_size and sample_size < len(sentences):
        sentences = random.sample(sentences, sample_size)
    return sentences


def results_to_dataframe(results) -> pd.DataFrame:
    rows = []
    for r in results:
        top_domains = [s.domain for s in r.ranked_senses[:2]]
        top_glosses = [s.gloss for s in r.ranked_senses[:2]]
        rows.append(
            {
                "word": r.word,
                "surface_form": r.surface_form,
                "sentence": r.sentence,
                "sentence_domain": r.sentence_domain,
                "ambiguity_score": round(r.ambiguity_score, 3),
                "semantic_similarity_spread": round(r.semantic_similarity_spread, 3),
                "contextual_confidence_gap": round(r.contextual_confidence, 3),
                "domain_diversity": round(r.domain_diversity, 3),
                "is_ambiguous": r.is_ambiguous,
                "ambiguity_type": r.ambiguity_type,
                "top_sense_1_domain": top_domains[0] if len(top_domains) > 0 else "",
                "top_sense_1_gloss": top_glosses[0] if len(top_glosses) > 0 else "",
                "top_sense_2_domain": top_domains[1] if len(top_domains) > 1 else "",
                "top_sense_2_gloss": top_glosses[1] if len(top_glosses) > 1 else "",
                "explanation": r.explanation,
            }
        )
    return pd.DataFrame(rows)


def make_charts(df: pd.DataFrame, outdir: str):
    # Chart 1: Ambiguity type counts
    counts = df["ambiguity_type"].value_counts()
    plt.figure(figsize=(6, 4))
    counts.plot(kind="bar", color=["#2BC4B8", "#E8935A", "#6E7F91"])
    plt.title("Ambiguity Type Counts (candidate word instances)")
    plt.ylabel("Count")
    plt.xlabel("Ambiguity Type")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "summary_ambiguity_type_counts.png"), dpi=150)
    plt.close()

    # Chart 2: Domain distribution of flagged ambiguous words (top sense 1 domain)
    flagged = df[df["is_ambiguous"]]
    if not flagged.empty:
        dom_counts = flagged["top_sense_1_domain"].value_counts()
        plt.figure(figsize=(7, 4))
        dom_counts.plot(kind="bar", color="#2BC4B8")
        plt.title("Domain Distribution of Flagged Ambiguous Words")
        plt.ylabel("Count")
        plt.xlabel("Domain (top candidate sense)")
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, "summary_domain_distribution.png"), dpi=150)
        plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="Pure_Annotate_Dataset.csv")
    parser.add_argument("--sample_size", type=int, default=400)
    parser.add_argument("--outdir", default="sample_output")
    parser.add_argument("--threshold", type=float, default=None,
                         help="Fixed ambiguity_score threshold. If omitted, --top_percentile is used instead.")
    parser.add_argument("--top_percentile", type=float, default=0.30,
                         help="Flag the top X fraction of candidate instances by ambiguity_score "
                              "within this corpus (used when --threshold is not given). Default: 0.30.")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    print(f"Loading sentences from {args.dataset} (sample_size={args.sample_size}) ...")
    sentences = load_sentences(args.dataset, args.sample_size)
    print(f"Loaded {len(sentences)} requirement sentences.")

    # First pass: score everything with threshold=0 so nothing is filtered,
    # then decide the operating threshold from either the fixed value or
    # the requested top-percentile of the *actual* score distribution in
    # this corpus.
    pipeline = AmbiguityPipeline(ambiguity_threshold=0.0)
    print("Fitting shared TF-IDF vocabulary (WordNet glosses + domain descriptors + sentences) ...")
    pipeline.fit_vocabulary(sentences)

    print("Analysing corpus ...")
    results = pipeline.analyse_corpus(sentences)
    df = results_to_dataframe(results)

    if args.threshold is not None:
        operating_threshold = args.threshold
        mode_desc = f"fixed threshold = {operating_threshold:.2f}"
    else:
        operating_threshold = float(df["ambiguity_score"].quantile(1 - args.top_percentile))
        mode_desc = f"top {args.top_percentile*100:.0f}% by score in this corpus (score >= {operating_threshold:.2f})"

    df["is_ambiguous"] = df["ambiguity_score"] >= operating_threshold
    df.loc[~df["is_ambiguous"], "ambiguity_type"] = "None"
    df.loc[~df["is_ambiguous"], ["top_sense_2_domain", "top_sense_2_gloss"]] = ""
    print(f"Operating point: {mode_desc}")

    full_path = os.path.join(args.outdir, "ambiguity_report_full.csv")
    flagged_path = os.path.join(args.outdir, "ambiguity_report_flagged.csv")
    df.to_csv(full_path, index=False)
    df[df["is_ambiguous"]].sort_values("ambiguity_score", ascending=False).to_csv(flagged_path, index=False)

    make_charts(df, args.outdir)

    n_sentences = len(sentences)
    n_candidates = len(df)
    n_flagged = int(df["is_ambiguous"].sum())
    n_inter = int((df["ambiguity_type"] == "Inter-Domain").sum())
    n_intra = int((df["ambiguity_type"] == "Intra-Domain").sum())
    top_words = (
        df[df["is_ambiguous"]]["word"].value_counts().head(15).to_string()
    )

    summary = (
        f"Operating point:               {mode_desc}\n"
        f"Sentences analysed:            {n_sentences}\n"
        f"Candidate word instances:      {n_candidates}\n"
        f"Flagged ambiguous instances:   {n_flagged} ({(n_flagged/n_candidates*100 if n_candidates else 0):.1f}%)\n"
        f"  - Inter-Domain:              {n_inter}\n"
        f"  - Intra-Domain:              {n_intra}\n"
        f"\nTop 15 most frequently flagged ambiguous words:\n{top_words}\n"
    )
    print(summary)
    with open(os.path.join(args.outdir, "summary.txt"), "w") as f:
        f.write(summary)

    print(f"\nOutputs written to: {os.path.abspath(args.outdir)}")


if __name__ == "__main__":
    main()
