# -*- coding: utf-8 -*-
"""
evaluate.py
------------
Preliminary evaluation of the Inter/Intra-Domain ambiguity framework against
a small, manually-labelled gold set of (sentence, target_word, true_label)
triples. This corresponds to the "Proposed Evaluation Methodology" of the
project report (Section 4.4) being carried out on an initial scale, as a
first empirical check ahead of the full benchmark evaluation described as
future work.

Scope note (important for the viva/review): the gold labels here reflect
whether the target WORD carries genuine Inter-Domain or Intra-Domain
polysemy potential given the healthcare/software application setting of the
PURE dataset -- i.e. whether the framework's domain-of-sense classification
should reasonably flag it -- rather than a full human-level, per-occurrence
word-sense-disambiguation judgement. Extending this to a larger, independently
double-annotated benchmark (as in the base papers reviewed, e.g. Talha et al.
2025's 425-requirement dataset) is future work, consistent with the report's
Section 4.4.

Usage:
    python3 evaluate.py
"""

import os
import numpy as np
import pandas as pd

from ambiguity_framework import AmbiguityPipeline

# --------------------------------------------------------------------------
# Gold evaluation set
# --------------------------------------------------------------------------
# label options: "Inter-Domain", "Intra-Domain", "None"
GOLD_EXAMPLES = [
    # --- Inter-Domain examples -------------------------------------------
    dict(word="record", sentence="The system shall create a single patient record for each patient.",
         label="Inter-Domain",
         note="database record (Software/IT) vs. medical record (Healthcare)"),
    dict(word="chart", sentence="The physician shall be able to view the patient chart from any workstation.",
         label="Inter-Domain",
         note="patient chart (Healthcare) vs. graph/chart (General)"),
    dict(word="note", sentence="The system shall provide the ability to cosign a note and record the date and time of signature.",
         label="Inter-Domain",
         note="clinical note (Healthcare) vs. short message/comment (Software/IT)"),
    dict(word="order", sentence="The system shall allow the physician to enter a new medication order for the patient.",
         label="Inter-Domain",
         note="physician's order/prescription (Healthcare) vs. sort/purchase order (Software/Finance)"),
    dict(word="plan", sentence="The system shall allow the care team to update the patient's care plan.",
         label="Inter-Domain",
         note="care plan (Healthcare) vs. insurance/project plan (Finance/General)"),
    dict(word="code", sentence="The system shall allow the coder to assign a diagnosis code to the encounter.",
         label="Inter-Domain",
         note="diagnosis code (Healthcare) vs. program code (Software/IT)"),
    dict(word="provider", sentence="The system shall display the name of the provider who signed the note.",
         label="Inter-Domain",
         note="healthcare provider (Healthcare) vs. generic service provider (General)"),
    dict(word="table", sentence="The system shall store the reference ranges in a lookup table.",
         label="Inter-Domain",
         note="database table (Software/IT) vs. furniture table (General)"),
    dict(word="system", sentence="The system shall allow the user to update the patient's demographic information.",
         label="Inter-Domain",
         note="computer system (Software/IT) vs. any organised system (General)"),
    dict(word="port", sentence="The device shall expose a serial port for data export to external systems.",
         label="Inter-Domain",
         note="report's own flagship example: network/serial port (Software/IT) vs. seaport (General/Shipping)"),
    dict(word="solution", sentence="The vendor shall propose a software solution that meets all functional requirements.",
         label="Inter-Domain",
         note="software solution/product (Software/IT) vs. chemical solution / general answer (General)"),

    # --- Intra-Domain examples --------------------------------------------
    dict(word="session", sentence="The user shall log in to the system using a valid session key.",
         label="Intra-Domain",
         note="login/authentication session vs. work session -- both Software/IT-adjacent concepts"),
    dict(word="key", sentence="Each record shall be uniquely identified using a primary key.",
         label="Intra-Domain",
         note="database key vs. cryptographic/access key -- both Software/IT"),
    dict(word="field", sentence="The system shall validate that the required field is not left blank on the form.",
         label="Intra-Domain",
         note="form field vs. data field -- both Software/IT"),
    dict(word="log", sentence="The system shall maintain an audit log of all user login attempts.",
         label="Intra-Domain",
         note="audit/system log vs. transaction log -- both Software/IT"),
    dict(word="service", sentence="The system shall expose a web service for retrieving patient records.",
         label="Intra-Domain",
         note="report's own flagship example: web service vs. background/customer service -- both Software/IT"),
    dict(word="review", sentence="The system shall allow the physician to review the patient's allergy list.",
         label="Intra-Domain",
         note="review (examine again) vs. review (formal assessment) -- both general software-workflow senses"),

    # --- Not ambiguous in this application context (None) ------------------
    dict(word="patient", sentence="The system shall display the patient's date of birth on the summary screen.",
         label="None",
         note="single dominant Healthcare sense; no cross-domain conflict"),
    dict(word="physician", sentence="The physician shall be able to sign the note electronically.",
         label="None",
         note="single dominant Healthcare sense"),
    dict(word="capability", sentence="The system shall provide the capability to print a summary of the visit.",
         label="None",
         note="single dominant, non-domain-conflicting sense"),
    dict(word="allergy", sentence="The system shall allow the nurse to document a new allergy for the patient.",
         label="None",
         note="single dominant Healthcare sense"),
    dict(word="screen", sentence="The system shall display an error message on the login screen.",
         label="None",
         note="single dominant Software/IT sense in this context"),
    dict(word="password", sentence="The system shall require the user to change the password every ninety days.",
         label="None",
         note="single dominant Software/IT sense"),
]


def evaluate_at_threshold(pipeline_results_by_word, threshold):
    """Given cached per-example scoring info, recompute is_ambiguous/type at
    a candidate threshold without re-running the whole NLP pipeline."""
    y_true_binary, y_pred_binary = [], []
    y_true_type, y_pred_type = [], []
    rows = []
    for ex in pipeline_results_by_word:
        score = ex["ambiguity_score"]
        pred_is_ambig = score >= threshold
        pred_type = ex["raw_type"] if pred_is_ambig else "None"

        true_is_ambig = ex["label"] != "None"
        y_true_binary.append(true_is_ambig)
        y_pred_binary.append(pred_is_ambig)
        y_true_type.append(ex["label"])
        y_pred_type.append(pred_type)
        rows.append(
            {
                "word": ex["word"], "sentence": ex["sentence"], "gold_label": ex["label"],
                "predicted_label": pred_type, "ambiguity_score": round(score, 3),
                "correct": ex["label"] == pred_type,
            }
        )
    return y_true_binary, y_pred_binary, y_true_type, y_pred_type, rows


def prf1(y_true, y_pred):
    tp = sum(1 for t, p in zip(y_true, y_pred) if t and p)
    fp = sum(1 for t, p in zip(y_true, y_pred) if not t and p)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t and not p)
    tn = sum(1 for t, p in zip(y_true, y_pred) if not t and not p)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    accuracy = (tp + tn) / len(y_true) if y_true else 0.0
    return dict(tp=tp, fp=fp, fn=fn, tn=tn, precision=precision, recall=recall, f1=f1, accuracy=accuracy)


def main():
    print(f"Building pipeline and analysing {len(GOLD_EXAMPLES)} gold examples ...")
    sentences = [ex["sentence"] for ex in GOLD_EXAMPLES]
    pipeline = AmbiguityPipeline(ambiguity_threshold=0.0)  # threshold applied later, offline
    pipeline.fit_vocabulary(sentences)

    cached = []
    for ex in GOLD_EXAMPLES:
        results = pipeline.analyse_sentence(ex["sentence"])
        match = next((r for r in results if r.word == ex["word"]), None)
        if match is None:
            # word wasn't detected as a candidate at all (e.g. below WordNet
            # sense-count threshold and not in the curated glossary)
            cached.append({**ex, "ambiguity_score": 0.0, "raw_type": "None"})
            continue
        raw_type = match.ambiguity_type if match.ambiguity_type != "None" else (
            "Inter-Domain" if len(match.ranked_senses) >= 2 and
            match.ranked_senses[0].domain != match.ranked_senses[1].domain else "Intra-Domain"
        )
        cached.append({**ex, "ambiguity_score": match.ambiguity_score, "raw_type": raw_type})

    print("\nThreshold sweep (binary ambiguous-vs-not detection):")
    print(f"{'threshold':>10} {'acc':>6} {'prec':>6} {'rec':>6} {'f1':>6}")
    best_threshold, best_f1 = 0.5, -1.0
    for thr in np.arange(0.10, 0.95, 0.05):
        y_true_b, y_pred_b, *_ = evaluate_at_threshold(cached, thr)
        m = prf1(y_true_b, y_pred_b)
        print(f"{thr:>10.2f} {m['accuracy']:>6.2f} {m['precision']:>6.2f} {m['recall']:>6.2f} {m['f1']:>6.2f}")
        if m["f1"] > best_f1:
            best_f1 = m["f1"]
            best_threshold = thr

    print(f"\nBest threshold by F1 on the gold set: {best_threshold:.2f} (F1={best_f1:.2f})")

    y_true_b, y_pred_b, y_true_t, y_pred_t, rows = evaluate_at_threshold(cached, best_threshold)
    m = prf1(y_true_b, y_pred_b)
    print("\nBinary ambiguous-vs-not detection @ best threshold:")
    print(f"  Accuracy={m['accuracy']:.2f}  Precision={m['precision']:.2f}  Recall={m['recall']:.2f}  F1={m['f1']:.2f}")
    print(f"  TP={m['tp']} FP={m['fp']} FN={m['fn']} TN={m['tn']}")

    # Type-classification accuracy, restricted to examples the gold set says
    # ARE genuinely ambiguous (Inter-Domain or Intra-Domain).
    type_correct = sum(
        1 for t, p in zip(y_true_t, y_pred_t) if t != "None" and t == p
    )
    type_total = sum(1 for t in y_true_t if t != "None")
    type_acc = type_correct / type_total if type_total else 0.0
    print(f"\nInter- vs Intra-Domain classification accuracy (on truly-ambiguous gold words): "
          f"{type_correct}/{type_total} = {type_acc:.2f}")

    os.makedirs("sample_output", exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv("sample_output/gold_evaluation_results.csv", index=False)
    with open("sample_output/gold_evaluation_summary.txt", "w") as f:
        f.write(f"Gold set size: {len(GOLD_EXAMPLES)}\n")
        f.write(f"Best threshold (tuned on gold set by F1): {best_threshold:.2f}\n\n")
        f.write("Binary ambiguous-vs-not detection:\n")
        f.write(f"  Accuracy={m['accuracy']:.2f}  Precision={m['precision']:.2f}  "
                f"Recall={m['recall']:.2f}  F1={m['f1']:.2f}\n")
        f.write(f"  TP={m['tp']} FP={m['fp']} FN={m['fn']} TN={m['tn']}\n\n")
        f.write(f"Inter- vs Intra-Domain classification accuracy (truly-ambiguous words only): "
                f"{type_correct}/{type_total} = {type_acc:.2f}\n\n")
        f.write("Per-example results:\n")
        f.write(df.to_string(index=False))
    print("\nWrote sample_output/gold_evaluation_results.csv and gold_evaluation_summary.txt")
    return best_threshold


if __name__ == "__main__":
    main()
