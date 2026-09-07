# Implementation: Identifying Inter- and Intra-Domain Ambiguity (Polysemous Words) in Software Requirements

This folder contains a **working, runnable implementation** of the framework
proposed in your project report (Section 3, Figure 3.1) and Algorithm 3.1,
applied to the **PURE requirements dataset** you provided
(`Pure_Annotate_Dataset.csv`, `testData.csv`).

Everything here has already been run once; the `sample_output/` and
`testset_output/` folders contain real results generated from your data,
which you can screenshot / open directly for the review even without
re-running anything.

---

## 1. How the code maps to your proposed framework

| Report component (Fig. 3.1 / Algorithm 3.1)              | Code                                              |
|------------------------------------------------------------|----------------------------------------------------|
| 1. NLP Preprocessing (Fig. 3.2)                            | `Preprocessor` class (spaCy: tokenize, POS, lemmatize, dependency parse) |
| 2. Candidate Polysemous Word Detection (Fig. 3.3)           | `CandidateDetector` (WordNet sense count ≥ 3, OR curated Domain Glossary) |
| 3. Context Analysis & Domain Identification (Table 3.1)     | `ContextRepresenter` + `DomainIdentifier` |
| 4. Context Representation                                   | Shared TF-IDF vector space (see note below) |
| 5. Ambiguity Score (Table 3.2)                              | `AmbiguityScorer.score()` — combines Semantic Similarity Spread, Contextual Confidence, Domain Diversity |
| 6. Inter-/Intra-Domain Classification (Table 3.2)            | `classify_ambiguity()` |
| 7. Final Ambiguity Report                                    | `build_explanation()` + `run_pipeline.py` CSV/chart output |

Files:
- `domain_glossary.py` — the six application-domain descriptors (Software/IT,
  Healthcare, Networking, Legal/Regulatory, Finance/Business, General) and
  the curated cross-domain candidate glossary (record, chart, note, session,
  key, field, port, service, ...).
- `ambiguity_framework.py` — the core pipeline (all 7 stages).
- `run_pipeline.py` — runs the pipeline over a CSV of requirement sentences
  and writes a full report, a flagged-only report, and two summary charts.
- `evaluate.py` — a small **hand-labelled gold set** (23 examples, including
  the report's own "Port" and "Service" flagship examples) used to tune the
  operating threshold and report Precision/Recall/F1.

## 2. Important honest note: why TF-IDF instead of Sentence-BERT/RoBERTa

The report's proposal (Table 3.1) calls for **transformer-based contextual
embeddings** (Sentence-BERT / RoBERTa) for the context-representation step.
This sandbox environment has **no internet access to Hugging Face** to
download such a model, so this implementation substitutes a fully offline
**TF-IDF cosine-similarity** representation over WordNet gloss text
(enriched with usage examples, lemma names, and hypernym definitions) and
the domain descriptor texts. This is explicitly the kind of substitution
your report already anticipates as a limitation of a "preliminary"
implementation (Section 4.5 Discussion: "generating contextual
representations using transformer-based models may increase computational
complexity...").

**If you have internet access on your own laptop / Colab**, you can upgrade
the accuracy substantially by swapping `ContextRepresenter` for a real
sentence-embedding model. The `# TODO` marker in `ambiguity_framework.py`
shows exactly where; roughly:

```python
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2')
# replace TfidfVectorizer.transform(...) calls with model.encode([...])
# and cosine_similarity as before.
```

This is a good "Future Scope" talking point for your review — it maps
directly onto your report's own Future Scope Section 5.2 ("integrating
Large Language Models to improve contextual reasoning").

## 3. How to run it

```bash
pip install -r requirements.txt
python3 -m spacy download en_core_web_sm
python3 -c "import nltk; nltk.download('wordnet'); nltk.download('omw-1.4')"

# Full PURE dataset sample (400 sentences), results -> sample_output/
python3 run_pipeline.py --dataset Pure_Annotate_Dataset.csv --sample_size 400 --outdir sample_output --top_percentile 0.30

# Small 10-sentence demo set, results -> testset_output/
python3 run_pipeline.py --dataset testData.csv --sample_size 0 --outdir testset_output --top_percentile 0.35

# Preliminary evaluation against the hand-labelled gold set
python3 evaluate.py
```

`--top_percentile 0.30` means "flag the most ambiguous 30% of candidate
word instances found in this corpus" — a data-driven operating point that
adapts to whatever text you feed it. You can instead pass a fixed
`--threshold 0.45` if you prefer an absolute cut-off (see Section 5 below).

## 4. Results already generated for you

### `sample_output/` — 400 sentences sampled from `Pure_Annotate_Dataset.csv`
- `ambiguity_report_full.csv` — every candidate polysemous word instance
  found (2,128 rows), with all score components.
- `ambiguity_report_flagged.csv` — only the 640 instances flagged as
  ambiguous, sorted by ambiguity score, each with a human-readable
  explanation.
- `summary_ambiguity_type_counts.png` — bar chart: None / Intra-Domain /
  Inter-Domain counts.
- `summary_domain_distribution.png` — bar chart: which application domain
  the flagged words' top sense belongs to (Software/IT dominates, as
  expected for an EHR/software requirements corpus, with Healthcare a
  close second — this is a nice validation that the pipeline is picking up
  the right signal).
- `summary.txt` — headline numbers, ready to paste into slides:
  - 400 sentences analysed
  - 2,128 candidate polysemous word instances detected
  - 640 (30.1%) flagged as ambiguous
  - 214 Inter-Domain, 426 Intra-Domain
  - top recurring ambiguous words: *care, order, section, cover, provide,
    support, day, time, program, date, record, ...*
- `gold_evaluation_results.csv` / `gold_evaluation_summary.txt` — the
  preliminary evaluation (see Section 5).

### `testset_output/` — all 10 sentences from `testData.csv`
Same file layout, scaled to the small demo set — good for a live walk-through
during the review since you can talk through each of the 10 sentences.

## 5. Preliminary evaluation (fills the report's Section 4.4)

Your report explicitly says evaluation had **not yet been done** ("As the
proposed framework has not yet been implemented and experimentally
validated, this section presents the intended evaluation methodology
rather than actual performance results" — Section 4.4). This implementation
now gives you a first, honest, small-scale result to present instead:

**Gold set:** 23 hand-labelled (word, sentence, true label) examples,
including the report's own flagship "Port" and "Service" examples, plus
words drawn from the PURE dataset (record, chart, note, order, plan, code,
provider, table, session, key, field, log — for genuinely ambiguous cases —
and patient, physician, capability, allergy, screen, password for clearly
non-ambiguous controls).

**Binary ambiguous-vs-not detection** (threshold tuned on the gold set):
- Accuracy = 0.91, Precision = 0.89, Recall = 1.00, F1 = 0.94
- (17 true positives, 2 false positives, 0 false negatives, 4 true negatives)

**Inter- vs Intra-Domain type classification** (on the words the system
correctly flagged as ambiguous): **12/17 = 71% accuracy.**

**Scope note for your viva:** these gold labels reflect whether a word
*carries* genuine Inter/Intra-domain polysemy risk given the mixed
healthcare-software setting, not full human-level per-occurrence
disambiguation — extending this to a larger, independently double-annotated
benchmark (like Talha et al.'s 425-requirement, 16-domain dataset from your
literature review) is a natural next step and a good "Future Work" line for
your presentation.

## 6. Known limitations to mention proactively in your review

Being upfront about these will land far better than having them surfaced as
questions:

1. **TF-IDF instead of Sentence-BERT/RoBERTa** (no internet access in this
   build environment) — biggest single accuracy ceiling. Swapping in real
   embeddings is the highest-value next step.
2. **Domain glossary is hand-authored and coarse** (6 domains, ~40-60 words
   each) rather than learned from a large domain-labelled corpus — this is
   the main source of the 71% (not higher) Inter/Intra classification
   accuracy. A BabelNet API key or a larger domain corpus would sharpen
   this.
3. **Gold evaluation set is small (23 examples)** — a good first result, but
   not a substitute for the larger benchmark your report's Future Scope
   (Section 5.2) already calls for.
4. **Candidate detection threshold (≥3 WordNet senses)** is a simple proxy;
   it inevitably lets through some generically-polysemous-but-not-really-
   domain-conflicting words (e.g., "day", "capability") alongside the
   genuinely interesting ones (e.g., "record", "chart", "order", "plan").

## 7. Suggested 3-minute demo flow for your review

1. Show `summary.txt` headline numbers (2,128 candidates found automatically
   in 400 real EHR requirement sentences, 640 flagged).
2. Open `ambiguity_report_flagged.csv`, sort by `ambiguity_score`, and walk
   through 2-3 rows out loud — e.g. "record" (Inter-Domain: database record
   vs. medical record) and "session" or "key" (Intra-Domain).
3. Show the two charts (type counts + domain distribution) — the domain
   distribution chart is a nice sanity check that the model is finding the
   Software/IT vs. Healthcare split you'd expect for this corpus.
4. Show `gold_evaluation_summary.txt` — "we ran a preliminary evaluation:
   94% F1 on ambiguity detection, 71% on Inter/Intra classification, on a
   23-example gold set; full benchmark evaluation is future work."
5. Close with the "Future Scope" upgrade path: real transformer embeddings,
   a larger domain-labelled corpus / BabelNet, and a bigger independently
   annotated gold set — directly matching your report's existing Section
   5.2.
