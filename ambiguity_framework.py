# -*- coding: utf-8 -*-
"""
ambiguity_framework.py
-----------------------
Implementation of the proposed framework for "Identifying Inter and Intra
Domain-Specific Ambiguity (Polysemous Words) from Natural Language Software
Requirements".

This module implements, in code, the seven-stage pipeline described in the
project report (Section 3, Figure 3.1 / 3.4) and Algorithm 3.1:

    1. NLP Preprocessing            -> Preprocessor
    2. Candidate Polysemous Word
       Detection                    -> CandidateDetector
    3. Context Analysis &
       Domain Identification        -> DomainIdentifier
    4. Context Representation       -> ContextRepresenter (TF-IDF vectors)
    5. Ambiguity Score computation  -> AmbiguityScorer
    6. Inter/Intra-Domain
       Classification               -> classify_ambiguity()
    7. Final Ambiguity Report       -> AmbiguityPipeline.analyse_sentence()

Because the runtime environment for this student project has no internet
access to a live BabelNet API or a hosted transformer embedding service,
this implementation substitutes:
    - WordNet (via NLTK)            for the lexical-resource / sense
                                     inventory role of WordNet + BabelNet,
    - a curated domain glossary     for the "Domain Glossary" lexical
                                     resource,
    - TF-IDF cosine similarity      for the "contextual semantic
                                     embeddings" (Sentence-BERT / RoBERTa)
                                     role described in the report as a
                                     lightweight, fully-offline stand-in
                                     that is appropriate for a proof-of-
                                     concept / preliminary implementation.

The module is intentionally organised so that each class/function maps
onto one labelled box of Figure 3.1 in the report, to make the mapping
between the proposal and the implementation explicit and easy to explain
in a viva/project review.
"""

from __future__ import annotations

import re
import string
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import spacy
from nltk.corpus import wordnet as wn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from domain_glossary import DOMAIN_DESCRIPTORS, CANDIDATE_GLOSSARY_TERMS


def _enriched_gloss_text(syn) -> str:
    """Builds a richer bag-of-words representation of a WordNet sense than
    the bare gloss, by appending WordNet usage examples, the sense's own
    lemma names, and its direct hypernyms' definitions. This substantially
    increases vocabulary overlap with real requirement sentences and domain
    descriptors, which is important because the bare one-line WordNet
    gloss is often too short for TF-IDF cosine similarity to be
    informative on its own.
    """
    parts = [syn.definition()]
    parts.extend(syn.examples())
    parts.extend(name.replace("_", " ") for name in syn.lemma_names())
    for hyper in syn.hypernyms():
        parts.append(hyper.definition())
    return ". ".join(parts)


# --------------------------------------------------------------------------
# Stage 1: NLP Preprocessing  (report Section 3.2, Figure 3.2)
# --------------------------------------------------------------------------
class Preprocessor:
    """Sentence segmentation, tokenisation, lemmatisation, POS tagging and
    dependency parsing, using spaCy's small English pipeline.

    This corresponds to Figure 3.2 in the report: "NLP Preprocessing"
    (Sentence Segmentation -> Tokenization -> Lemmatization -> POS Tagging
    -> Dependency Parsing -> Preprocessed Requirement Text).
    """

    def __init__(self, model_name: str = "en_core_web_sm"):
        self.nlp = spacy.load(model_name)

    def process(self, text: str):
        """Return a spaCy Doc with tokens, lemmas, POS tags and dependency
        parses already computed."""
        return self.nlp(text)


# --------------------------------------------------------------------------
# Stage 2: Candidate Polysemous Word Detection (report Section 3.3, Fig 3.3)
# --------------------------------------------------------------------------
@dataclass
class CandidateWord:
    lemma: str
    pos_wn: str            # WordNet POS tag used to look up senses
    token_index: int       # position of the token within the sentence
    surface_form: str
    senses: List["wn.synset"] = field(default_factory=list)


class CandidateDetector:
    """Identifies candidate polysemous words using WordNet sense counts and
    a curated Domain Glossary, corresponding to Figure 3.3 ("Lexical
    Resource Lookup: WordNet + BabelNet + Domain Glossary" ->
    "Multiple Meanings Available?" -> "Candidate Polysemous Word").
    """

    #: spaCy POS -> WordNet POS mapping (nouns are the primary focus,
    #: since Inter/Intra-Domain examples in the report -- Port, Service --
    #: are both nouns; verbs are supported too for completeness).
    _POS_MAP = {"NOUN": wn.NOUN, "PROPN": wn.NOUN, "VERB": wn.VERB}

    def __init__(self, min_wordnet_senses: int = 3, min_word_len: int = 3):
        self.min_wordnet_senses = min_wordnet_senses
        self.min_word_len = min_word_len

    def detect(self, doc) -> List[CandidateWord]:
        candidates: List[CandidateWord] = []
        seen_positions = set()
        for tok in doc:
            if tok.i in seen_positions:
                continue
            if tok.is_stop or tok.is_punct or not tok.is_alpha:
                continue
            if len(tok.lemma_) < self.min_word_len:
                continue
            wn_pos = self._POS_MAP.get(tok.pos_)
            if wn_pos is None:
                continue

            lemma = tok.lemma_.lower()
            senses = wn.synsets(lemma, pos=wn_pos)

            is_glossary_term = lemma in CANDIDATE_GLOSSARY_TERMS
            is_wordnet_polysemous = len(senses) >= self.min_wordnet_senses

            if is_glossary_term or is_wordnet_polysemous:
                # Fall back to any-POS synsets for glossary terms whose
                # WordNet sense count under the detected POS is thin
                # (keeps recall high for known cross-domain terms).
                if is_glossary_term and len(senses) < 2:
                    senses = wn.synsets(lemma)
                candidates.append(
                    CandidateWord(
                        lemma=lemma,
                        pos_wn=wn_pos,
                        token_index=tok.i,
                        surface_form=tok.text,
                        senses=senses,
                    )
                )
                seen_positions.add(tok.i)
        return candidates


# --------------------------------------------------------------------------
# Stage 3 & 4: Context Representation and Domain Identification
# (report Section 3.4, Table 3.1)
# --------------------------------------------------------------------------
class ContextRepresenter:
    """Builds a single shared TF-IDF vector space over:
        - all candidate WordNet sense definitions (glosses) encountered,
        - the domain descriptor texts (DOMAIN_DESCRIPTORS), and
        - the requirement sentences being analysed.

    This stands in for the "contextual semantic embeddings generated by
    transformer-based language models (Sentence-BERT / RoBERTa)" described
    in Table 3.1, in a fully offline setting. The vectoriser must be fit
    once on a representative corpus before use (see `fit`).

    # TODO (upgrade path / Future Scope): if you have internet access,
    # replace the TfidfVectorizer-based fit()/transform() below with a
    # SentenceTransformer, e.g.:
    #     from sentence_transformers import SentenceTransformer
    #     model = SentenceTransformer('all-MiniLM-L6-v2')
    #     vectors = model.encode(list_of_texts)
    # and use cosine_similarity(vectors_a, vectors_b) exactly as now. This
    # directly matches the transformer-embedding component originally
    # proposed in the report (Table 3.1) and the LLM-integration item in
    # the report's Future Scope (Section 5.2).
    """

    def __init__(self):
        self.vectoriser = TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            token_pattern=r"(?u)\b[a-zA-Z]{2,}\b",
        )
        self._fitted = False

    def fit(self, corpus: List[str]):
        self.vectoriser.fit(corpus)
        self._fitted = True
        # Pre-compute domain vectors once fitted.
        self.domain_names = list(DOMAIN_DESCRIPTORS.keys())
        self.domain_vectors = self.vectoriser.transform(
            [DOMAIN_DESCRIPTORS[d] for d in self.domain_names]
        )
        return self

    def transform(self, texts: List[str]):
        assert self._fitted, "ContextRepresenter must be fit() before use."
        return self.vectoriser.transform(texts)

    def best_domain(self, text_vector) -> Tuple[str, float]:
        """Return (domain_name, similarity) for the domain descriptor
        closest to `text_vector`."""
        sims = cosine_similarity(text_vector, self.domain_vectors)[0]
        idx = int(np.argmax(sims))
        return self.domain_names[idx], float(sims[idx])


class DomainIdentifier:
    """Wraps ContextRepresenter to answer: "which application domain does
    a WordNet sense (or a whole sentence) most likely belong to?"
    Corresponds to report Section 3.4 / Table 3.1 ("Domain Identification").
    """

    def __init__(self, representer: ContextRepresenter):
        self.representer = representer

    def domain_of_sentence(self, sentence: str) -> Tuple[str, float]:
        vec = self.representer.transform([sentence])
        return self.representer.best_domain(vec)

    def domain_of_sense(self, gloss: str) -> Tuple[str, float]:
        vec = self.representer.transform([gloss])
        return self.representer.best_domain(vec)


# --------------------------------------------------------------------------
# Stage 5: Ambiguity Score (report Section 3.6, Table 3.2)
# --------------------------------------------------------------------------
@dataclass
class SenseScore:
    synset_name: str
    gloss: str
    domain: str
    domain_similarity: float
    context_similarity: float


@dataclass
class AmbiguityResult:
    word: str
    surface_form: str
    sentence: str
    sentence_domain: str
    sentence_domain_similarity: float
    ranked_senses: List[SenseScore]
    semantic_similarity_spread: float
    contextual_confidence: float
    domain_diversity: float
    ambiguity_score: float
    is_ambiguous: bool
    ambiguity_type: str          # "Inter-Domain" | "Intra-Domain" | "None"
    explanation: str


class AmbiguityScorer:
    """Computes the Ambiguity Score from three factors, mirroring
    Table 3.2 of the report:
        - Semantic Similarity   -> closeness of context to each sense
        - Contextual Confidence -> how strongly context favours one sense
                                    over the runner-up sense
        - Domain Relevance      -> how many distinct domains the top
                                    candidate senses spread across

    A word is classified as ambiguous when `ambiguity_score` exceeds
    `threshold` (Algorithm 3.1, Step 8).
    """

    def __init__(self, threshold: float = 0.35, top_k: int = 3):
        self.threshold = threshold
        self.top_k = top_k

    def score(
        self,
        context_vector,
        candidate: CandidateWord,
        representer: ContextRepresenter,
        domain_identifier: DomainIdentifier,
    ) -> Tuple[List[SenseScore], float, float, float, float]:

        sense_scores: List[SenseScore] = []
        for syn in candidate.senses:
            gloss = syn.definition()
            gloss_vec = representer.transform([_enriched_gloss_text(syn)])
            context_sim = float(cosine_similarity(context_vector, gloss_vec)[0][0])
            domain, domain_sim = domain_identifier.domain_of_sense(_enriched_gloss_text(syn))
            sense_scores.append(
                SenseScore(
                    synset_name=syn.name(),
                    gloss=gloss,
                    domain=domain,
                    domain_similarity=domain_sim,
                    context_similarity=context_sim,
                )
            )

        # Rank senses by how well they match the sentence context.
        sense_scores.sort(key=lambda s: s.context_similarity, reverse=True)
        top = sense_scores[: self.top_k] if sense_scores else []

        # --- Semantic Similarity Spread ---------------------------------
        # TF-IDF cosine similarities between a one-line WordNet gloss and a
        # full sentence live on a small, noisy absolute scale (typically
        # 0.0-0.3), so an *absolute* gap is not informative. Instead we use
        # a *relative* gap between the top-2 candidate senses: if the two
        # closest senses are nearly equally good matches for the context
        # (relative gap -> 0), that is exactly the "genuinely ambiguous in
        # context" situation the report's Table 3.2 describes.
        _MIN_MEANINGFUL_SIM = 0.05  # below this, TF-IDF similarity is noise, not signal
        if len(top) >= 2 and top[0].context_similarity >= _MIN_MEANINGFUL_SIM:
            s1, s2 = top[0].context_similarity, top[1].context_similarity
            relative_gap = (s1 - s2) / (s1 + s2)
            semantic_similarity_spread = 1.0 - relative_gap  # near-tied senses -> near 1.0
        else:
            # No reliable contextual signal either way; stay neutral rather
            # than let near-zero-similarity noise masquerade as ambiguity.
            semantic_similarity_spread = 0.5

        # --- Contextual Confidence ---------------------------------------
        # Uses the *number of WordNet senses* available for the word as a
        # proxy for how much interpretive latitude the word carries in
        # general (this mirrors the WordNet-sense-count threshold already
        # used for candidate selection in Section 3.3), normalised to [0,1].
        n_senses = len(candidate.senses)
        contextual_confidence = min(1.0, max(0.0, (n_senses - 2) / 10.0))

        # --- Domain Relevance / Diversity --------------------------------
        distinct_domains = {s.domain for s in top}
        domain_diversity = (len(distinct_domains) - 1) / max(1, (len(top) - 1)) if len(top) > 1 else 0.0

        ambiguity_score = float(
            0.65 * semantic_similarity_spread
            + 0.35 * contextual_confidence
        )
        # NOTE: domain_diversity deliberately does NOT feed into the binary
        # ambiguity_score. It answers a different question ("if this word
        # IS ambiguous, do the competing senses cross a domain boundary?")
        # and is used downstream, in classify_ambiguity(), to decide the
        # Inter-Domain vs. Intra-Domain TYPE label -- not whether the word
        # is ambiguous in the first place. Folding it into the detection
        # score as well caused a large share of instances to be flagged
        # purely because *some* domain disagreement exists among senses,
        # even when the context gave a clear, confident best-match sense.
        return sense_scores, semantic_similarity_spread, contextual_confidence, domain_diversity, ambiguity_score


# --------------------------------------------------------------------------
# Stage 6: Inter / Intra-Domain Classification (report Section 3.5, Table 3.2)
# --------------------------------------------------------------------------
def classify_ambiguity(top_senses: List[SenseScore]) -> str:
    """Classify ambiguity as Inter-Domain or Intra-Domain by comparing the
    application domains of the top-2 contextually closest senses, per the
    classification criteria in report Table 3.2:
        - Inter-Domain: top senses map to *different* application domains.
        - Intra-Domain: top senses map to the *same* application domain but
          represent distinct concepts within it.
    """
    if len(top_senses) < 2:
        return "None"
    d1, d2 = top_senses[0].domain, top_senses[1].domain
    return "Inter-Domain" if d1 != d2 else "Intra-Domain"


def build_explanation(result_word: str, sentence_domain: str, top: List[SenseScore], ambiguity_type: str) -> str:
    if not top:
        return f'"{result_word}" has no WordNet senses available for analysis.'
    lines = [f'The word "{result_word}" appears in a sentence whose detected domain is "{sentence_domain}".']
    for i, s in enumerate(top[:2], start=1):
        lines.append(
            f"  Candidate sense {i}: [{s.domain}] {s.gloss} "
            f"(context similarity={s.context_similarity:.2f})"
        )
    if ambiguity_type == "Inter-Domain":
        lines.append(
            f'Classified as INTER-DOMAIN ambiguity: the top two candidate senses belong to '
            f'different application domains ("{top[0].domain}" vs. "{top[1].domain}").'
        )
    elif ambiguity_type == "Intra-Domain":
        lines.append(
            f'Classified as INTRA-DOMAIN ambiguity: the top two candidate senses both belong to '
            f'the "{top[0].domain}" domain but represent distinct concepts.'
        )
    else:
        lines.append("Not flagged as ambiguous in this context.")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# Stage 7: Orchestration / Final Ambiguity Report (Algorithm 3.1)
# --------------------------------------------------------------------------
class AmbiguityPipeline:
    """End-to-end orchestration of Algorithm 3.1, steps 1-9."""

    def __init__(
        self,
        min_wordnet_senses: int = 3,
        min_word_len: int = 3,
        ambiguity_threshold: float = 0.35,
        top_k_senses: int = 3,
    ):
        self.preprocessor = Preprocessor()
        self.candidate_detector = CandidateDetector(min_wordnet_senses, min_word_len)
        self.representer = ContextRepresenter()
        self.domain_identifier = DomainIdentifier(self.representer)
        self.scorer = AmbiguityScorer(threshold=ambiguity_threshold, top_k=top_k_senses)
        self._fitted = False

    def fit_vocabulary(self, sentences: List[str]):
        """Step 0 (setup): build the shared TF-IDF vector space (Stage 4)
        over a representative sample of requirement sentences plus all
        WordNet glosses that are reachable from those sentences' candidate
        words, plus the domain descriptor texts. Must be called once
        before `analyse_sentence`.
        """
        gloss_corpus = set()
        for sent in sentences:
            doc = self.preprocessor.process(sent)
            for cand in self.candidate_detector.detect(doc):
                for syn in cand.senses:
                    gloss_corpus.add(_enriched_gloss_text(syn))

        corpus = list(sentences) + list(gloss_corpus) + list(DOMAIN_DESCRIPTORS.values())
        self.representer.fit(corpus)
        self._fitted = True
        return self

    def analyse_sentence(self, sentence: str) -> List[AmbiguityResult]:
        assert self._fitted, "Call fit_vocabulary() before analyse_sentence()."

        doc = self.preprocessor.process(sentence)                       # Step 2
        candidates = self.candidate_detector.detect(doc)                # Step 3
        sentence_domain, sentence_domain_sim = self.domain_identifier.domain_of_sentence(sentence)  # Step 5
        context_vector = self.representer.transform([sentence])         # Step 6

        results: List[AmbiguityResult] = []
        for cand in candidates:
            (
                sense_scores,
                spread,
                confidence,
                diversity,
                ambiguity_score,
            ) = self.scorer.score(context_vector, cand, self.representer, self.domain_identifier)  # Step 7

            top = sorted(sense_scores, key=lambda s: s.context_similarity, reverse=True)[:3]
            is_ambiguous = ambiguity_score >= self.scorer.threshold and len(top) >= 2  # Step 8
            ambiguity_type = classify_ambiguity(top) if is_ambiguous else "None"

            explanation = build_explanation(cand.lemma, sentence_domain, top, ambiguity_type)  # Step 9

            results.append(
                AmbiguityResult(
                    word=cand.lemma,
                    surface_form=cand.surface_form,
                    sentence=sentence,
                    sentence_domain=sentence_domain,
                    sentence_domain_similarity=sentence_domain_sim,
                    ranked_senses=top,
                    semantic_similarity_spread=spread,
                    contextual_confidence=confidence,
                    domain_diversity=diversity,
                    ambiguity_score=ambiguity_score,
                    is_ambiguous=is_ambiguous,
                    ambiguity_type=ambiguity_type,
                    explanation=explanation,
                )
            )
        return results

    def analyse_corpus(self, sentences: List[str]) -> List[AmbiguityResult]:
        all_results = []
        for sent in sentences:
            all_results.extend(self.analyse_sentence(sent))
        return all_results


if __name__ == "__main__":
    sample_sentence = "The system shall create a single patient record for each patient."
    print("Initializing AmbiguityPipeline and running self-test...")
    pipeline = AmbiguityPipeline(ambiguity_threshold=0.3)
    pipeline.fit_vocabulary([sample_sentence])
    results = pipeline.analyse_sentence(sample_sentence)
    print(f"Sample sentence: \"{sample_sentence}\"")
    print(f"Detected {len(results)} candidate word(s):")
    for r in results:
        print(f"  - Word: '{r.word}' (surface: '{r.surface_form}') | Score: {r.ambiguity_score:.3f} | "
              f"Ambiguous: {r.is_ambiguous} | Type: {r.ambiguity_type}")
        if r.is_ambiguous:
            print(f"    Explanation: {r.explanation}")
    print("\n[SUCCESS] ambiguity_framework.py self-test completed.")
