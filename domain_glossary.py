# -*- coding: utf-8 -*-
"""
domain_glossary.py
-------------------
Domain-specific glossaries used by the framework's "Domain Identification"
module (Section 3.4 of the project report / Figure 3.1, stage 4).

Each domain is represented by a short descriptor text built from
representative vocabulary for that application domain. These descriptor
texts are vectorised (TF-IDF) alongside WordNet sense definitions and
requirement-sentence contexts so that cosine similarity can be used to:
    (a) identify the most likely application domain of a requirement
        sentence, and
    (b) identify the most likely application domain of an individual
        WordNet sense (gloss) of a candidate polysemous word.

The candidate glossary (CANDIDATE_GLOSSARY_TERMS) supplements WordNet-based
candidate detection with a curated list of terms that are known, from the
literature and from manual inspection of the PURE requirements dataset, to
carry different meanings across the Software/IT and Healthcare domains
(e.g. "record", "chart", "note", "session", "key", "field", "plan").
This mirrors the "Domain Glossary" lexical resource in the proposed
framework (Section 3.3 / Figure 3.3), since a BabelNet API is not available
in this offline implementation.
"""

DOMAIN_DESCRIPTORS = {
    "Software/IT": (
        "system software application database server client interface "
        "program code module function process file record field key "
        "table index session login password network protocol port "
        "screen user interface data storage memory cache log error "
        "exception configuration parameter query input output display "
        "browser website application programming interface api"
    ),
    "Healthcare": (
        "patient physician nurse clinician doctor diagnosis medication "
        "prescription treatment allergy immunization vaccine provider "
        "clinic hospital chart record encounter visit vital sign lab "
        "laboratory result problem list care plan discharge admission "
        "medical history symptom disease condition dosage drug therapy"
    ),
    "Networking": (
        "network port protocol packet router switch firewall bandwidth "
        "connection socket ethernet ip address gateway node latency "
        "transmission wireless server client host traffic subnet"
    ),
    "Legal/Regulatory": (
        "policy regulation compliance law statute rule authorization "
        "consent privacy confidentiality hipaa audit disclosure "
        "requirement standard certification obligation liability "
        "governance security rule breach notification"
    ),
    "Finance/Business": (
        "payment billing invoice account balance transaction cost fee "
        "insurance claim reimbursement charge revenue budget contract "
        "vendor procurement price rate financial statement"
    ),
    "General": (
        "information process activity item entity object event action "
        "operation task step procedure method report document"
    ),
}

# Curated terms known (from the requirements-ambiguity literature and manual
# inspection of the PURE / EHR requirements corpus) to be genuinely
# polysemous across the Software/IT and Healthcare domains, even when their
# WordNet sense-count alone might not clear the automatic threshold.
CANDIDATE_GLOSSARY_TERMS = {
    "record", "chart", "note", "session", "key", "field", "plan", "order",
    "code", "history", "discharge", "provider", "claim", "policy", "port",
    "table", "log", "history", "exchange", "transaction", "profile",
    "summary", "report", "flag", "status", "role", "account", "identifier",
    "authorization", "reference", "template", "form", "review", "alert",
    "message", "queue", "encounter",
}


if __name__ == "__main__":
    print("Domain Glossary Module")
    print("----------------------")
    print(f"Total Domains Defined: {len(DOMAIN_DESCRIPTORS)}")
    for d, desc in DOMAIN_DESCRIPTORS.items():
        print(f" - {d}: {len(desc.split())} descriptor terms")
    print(f"\nTotal Candidate Glossary Terms: {len(CANDIDATE_GLOSSARY_TERMS)}")
    print(f"Terms: {', '.join(sorted(CANDIDATE_GLOSSARY_TERMS))}")
    print("\n[SUCCESS] domain_glossary.py executed successfully.")
