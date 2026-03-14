"""Preprocessing pipeline for ArXiv papers."""

import logging
import re

import pandas as pd
import spacy

logger = logging.getLogger(__name__)

TARGET_CATEGORIES = {
    "cs.AI",
    "cs.LG",
    "cs.CL",
    "cs.CV",
    "cs.RO",
    "cs.NE",
    "cs.MA",
    "cs.IR",
}

MIN_ABSTRACT_WORDS = 30

SURVEY_PATTERN = re.compile(
    r"\b(survey|review|overview|tutorial|systematic review"
    r"|literature review|comprehensive review|state of the art|sota review)\b",
    re.IGNORECASE,
)


def filter_papers(df, categories=TARGET_CATEGORIES, min_words=MIN_ABSTRACT_WORDS):
    """Filter to target categories and drop short abstracts."""
    before = len(df)
    df = df[df["primary_category"].isin(categories)].copy()
    logger.info(f"Category filter: {before:,} -> {len(df):,} papers")

    before = len(df)
    df = df[df["abstract"].str.split().str.len() >= min_words].copy()
    logger.info(f"Short abstract filter: {before:,} -> {len(df):,} papers")

    df = df.reset_index(drop=True)
    return df


def clean_abstracts(df):
    """Normalize whitespace and create title+abstract column for embedding."""
    cleaned = df["abstract"].str.replace(r"\s+", " ", regex=True)
    df["abstract_clean"] = cleaned.str.strip()
    df["embed_text"] = df["title"] + ". " + df["abstract_clean"]
    return df


def flag_surveys(df):
    """Flag survey/review papers based on title keywords."""
    df["is_survey"] = df["title"].apply(lambda t: bool(SURVEY_PATTERN.search(str(t))))
    n_surveys = df["is_survey"].sum()
    logger.info(f"Flagged {n_surveys:,} survey papers ({n_surveys / len(df):.1%})")
    return df


def lemmatize_abstracts(df, batch_size=1000):
    """Lemmatize abstracts: lowercase, remove stopwords, keep alpha tokens > 2 chars."""
    nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])

    lemmatized = []
    for doc in nlp.pipe(df["abstract_clean"], batch_size=batch_size):
        tokens = [
            t.lemma_.lower() for t in doc if not t.is_stop and t.is_alpha and len(t) > 2
        ]
        lemmatized.append(" ".join(tokens))

    df["abstract_lemma"] = lemmatized
    logger.info(f"Lemmatized {len(df):,} abstracts")
    return df


def preprocess_pipeline(raw_path, output_path):
    """Full pipeline: load, filter, clean, flag, lemmatize, save."""
    df = pd.read_csv(raw_path)
    logger.info(f"Loaded {len(df):,} papers from {raw_path}")

    df = filter_papers(df)
    df = clean_abstracts(df)
    df = flag_surveys(df)
    df = lemmatize_abstracts(df)

    df.to_csv(output_path, index=False)
    logger.info(f"Saved {len(df):,} papers to {output_path}")
    return df
