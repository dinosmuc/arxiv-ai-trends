"""Smoke tests for preprocessing pipeline functions."""

import pandas as pd

from src.preprocess import clean_abstracts, filter_papers, flag_surveys


def test_filter_papers_keeps_only_target_categories():
    df = pd.DataFrame(
        {
            "primary_category": ["cs.AI", "cs.LG", "cs.CR", "cs.DB"],
            "abstract": [" ".join(["word"] * 50)] * 4,
        }
    )
    out = filter_papers(df)
    assert set(out["primary_category"]) == {"cs.AI", "cs.LG"}


def test_filter_papers_drops_abstracts_under_30_words():
    df = pd.DataFrame(
        {
            "primary_category": ["cs.AI"] * 3,
            "abstract": [
                " ".join(["word"] * 10),
                " ".join(["word"] * 29),
                " ".join(["word"] * 30),
            ],
        }
    )
    out = filter_papers(df)
    assert len(out) == 1


def test_flag_surveys_matches_survey_titles():
    df = pd.DataFrame(
        {
            "title": [
                "A Survey of Deep Learning Methods",
                "Literature Review of Large Language Models",
                "Tutorial on Diffusion Models",
            ]
        }
    )
    out = flag_surveys(df)
    assert out["is_survey"].all()


def test_flag_surveys_ignores_non_survey_titles():
    df = pd.DataFrame(
        {
            "title": [
                "Novel Attention Mechanism for Transformers",
                "Efficient Training Methods via Sparse Gradients",
                "Graph Learning Framework for Drug Discovery",
            ]
        }
    )
    out = flag_surveys(df)
    assert not out["is_survey"].any()


def test_clean_abstracts_normalizes_whitespace():
    df = pd.DataFrame({"title": ["Foo"], "abstract": ["Hello   \n\t world"]})
    out = clean_abstracts(df)
    assert out["abstract_clean"].iloc[0] == "Hello world"


def test_clean_abstracts_creates_embed_text():
    df = pd.DataFrame({"title": ["Foo"], "abstract": ["Bar"]})
    out = clean_abstracts(df)
    assert out["embed_text"].iloc[0] == "Foo. Bar"
