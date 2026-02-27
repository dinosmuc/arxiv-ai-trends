"""
Collect AI research papers from ArXiv for trend analysis.
"""

import logging
import time
from datetime import datetime, timezone
from pathlib import Path

import arxiv
import pandas as pd

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# ArXiv CS categories covering the AI research landscape
CATEGORIES = ["cs.AI", "cs.LG", "cs.CL", "cs.CV", "cs.RO", "cs.NE", "cs.MA", "cs.IR"]

# Everything from Jan 2024 to now
START_DATE = datetime(2024, 1, 1, tzinfo=timezone.utc)
END_DATE = datetime.now(timezone.utc)


def get_monthly_windows(start, end):
    """Break the date range into half-month chunks for the API."""
    windows = []
    current = start

    while current < end:
        if current.day == 1:
            mid = current.replace(day=16)
            window_end = min(mid, end)
        else:
            if current.month == 12:
                next_month = current.replace(year=current.year + 1, month=1, day=1)
            else:
                next_month = current.replace(month=current.month + 1, day=1)
            window_end = min(next_month, end)

        windows.append((current, window_end))
        current = window_end

    return windows


def make_query(window_start, window_end):
    """Build ArXiv query string: categories + date range."""
    cats = " OR ".join(f"cat:{c}" for c in CATEGORIES)
    start_str = window_start.strftime("%Y%m%d%H%M")
    end_str = window_end.strftime("%Y%m%d%H%M")
    return f"({cats}) AND submittedDate:[{start_str} TO {end_str}]"


def extract_paper_info(result):
    """Flatten an ArXiv result into a dict."""
    return {
        "arxiv_id": result.get_short_id(),
        "title": result.title.replace("\n", " ").strip(),
        "abstract": result.summary.replace("\n", " ").strip(),
        "authors": ", ".join(a.name for a in result.authors),
        "primary_category": result.primary_category,
        "categories": ", ".join(result.categories),
        "published": result.published.strftime("%Y-%m-%d"),
        "updated": result.updated.strftime("%Y-%m-%d"),
    }


def fetch_one_window(client, window_start, window_end):
    """Fetch all papers in a single time window."""
    query = make_query(window_start, window_end)
    label = window_start.strftime("%Y-%m-%d")

    search = arxiv.Search(
        query=query,
        max_results=None,
        sort_by=arxiv.SortCriterion.SubmittedDate,
        sort_order=arxiv.SortOrder.Ascending,
    )

    papers = []
    for r in client.results(search):
        papers.append(extract_paper_info(r))

    logger.info(f"  {label}: got {len(papers)} papers")
    return papers


def collect_papers(output_dir="data/raw", start_date=START_DATE, end_date=END_DATE):
    """
    Pull all AI papers from ArXiv, half-month by half-month.
    Each window is saved as a checkpoint CSV. Already-fetched windows are skipped.
    Returns a single deduplicated DataFrame.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    client = arxiv.Client(page_size=2000, delay_seconds=3.0, num_retries=5)

    windows = get_monthly_windows(start_date, end_date)
    logger.info(f"Collecting {len(windows)} windows of AI papers from ArXiv")

    window_dfs = []

    for w_start, w_end in windows:
        label = w_start.strftime("%Y-%m-%d")
        checkpoint = output_dir / f"arxiv_{label}.csv"

        if checkpoint.exists():
            logger.info(f"  {label}: found checkpoint, skipping")
            window_dfs.append(pd.read_csv(checkpoint))
            continue

        try:
            papers = fetch_one_window(client, w_start, w_end)
        except Exception as e:
            logger.error(f"  {label}: failed — {e}")
            continue

        if not papers:
            logger.warning(f"  {label}: empty — no papers in this window")
            continue

        df = pd.DataFrame(papers)
        df.to_csv(checkpoint, index=False)
        window_dfs.append(df)
        time.sleep(1)

    if not window_dfs:
        logger.error("Nothing collected — check your connection and date range")
        return pd.DataFrame()

    all_papers = pd.concat(window_dfs, ignore_index=True)
    n_raw = len(all_papers)
    all_papers = all_papers.drop_duplicates(subset="arxiv_id", keep="first")
    n_unique = len(all_papers)
    logger.info(
        f"Collected {n_raw} total, {n_unique} unique "
        f"({n_raw - n_unique} duplicates removed)"
    )

    return all_papers.sort_values("published", ascending=False).reset_index(drop=True)


if __name__ == "__main__":
    df = collect_papers()
    print(f"\nFinal dataset shape: {df.shape}")
