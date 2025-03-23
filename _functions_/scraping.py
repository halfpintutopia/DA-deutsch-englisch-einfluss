#!/usr/bin/env python
# coding: utf-8

import random
import time
import pandas as pd
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from tqdm import tqdm
from typing import Optional
import requests

from helpers import get_sitemap_urls, get_article_urls, is_valid_article_url, scrape_article_full

# Config Variables
URL_LIST_PATH = "businessinsider_urls.txt"
OUTPUT_CSV = "scraped_articles_parallel.csv"
MAX_WORKER = 5
MAX_RETRIES = 3
DELAY_RANGE = (1.5, 3.5)
RESUME = True

# Load URL list
with open(URL_LIST_PATH, "r", encoding="utf-8") as f:
    all_urls = [line.strip() for line in f if line.strip()]


def scrape_with_retries(url: str) -> Optional[dict]:
    """
    Attempt to scrape an article from a given URL with multiple retries.

    This function:
    - Skips URLs that have already been processed (present in `done_urls`)
    - Retries scraping up to `MAX_RETRIES` times in case of failure
    - Applies an increasing delay (with randomness) between retries
    - Saves successful results to a CSV file in a thread-safe way using `csv_lock`

    Parameters:
        url (str): The URL of the article to scrape.

    Returns:
        dict or None: The scraped article data as a dictionary if successful,
        or None if all retries fail or URL was already processed.
    """
    if url in done_urls:
        return None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            result = scrape_article_full(url=url)
            if result:
                with csv_lock:
                    pd.DataFrame(
                        [result]
                    ).to_csv(
                        OUTPUT_CSV, mode="a", index=False, header=not Path(OUTPUT_CSV).exists()
                    )
                return result
        except (requests.RequestException, ValueError) as e:
            print(f"{url} Attempt {attempt} failed: {e}")

        sleep_time = random.uniform(*DELAY_RANGE) * attempt
        time.sleep(sleep_time)

    print(f"[Failed] {url} after {MAX_RETRIES} retries")
    return None
