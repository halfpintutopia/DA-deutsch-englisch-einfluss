#!/usr/bin/env python
# coding: utf-8

import random
import time
import pandas as pd
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from tqdm import tqdm
from typing import Optional, Callable
import requests

from helpers import get_sitemap_urls, get_article_urls, is_valid_article_url, scrape_article_full


def scrape_with_retries(
    url: str,
    output_csv: str,
    scrape_func: Callable[[], None],
    done_urls,
    csv_lock,
    max_retries: int = 3,
    delay_range: tuple = (1.5, 3.5),
) -> Optional[dict]:
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

    for attempt in range(1, max_retries + 1):
        try:
            print(f"Function: {scrape_func}, url: {url}")
            result = scrape_func(url)
            if result:
                with csv_lock:
                    pd.DataFrame(
                        [result]
                    ).to_csv(
                        output_csv, mode="a", index=False, header=not Path(output_csv).exists()
                    )
                return result
        except (requests.RequestException, ValueError) as e:
            print(f"{url} Attempt {attempt} failed: {e}")

        sleep_time = random.uniform(*delay_range) * attempt
        time.sleep(sleep_time)

    print(f"[Failed] {url} after {max_retries} retries")
    return None


def initiate_scraping_in_parallel(
        urls: list[str],
        output_csv: str,
        scrape_func: Callable[[], None],
        max_workers: int = 5,
        max_retries: int = 3,
        delay_range: tuple = (1.5, 3.5),
        rerume: bool = True

):
    csv_lock = Lock()

    # Resume
    if Path(output_csv).exists() and rerume:
        existing_df = pd.read_csv(output_csv)
        done_urls = set(existing_df["url"].tolist())
    else:
        existing_df = pd.DataFrame()
        done_urls = set()

    # Check and filer already scraped
    remaining_urls = [u for u in urls if u not in done_urls]

    print(
        f"\n Starting parallel scrape with {max_workers} threads on {len(remaining_urls)} URLs...\n")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                scrape_with_retries,
                url,
                scrape_func,
                done_urls,
                output_csv,
                csv_lock,
                max_retries,
                delay_range
            ): url
            for url in remaining_urls
        }
        for future in tqdm(as_completed(futures), total=len(futures), desc="Scraping"):
            _ = future.result()

        print(
            f"\n Done! Total scraped articles:{len(existing_df) + len(futures)}")
        

def scrape_with_retries_2(
    url,
    scrape_func,
    done_urls,
    output_csv,
    csv_lock,
    max_retries=3,
    delay_range=(1.5, 3.5)
):
    import pandas as pd
    import time
    import random
    from pathlib import Path

    if url in done_urls:
        return None

    for attempt in range(1, max_retries + 1):
        try:
            result = scrape_func(url)
            if result:
                with csv_lock:
                    pd.DataFrame([result]).to_csv(
                        output_csv, mode='a', index=False, header=not Path(output_csv).exists()
                    )
                return result
        except Exception as e:
            print(f"[{url}] Attempt {attempt} failed: {e}")
        time.sleep(random.uniform(*delay_range) * attempt)

    print(f"[Failed] {url} after {max_retries} retries.")
    return None


def run_parallel_scraper(
    urls,
    output_csv,
    scrape_func,
    max_workers=5,
    max_retries=3,
    delay_range=(1.5, 3.5),
    resume=True
):
    import pandas as pd
    from pathlib import Path
    from threading import Lock
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from tqdm import tqdm

    csv_lock = Lock()

    if Path(output_csv).exists() and resume:
        existing_df = pd.read_csv(output_csv)
        done_urls = set(existing_df["url"].tolist())
    else:
        existing_df = pd.DataFrame()
        done_urls = set()

    remaining_urls = [u for u in urls if u not in done_urls]

    print(
        f"\n🚀 Starting parallel scrape with {max_workers} threads on {len(remaining_urls)} URLs...\n")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                scrape_with_retries_2,
                url,
                scrape_func,
                done_urls,
                output_csv,
                csv_lock,
                max_retries,
                delay_range
            ): url
            for url in remaining_urls
        }
        for future in tqdm(as_completed(futures), total=len(futures), desc="Scraping"):
            _ = future.result()

    print(
        f"\n✅ Done! Total scraped articles: {len(existing_df) + len(futures)}")
