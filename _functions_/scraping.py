#!/usr/bin/env python
# coding: utf-8

import random
import time
import pandas as pd

from helpers import get_sitemap_urls, get_article_urls, is_valid_article_url, scrape_article_full

def initiate_scraping(sitemap_xml_file: str) -> list[dict]:
    """
    Initiates the full scraping process using a sitemap index file.

    This function performs the following steps:
    - Retrieves individual sitemap URLs from the sitemap index file.
    - Extracts article URLs from each sitemap.
    - Scrapes each article and collects relevant metadata and analysis.

    Parameters:
        sitemap_xml_file (str): The URL of the sitemap index XML file.

    Returns:
        list[dict]: A list of dictionaries, each containing metadata and analysis
                    for one successfully scraped article.
    """
    sitemap_urls = get_sitemap_urls(sitemap_xml_file)

    all_article_urls = []
    article_records = []

    for sitemap_url in sitemap_urls:
        print(f"Parsing {sitemap_url}...")
        urls = get_article_urls(sitemap_url=sitemap_url)
        all_article_urls.extend(urls)
        time.sleep(random.randint(1, 5))

    for url in all_article_urls:
        if is_valid_article_url(url=url):
            print(f"Scraping: {url}")
            record = scrape_article_full(url=url)
            if record:
                article_records.append(record)
            time.sleep(random.randint(1, 5))

    return article_records


def convert_to_dataframe_and_save_csv(title: str, data: list[dict]) -> None:
    """
    Converts the given data into a pandas DataFrame, saves it as a CSV file, and prints a summary.

    Args:
        title (str): The name of the CSV file (without extension) to save the DataFrame to.
        data: A list of dictionaries or a similar structure compatible with pandas DataFrame construction.

    Side Effects:
        - Saves the DataFrame as a CSV file named '{title}.csv' in the current directory.
        - Prints the number of processed articles.
        - Displays the first few rows of selected columns: 'url', 'top_loanwords', and 'all_loanwords'.

    Returns:
        None
    """
    df = pd.DataFrame(data)
    df.to_csv(f"{title}.csv", index=False)
    print(f"\n Scraped and processed {len(df)} articles.")
    print(df[["url", "top_loanwords", "all_loanwords"]].head())
