#!/usr/bin/env python
# coding: utf-8

import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse
from typing import Optional

headers = {
    "User-Agent": "Mozilla/5.0"
}

from scraping_helpers import (
    is_valid_article_url,
    extract_meta_data,
    extract_jsonld_date,
    extract_year_from_url,
    extract_headline
)


def scrape_article_full(url: str) -> Optional[dict]:
    """
    Scrapes an article from the given URL and extracts metadata, text content,
    loanword analysis, and sentiment.

    This function performs the following:
    - Downloads the article content
    - Parses HTML to extract paragraphs
    - Extracts publication date (from metadata, JSON-LD, or URL)
    - Calculates word count, loanword stats, and sentiment
    - Returns all information in a dictionary

    Parameters:
        url (str): The URL of the article to scrape.

    Returns:
        dict or None: A dictionary containing article data and analysis results.
        Returns None if scraping or parsing fails.
    """
    if not is_valid_article_url(url=url):
        print(f"Skipping non-article URL: {url}")
        return None

    try:
        res = requests.get(url, headers=headers, timeout=10)

        if "text/html" not in res.headers.get("Content-Type", ""):
            print(f"Non-HTML content for {url}")
            return None

        res.encoding = res.apparent_encoding
        soup = BeautifulSoup(res.text, "html.parser")
        paragraphs = soup.find_all("p")

        if len(paragraphs) < 5:
            print(f"Too few paragraphs at {url}")
            return None

        text = " ".join(p.get_text() for p in paragraphs).strip()
        if not text:
            print(f"No text extracted from {url}")
            return None

        text = " ".join(p.get_text() for p in paragraphs)
        date = extract_meta_data(soup=soup) or extract_jsonld_date(
            soup=soup) or extract_year_from_url(url=url)
        year = date.split("-")[0] if date else None
        domain = urlparse(url).netloc
        source_site = domain.replace("www.", "")
        headline = extract_headline(soup=soup)
        word_count = len(text.split())

        return {
            "url": url,
            "source_site": source_site,
            "domain": domain.split(".")[0],
            "date": date,
            "year": int(year) if year else None,
            "text": text,
            "headline": headline,
            "word_count": word_count,
            "paragraphs": len(paragraphs)
        }

    except (requests.RequestException, AttributeError, ValueError) as e:
        print(f"Error scraping {url}: {e}")
        return None
