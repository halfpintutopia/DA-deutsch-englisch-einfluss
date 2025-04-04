#!/usr/bin/env python
# coding: utf-8

import requests
from bs4 import BeautifulSoup
import time
import random
import re
import spacy
from langdetect import detect
from collections import Counter
import pandas as pd
from urllib.parse import urlparse
import json
from langdetect.lang_detect_exception import LangDetectException
from transformers import pipeline
from typing import Optional

headers = {
    "User-Agent": "Mozilla/5.0"
}
sentiment_model = pipeline(
    "sentiment-analysis", model="oliverguhr/german-sentiment-bert")

nlp = spacy.load("de_core_news_sm")  # de_core_news_lg best for accuracy


def get_sitemap_urls(index_url: str) -> list[str]:
    """
    Retrieves a list of sitemap URLs from a sitemap index XML file.

    This function sends a GET request to the provided sitemap index URL,
    parses the XML content, and returns a list of URLs that contain
    the substring "post-sitemap" (commonly used for blog post sitemaps).

    Parameters:
        index_url (str): The URL of the sitemap index file (typically ending in .xml).

    Returns:
        list[str]: A list of sitemap URLs filtered to include only those related to posts.
    """
    res = requests.get(index_url, headers=headers)
    soup = BeautifulSoup(res.text, "xml")
    return [loc.text for loc in soup.find_all("loc") if "post-sitemap" in loc.text]


def get_article_urls(sitemap_url: str) -> list[str]:
    """
    Extracts all article URLs from a given sitemap XML URL.

    This function sends a GET request to the provided sitemap URL,
    parses the XML content, and extracts all URLs listed within
    <loc> tags. It assumes the sitemap contains direct links to articles.

    Parameters:
        sitemap_url (str): The URL of the sitemap XML file containing article links.

    Returns:
        list[str]: A list of article URLs found in the sitemap.
    """
    res = requests.get(sitemap_url, headers=headers)
    soup = BeautifulSoup(res.text, "xml")

    return [loc.text for loc in soup.find_all("loc")]


# TODO change the startswith url to be reusable
def is_valid_article_url(url: str) -> bool:
    """
    Check if a given URL is a valid article link.

    A valid article URL is defined as one that:
    - Ends with "html"
    - Does not contain any of the substrings listed in `non_article_pages`

    Parameters:
        url (str): The URL to validate.

    Returns:
        bool: True if the URL is considered a valid article URL, False otherwise.
    """
    non_article_pages = ["/video/,", ".jpg", ".jpeg",
                         ".png", ".gif", "/bilder/", "/photo/", "/live/"]

    return (
        url.startswith("https://www.businessinsider.de/")
        and not any(pattern in url for pattern in non_article_pages)
    )


def extract_meta_data(soup: BeautifulSoup) -> Optional[str]:
    """
    Extracts the publication date from a BeautifulSoup-parsed HTML document.

    This function looks for a meta tag with the property "article:published_time"
    and extracts the date portion from its "content" attribute, formatted as YYYY-MM-DD.

    Parameters:
        soup (BeautifulSoup): A BeautifulSoup object representing the parsed HTML content.

    Returns:
        str or None: The publication date as a string in 'YYYY-MM-DD' format if found,
        otherwise None.
    """
    meta_date = soup.find("meta", {"property": "article:published_time"})
    if meta_date:
        return meta_date.get("content", "").split("T")[0]
    return None


def extract_jsonld_date(soup: BeautifulSoup) -> Optional[str]:
    """
    Extracts the publication date from JSON-LD structured data within an HTML document.

    This function searches for <script> tags of type "application/ld+json", parses their content
    as JSON, and attempts to extract the "datePublished" field. If found, it returns the date
    in 'YYYY-MM-DD' format.

    Parameters:
        soup (BeautifulSoup): A BeautifulSoup object representing the parsed HTML content.

    Returns:
        str or None: The publication date as a string in 'YYYY-MM-DD' format if found,
        otherwise None.
    """
    scripts = soup.find_all("script", {"type": "application/ld+json"})
    for script in scripts:
        try:
            data = json.loads(script.string)
            if isinstance(data, dict) and "datePublished" in data:
                return data["datePublished"].split("T")[0]
        except (json.JSONDecodeError, TypeError):
            continue
    return None


def extract_year_from_url(url: str) -> Optional[str]:
    """
    Extracts a four-digit year (starting with 20) from a URL string.

    This function searches for a year in the format '/20xx/' within the URL
    using a regular expression and returns the first match found.

    Parameters:
        url (str): The URL string to search for a year.

    Returns:
        str or None: The extracted year as a string (e.g., "2023") if found,
        otherwise None.
    """
    match = re.search(r"/(20\d{2})/", url)
    if match:
        return match.group(1)
    return None


def extract_headline(soup: BeautifulSoup) -> Optional[str]:
    """
    Extract the headline from a BeautifulSoup-parsed HTML document.

    The function attempts to retrieve the headline in the following order:
    1. The content of the <meta property="og:title"> tag (Open Graph title)
    2. The content of the <title> tag
    3. The text inside the first <h1> tag

    Parameters:
        soup (BeautifulSoup): A BeautifulSoup object representing the HTML document.

    Returns:
        str or None: The extracted headline as a string, or None if no suitable element is found.
    """
    og_title = soup.find("meta", property="og:title")
    if og_title and og_title.get("content"):
        return og_title["content"].strip()

    title_tag = soup.find("title")
    if title_tag:
        return title_tag.get_text(strip=True)

    h1 = soup.find("h1")
    if h1:
        return h1.get_text(strip=True)

    return None


def detect_loanwords(text: str) -> list[str]:
    """
    Detects potential English loanwords in a given text.

    This function tokenizes the input text using a spaCy NLP pipeline,
    filters out stop words and non-alphabetic tokens, and checks if
    each remaining word is likely to be English using a language detection
    library. Only words with 4 or more alphabetic characters are considered.

    Parameters:
        text (str): The input text to analyze.

    Returns:
        list of str: A list of lowercase English words that are likely
        loanwords in the context of the input text.
    """
    doc = nlp(text)
    loanwords = []
    for token in doc:
        if token.is_alpha and not token.is_stop:
            word = token.text
            if re.match(r"^[A-Za-z]{4,}$", word):
                try:
                    if detect(word) == "en":
                        loanwords.append(word.lower())
                except LangDetectException:
                    continue
    return loanwords


def analyse_sentiment(text: str) -> Optional[str]:
    """
    Analyses the sentiment of the given text using a preloaded sentiment model.

    This function takes the first 516 characters of the input text, performs sentiment
    analysis using the `sentiment_model`, and returns the predicted sentiment label
    in lowercase (e.g., 'positive', 'negative', 'neutral').

    Parameters:
        text (str): The input text to analyze.

    Returns:
        str: The sentiment label in lowercase. Returns "unknown" if analysis fails.
    """
    try:
        result = sentiment_model(text[:516])[0]
        return result["label"].lower()
    except (KeyError, IndexError, TypeError) as e:
        print(f"Sentiment analysis error: {e}")
        return "unknown"
