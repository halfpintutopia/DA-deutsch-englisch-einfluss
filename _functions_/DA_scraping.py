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

headers = {"User-Agent": "Mozilla/5.0"}
nlp = spacy.load("de.core_news_sm")
sitemap_index_url = "https://www.businessinsider.de/sitemap_index.xml"


def get_sitemap_urls(index_url):
    res = requests.get(index_url, headers=headers)
    soup = BeautifulSoup(res.text, "xml")
    return [loc.text for loc in soup.find_all("loc") if "post-sitemap" in loc.text]


def get_article_urls(sitemap_url):
    res = requests.get(sitemap_url, headers=headers)
    soup = BeautifulSoup(res.text, "xml")
    paragraphs = soup.find_all("p")
    return [loc.text for loc in soup.find_all("loc")]


all_article_urls = []

for sitemap_url in sitemap_urls:
    print(f"Parsing {sitemap_url}...")
    urls = get_article_urls(sitemap_url)
    all_article_urls.extend(urls)
    time.sleep(random.randint(1, 5))

print(f"Collected {len(all_article_urls)} article URLs")

sitemap_urls = get_sitemap_urls(sitemap_index_url)

print(f"Found {len(sitemap_urls)} sitemap files.")


nlp = spacy.load("de_core_news_sm")


def detect_loanwords_from_articles(articles_list: list[str]):
    """
    Detects and returns a list of potential English loanwords from a list of articles.

    This function processes each article using a spaCy NLP pipeline, tokenizes the text,
    and filters out non-alphabetic tokens, stop words, and words shorter than 4 characters.
    It then uses language detection to identify words that are likely English and collects them.

    Parameters:
        articles_list (list of str): A list of article texts to analyze.

    Returns:
        list of str: A list of lowercase English words that may be loanwords in the context
        of the input articles.
    """
    all_loanwords = []

    for text in articles_list:
        doc = nlp(text)
        for token in doc:
            if token.is_alpha and not token.is_stop:
                word = token.text
                if re.match(r"^[A-Za-z]{4,}$", word):
                    try:
                        if detect(word) == "en":
                            all_loanwords.append(word.lower())
                    except:
                        continue
    return all_loanwords
