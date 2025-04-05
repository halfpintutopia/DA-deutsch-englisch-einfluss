#!/usr/bin/env python
# coding: utf-8
# pylint: skip-file

from transformers import pipeline
import pandas as pd 
from tqdm import tqdm
from typing import List, Optional
import numpy as np 

sentiment_model = pipeline(
    # for speed: model="nlptown/bert-base-multilingual-uncased-sentiment"
    "sentiment-analysis", model="oliverguhr/german-sentiment-bert"
)


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
        result = sentiment_model(text)[0]
        return result["label"].lower()
    except (KeyError, IndexError, TypeError) as e:
        print(f"Sentiment analysis error: {e}")
        return "unknown"


def batch_analyse_sentiment(
    df: pd.DataFrame, 
    text_column: str = "text",
    new_column: str = "sentiment"
) -> pd.DataFrame:
    tqdm.pandas(desc="Sentiment Analysis")
    df[new_column] = df[text_column].progress_apply(analyse_sentiment)
    return df


def batch_analyse_sentiment_fast(
    df: pd.DataFrame,
    text_column: str = "text",
    new_column: str = "sentiment",
    batch_size: int = 32
) -> pd.DataFrame:
    sentiments = []
    texts = df[text_column].fillna("").astype(str).tolist()

    for i in tqdm(range(0, len(texts), batch_size), desc="Batch Sentiment Analysis"):
        batch = texts[i:i+batch_size]
        try:
            results = sentiment_model(batch)
            labels = [r["label"].lower() if isinstance(r, dict) else "unknown" for r in results]
        except Exception as e:
            print(f"Batch failed at index {i}: {e}")
            labels = ["unknown"] * len(batch)
        sentiments.extend(labels)

    df[new_column] = sentiments
    return df