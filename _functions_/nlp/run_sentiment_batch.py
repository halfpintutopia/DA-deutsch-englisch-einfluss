#!/usr/bin/env python
# coding: utf-8
# pylint: skip-file


import pandas as pd
import logging
from sentiment_helpers import batch_analyse_sentiment_fast

# === CONFIGURATION ===
INPUT_CSV = "scraped_articles_clean_v1"
OUTPUT_CSV = "scraped_articles_enriched.csv"
TEXT_COLUMN = "text"
BATCH_SIZE = 32
LOG_FILE = "sentiment_batch.log"


# === LOGGING SETUP ===
logging.basicConfig(
    filename=LOG_FILE,
    filemode="a",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logging.info("Sentiment batch enrichment started")


# === LOAD DATA ===
logging.info("Loading input CSV")
df = pd.read_csv(INPUT_CSV)


# === RUN SENTIMENT PIPELINE ===
logging.info("Running fast sentiment analysis...")
df = batch_analyse_sentiment_fast(
    df, text_column=TEXT_COLUMN, new_column="sentiment", batch_size=BATCH_SIZE
)


# === SAVE OUTPUT ===
logging.info("Saving output to CSV")
df.to_csv(OUTPUT_CSV, index=False)
logging.info(f"Done! Output saved to {OUTPUT_CSV}")
