#!/usr/bin/env python
# coding: utf-8

import os
import pandas as pd
import json
import time
import logging
from tqdm import tqdm
from llm_helpers import (
    classify_tone,
    classify_topic,
    summarise_article,
    explain_loanwords_usage,
    detect_marketing_loanwords,
    detect_country_influence,
    detect_unwanted_loanwords
)

print("🚀 LLM enrichment pipeline starting...")


def process_scraped_csv_in_batches(
    input_csv: str = "scraped_articles_parallel.csv",
    output_csv: str = "enriched_articles.csv",
    checkpoint_csv: str = "llm_enrich_checkpoint.csv",
    batch_size: int = 50,
    limit: int = 5,
    log_path: str = "llm_enrichment.log"
):
    logger = logging.getLogger("llm_enrichment")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = logging.FileHandler(log_path)
        formatter = logging.Formatter('%(asctime)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    processed_ids = set()
    if not os.path.exists(input_csv):
        logger.error(f"❌ Input CSV not found: {input_csv}")
        print(f"❌ Input CSV not found: {input_csv}")
        return

    all_rows = pd.read_csv(input_csv)
    if "article_id" not in all_rows.columns:
        logging.error("❌ 'article_id' column not found in input CSV")
        print("❌ 'article_id' column not found in input CSV")
        return
    all_rows = all_rows[~all_rows["article_id"].isin(processed_ids)]
    if limit:
        all_rows = all_rows.head(limit)

    print(f"Starting enrichment on {len(all_rows)} articles...")

    new_rows = []

    for i in tqdm(range(0, len(all_rows), batch_size), desc="LLM Enrichment"):
        batch = all_rows.iloc[i:i+batch_size]
        for _, row in batch.iterrows():
            try:
                article_id = row["article_id"]
                text = row["text"]

                enriched = {
                    "article_id": article_id,
                    "tone": classify_tone(text),
                    "topic": classify_topic(text),
                    "summary": summarise_article(text),
                    "loanword_context": explain_loanwords_usage(text),
                    "marketing_loanwords": detect_marketing_loanwords(text),
                    "us_influence": detect_country_influence(text)
                }
                new_rows.append(enriched)
                logger.info(f"Processed article: {article_id}")
            except Exception as e:
                logging.error(f"Error processing article {row.get('article_id')}: {e}")

        # Save progress
        pd.DataFrame(new_rows).to_csv(checkpoint_csv, mode='a', index=False, header=not os.path.exists(checkpoint_csv))
        new_rows = []

    print("✅ LLM enrichment complete.")
