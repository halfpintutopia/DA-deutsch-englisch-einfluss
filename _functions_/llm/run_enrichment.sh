#!/bin/bash

# Run LLM Enrichment Pipeline in background using nohup
# Saves output to enrich_log.txt

nohup python llm_enrichment_pipeline.py \
  > enrich_log.txt 2>&1 &

echo "🚀 Enrichment script started in background. Logging to enrich_log.txt"
