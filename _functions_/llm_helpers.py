#!/usr/bin/env python
# coding: utf-8

import ollama
import pandas as pd 
from typing import Optional
from tqdm import tqdm
import time

def ask_ollama(
        prompt: str, 
        model: str = "mistral",
        system: Optional[str] = None,
        retries: int = 3,
        delay: float = 2.0
    ) -> str:
    """Send a prompt to the specified Ollama language model and return the model's response."""
    messages = []
    
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    for attempt in range(retries):
        try:
            response = ollama.chat(model=model, messages=messages)
            return response["message"]["content"].strip()
        except Exception as e:
            print(f"[Ollama error - attempt {attempt+1}] {e}")
            time.sleep(delay)
        return "error"

    response = ollama.chat(model=model, messages=[
        {"role": "system", "content": system} if system else {},
        {"role": "user", "content": prompt}
    ])
    return response["message"]["content"]

def classify_tone(text: str) -> str:
    """Classify the tone of a given text as either 'formal' or 'informal'."""
    prompt = (
        "Is the following article written in a formal or informal tone? "
        "Respond with only one word: 'formal' or 'informal'.\n\n"
        f"Text:\n{text}"
    )
    return ask_ollama(prompt=prompt).strip().lower()

def classify_topic(text: str) -> str:
    """Classify the topic of a given text into one of the predefined categories: 'business', 'technology', 'lifestyle', 'politics', or 'culture'."""
    prompt = (
        "Classify the following article as one of the following categories: "
        "'business', 'technology', 'lifestyle', 'politics', 'culture'. "
        "Respond with only one word.\n\n"
        f"Text:\n{text}"
    )

    return ask_ollama(prompt=prompt).strip().lower()

def summarise_article(text: str) -> str:
    """Generate a 2–3 sentence summary of a German article."""
    prompt = (
        "Summarise the German article in 2-3 sentences:\n\n"
        f"{text}"
    )
    
    return ask_ollama(prompt=prompt)

def explain_loanwords_usage(text: str) -> str:
    """Explain the use of English loanwords in a German article and infer possible context or target audience implications."""
    prompt = (
        "Why does this German article use English words? "
        "What might this say about the context or target audience?\n\n"
        f"{text}"
    )

    return ask_ollama(prompt=prompt)

def detect_marketing_loanwords(text: str) -> str:
    """Identify English loanwords in a German article that are used in a marketing or advertising context."""
    prompt = (
        "Here is an article in German:\n\n"
        f"{text}"
        "Which of these English loanwords are used in a marketing or advertising context? "
        "Return a list."
    )

    return ask_ollama(prompt=prompt)

def detect_country_influence(text: str) -> str:
    """Detect and return a single influential country (or 'Multiple'/'Other') whose culture is reflected in the German article."""
    prompt = (
        "Does this German article show cultural influence from any of the following influential countries: "
        "USA, China, Russia, India, France, Germany, UK, Japan, Saudi Arabia, Italy, Canada, Israel, Australia, Spain, South Korea, Turkey, Switzerland, Iran. "
        "If there is influence from more than one, respond with all relevant countries. "
        "If the influence comes from a country not on this list, name the specific country or countries explicitly. "
        "Provide your response in JSON format with two fields:\n"
        "1. \"countries\": a list of influenced countries\n"
        "2. \"reason\": a short explanation of the cultural influence observed\n\n"
        f"{text}"
    )
    
    return ask_ollama(prompt=prompt)


