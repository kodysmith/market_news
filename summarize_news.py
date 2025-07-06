import requests
import subprocess
import json
import os
import time
from dotenv import load_dotenv

# --- CONFIG ---
# Always load .env from project root
load_dotenv(dotenv_path=".env")
FMP_API_KEY = os.environ.get("FMP_API_KEY")
if not FMP_API_KEY:
    raise RuntimeError("FMP_API_KEY not found in .env file. Please add it as FMP_API_KEY=your_key_here")
NEWS_LIMIT = 10  # Number of news articles to fetch
NEWS_URL = f"https://financialmodelingprep.com/api/v4/general_news?page=0&limit={NEWS_LIMIT}&apikey={FMP_API_KEY}"
NEWS_JSON_PATH = "news.json"
GEMINI_CLI = "gemini"  # Change if your CLI is named differently

# --- Fetch news from FMP ---
def fetch_news():
    resp = requests.get(NEWS_URL)
    if resp.status_code == 200:
        return resp.json()
    print(f"Failed to fetch news: {resp.status_code}")
    return []

# --- Summarize with Gemini CLI ---
def summarize_with_gemini(text):
    try:
        result = subprocess.run(
            [GEMINI_CLI, "summarize", "--text", text],
            capture_output=True, text=True, timeout=30
        )
        if result.returncode != 0 or not result.stdout.strip():
            print("Gemini error:", result.stderr)
        return result.stdout.strip() if result.returncode == 0 else ""
    except Exception as e:
        print(f"Gemini error: {e}")
        return ""

# --- Main workflow ---
def main():
    news_items = fetch_news()
    summarized = []
    for item in news_items:
        headline = item.get("title") or item.get("headline")
        url = item.get("url")
        source = item.get("site") or item.get("source")
        raw_text = item.get("text") or item.get("summary") or headline
        print(f"Summarizing: {headline}")
        summary = summarize_with_gemini(raw_text)
        summarized.append({
            "headline": headline,
            "source": source,
            "url": url,
            "summary": summary
        })
        time.sleep(1)  # Be nice to the API/CLI
    with open(NEWS_JSON_PATH, "w") as f:
        json.dump(summarized, f, indent=2)
    print(f"Wrote {len(summarized)} news items to {NEWS_JSON_PATH}")

if __name__ == "__main__":
    main() 
