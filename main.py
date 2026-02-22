import feedparser
import json
import os
import requests
from datetime import datetime

ARXIV_URL = (
    "http://export.arxiv.org/api/query?"
    "search_query=cat:cs.AI+OR+cat:cs.RO+OR+cat:cs.LG&"
    "sortBy=submittedDate&max_results=5"
)

DB_FILE = "papers_db.json"

HF_TOKEN = os.getenv("HF_TOKEN")
# Hugging Face Router (旧 api-inference は 410 で廃止)
HF_CHAT_URL = "https://router.huggingface.co/v1/chat/completions"
HF_MODEL = "mistralai/Mistral-7B-Instruct-v0.2:fastest"

def load_db():
    if not os.path.exists(DB_FILE):
        return []
    with open(DB_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def save_db(data):
    with open(DB_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def summarize_to_japanese(text):
    headers = {
        "Authorization": f"Bearer {HF_TOKEN}",
        "Content-Type": "application/json",
    }
    prompt = f"""以下の論文を日本語で要約してください。研究背景・提案手法・実験結果・実用的意義を8〜12行で説明してください。

{text}
"""
    payload = {
        "model": HF_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 512,
    }

    response = requests.post(
        HF_CHAT_URL,
        headers=headers,
        json=payload,
        timeout=90,
    )

    if response.status_code != 200:
        print("ERROR:", response.status_code, response.text[:300] if response.text else "")
        return "日本語要約に失敗しました。原文を参照してください。"

    try:
        data = response.json()
        return data["choices"][0]["message"]["content"].strip()
    except (KeyError, IndexError, TypeError):
        return "日本語要約に失敗しました。原文を参照してください。"


def fetch_latest_papers():
    feed = feedparser.parse(ARXIV_URL)
    papers = []

    for entry in feed.entries:
        papers.append({
            "id": entry.id,
            "title": entry.title,
            "summary_en": entry.summary,
            "published": entry.published,
            "link": entry.link
        })

    return papers


def main():
    if not HF_TOKEN:
        print("HF_TOKEN not set")
        return

    db = load_db()
    existing_ids = {paper["id"] for paper in db}

    papers = fetch_latest_papers()
    new_entries = []

    for paper in papers:
        if paper["id"] in existing_ids:
            continue

        print(f"Summarizing: {paper['title']}")
        summary_ja = summarize_to_japanese(paper["summary_en"])

        entry = {
            **paper,
            "summary_ja": summary_ja,
            "saved": False,
            "created_at": datetime.now().strftime("%Y-%m-%d")
        }

        new_entries.append(entry)

    if new_entries:
        db.extend(new_entries)
        save_db(db)
        print(f"Added {len(new_entries)} new papers to DB.")
    else:
        print("No new papers.")


if __name__ == "__main__":
    main()
