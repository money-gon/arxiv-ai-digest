import feedparser
import json
import os
import re
import requests
import time
from datetime import datetime
from typing import List, Tuple

# ==========================
# arXiv設定
# ==========================
# 3種類のタグに対応する arXiv 検索クエリ（各タグから論文を取得してマージ）
ARXIV_BASE = "http://export.arxiv.org/api/query?"
ARXIV_QUERIES = [
    # AIエージェント: cs.AI + agent 関連
    "search_query=cat:cs.AI+AND+all:agent&sortBy=submittedDate&max_results=2",
    # Robotics: ロボティクス
    "search_query=cat:cs.RO&sortBy=submittedDate&max_results=2",
    # ハンド模倣学習: 手・模倣・デクスタース操作
    "search_query=(cat:cs.RO+OR+cat:cs.LG)+AND+(all:hand+OR+all:imitation+OR+all:dexterous+OR+all:manipulation)&sortBy=submittedDate&max_results=2",
]

MAX_PAPERS_PER_RUN = 4  # マージ後、日付でソートしてこの件数まで
DB_FILE = "papers_db.json"

# ==========================
# Hugging Face設定
# ==========================
HF_TOKEN = os.getenv("HF_TOKEN")
# Hugging Face Router (旧 api-inference は 410 で廃止)
HF_CHAT_URL = "https://router.huggingface.co/v1/chat/completions"
HF_MODEL = "Qwen/Qwen2.5-7B-Instruct"
# HF_MODEL = "mistralai/Mistral-7B-Instruct-v0.2:fastest"

ALLOWED_TAGS = {"AIエージェント", "Robotics", "ハンド模倣学習"}


# ==========================
# DB処理
# ==========================
def load_db() -> List[dict]:
    if not os.path.exists(DB_FILE):
        return []
    try:
        with open(DB_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError:
        print("WARNING: DB JSON broken. Starting fresh.")
        return []


def save_db(data: List[dict]):
    with open(DB_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


# ==========================
# 要約処理
# ==========================
def clean_abstract(text: str) -> str:
    return " ".join(text.replace("\n", " ").split())


def summarize_to_japanese(text: str) -> str:
    headers = {
        "Authorization": f"Bearer {HF_TOKEN}",
        "Content-Type": "application/json",
    }

    prompt = f"""あなたは研究者向けの「わかりやすい要約」を書く担当です。

【ルール】
・専門用語にはカッコで簡単な補足を入れる（例: 剪定（不要な計算を省く手法））。
・1文は短めにし、箇条書きや「〜である。」を適度に使う。
・次の4項目を必ずすべて書き、見出しは【】で囲む。最後の【ポイント】まで必ず書き切ること。途中で切らさないこと。
・各項目は簡潔に（背景・提案・結果は各2〜3文、ポイントは1〜2文）。
・要約の最後に1行だけ「タグ: 〇〇, 〇〇」または「タグ: なし」を書く。タグは次の3つから該当するものだけカンマ区切りで: AIエージェント, Robotics, ハンド模倣学習。該当がなければ「タグ: なし」。
・タグ行は必ず最後に1行だけ書く。
・タグ以外の場所に「タグ」という単語を書かない。

【背景】 何が問題で、なぜ重要か
【提案】 この論文で何をしたか
【結果】 実験や評価で何がわかったか
【ポイント】 一言で言うと、どんな貢献か

論文アブストラクト:
{text}
"""

    payload = {
        "model": HF_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 768,
        "temperature": 0.2,
    }

    for attempt in range(3):  # retry 3 times
        try:
            response = requests.post(
                HF_CHAT_URL,
                headers=headers,
                json=payload,
                timeout=60,
            )

            print("HF STATUS:", response.status_code)

            if response.status_code == 429:
                print("Rate limit. Retrying...")
                time.sleep(5)
                continue

            if response.status_code != 200:
                print("HF ERROR:", response.status_code, response.text[:300])
                return "日本語要約に失敗しました。"

            data = response.json()
            if "choices" not in data or not data["choices"]:
                print("HF EMPTY RESPONSE:", data)
                return "日本語要約に失敗しました。"

            content = data["choices"][0]["message"]["content"].strip()
            if not content:
                print("HF BLANK CONTENT")
                return "日本語要約に失敗しました。"

            return content

        except requests.RequestException as e:
            print("HF request error:", e)
            time.sleep(5)

    return "日本語要約に失敗しました。"


# ==========================
# タグ抽出
# ==========================
def extract_tags_from_summary(summary_ja: str) -> Tuple[str, List[str]]:
    """
    要約全文から「タグ: ...」行をすべて検出し、
    ALLOWED_TAGS に一致するものだけ抽出。
    その行は本文から削除する。
    """

    if not summary_ja:
        return "", []

    tags = []
    text = summary_ja

    # タグ行を正規表現で検出（太字対応）
    pattern = r"\*?\*?タグ[:：]\s*(.+)"

    matches = re.findall(pattern, text)

    for match in matches:
        tag_part = match.strip()
        if tag_part != "なし":
            for t in tag_part.split(","):
                t = t.strip()
                if t in ALLOWED_TAGS and t not in tags:
                    tags.append(t)

    # タグ行をすべて削除
    text = re.sub(r"\*?\*?タグ[:：].*", "", text)

    clean_summary = text.strip()
    return clean_summary, tags


# ==========================
# arXiv取得
# ==========================
def fetch_latest_papers() -> List[dict]:
    seen_ids = set()
    papers = []
    for q in ARXIV_QUERIES:
        url = ARXIV_BASE + q
        feed = feedparser.parse(url)
        for entry in feed.entries:
            if entry.id in seen_ids:
                continue
            seen_ids.add(entry.id)
            papers.append({
                "id": entry.id,
                "title": entry.title.strip(),
                "summary_en": clean_abstract(entry.summary),
                "published": entry.published,
                "link": entry.link,
            })

    # 公開日で新しい順にソートし、最大件数に制限
    papers.sort(key=lambda p: p["published"] or "", reverse=True)
    return papers[:MAX_PAPERS_PER_RUN]


# ==========================
# メイン処理
# ==========================
def main():
    if not HF_TOKEN:
        print("HF_TOKEN not set")
        return

    db = load_db()
    db_map = {paper["id"]: paper for paper in db}

    papers = fetch_latest_papers()
    new_count = 0

    for paper in papers:
        if paper["id"] in db_map:
            continue

        print(f"Summarizing: {paper['title']}")
        summary_ja = summarize_to_japanese(paper["summary_en"])
        summary_clean, tags = extract_tags_from_summary(summary_ja)

        db_map[paper["id"]] = {
            **paper,
            "summary_ja": summary_clean,
            "tags": tags,
            "saved": False,
            "created_at": datetime.now().strftime("%Y-%m-%d"),
        }

        new_count += 1

    if new_count > 0:
        updated_db = sorted(
            db_map.values(),
            key=lambda p: p["published"] or "",
            reverse=True
        )
        save_db(updated_db)
        print(f"Added {new_count} new papers.")
    else:
        print("No new papers.")


if __name__ == "__main__":
    main()
