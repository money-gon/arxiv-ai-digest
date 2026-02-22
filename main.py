import feedparser
import json
import os
import requests
from datetime import datetime

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
MAX_PAPERS_PER_RUN = 6  # マージ後、日付でソートしてこの件数まで

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
    prompt = f"""あなたは研究者向けの「わかりやすい要約」を書く担当です。

【ルール】
・専門用語にはカッコで簡単な補足を入れる（例: 剪定（不要な計算を省く手法））。
・1文は短めにし、箇条書きや「〜である。」を適度に使う。
・次の4項目を必ずすべて書き、見出しは【】で囲む。最後の【ポイント】まで必ず書き切ること。途中で切らさないこと。
・各項目は簡潔に（背景・提案・結果は各2〜3文、ポイントは1〜2文）。
・要約の最後に1行だけ「タグ: 〇〇, 〇〇」または「タグ: なし」を書く。タグは次の3つから該当するものだけカンマ区切りで: AIエージェント, Robotics, ハンド模倣学習。該当がなければ「タグ: なし」。

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
        "max_tokens": 1024,
    }

    response = requests.post(
        HF_CHAT_URL,
        headers=headers,
        json=payload,
        timeout=120,
    )

    if response.status_code != 200:
        print("ERROR:", response.status_code, response.text[:300] if response.text else "")
        return "日本語要約に失敗しました。原文を参照してください。"

    try:
        data = response.json()
        return data["choices"][0]["message"]["content"].strip()
    except (KeyError, IndexError, TypeError):
        return "日本語要約に失敗しました。原文を参照してください。"


ALLOWED_TAGS = {"AIエージェント", "Robotics", "ハンド模倣学習"}


def extract_tags_from_summary(summary_ja):
    """要約末尾の「タグ: 〇〇, 〇〇」行を解析し、タグリストとタグ行を除いた要約を返す。"""
    tags = []
    text = summary_ja or ""
    lines = text.strip().split("\n")
    while lines:
        last = lines[-1].strip()
        if last.startswith("タグ:") or last.startswith("タグ："):
            tag_part = last.replace("タグ:", "").replace("タグ：", "").strip()
            if tag_part != "なし":
                for t in tag_part.split(","):
                    t = t.strip()
                    if t in ALLOWED_TAGS:
                        tags.append(t)
            lines.pop()
            break
        break
    clean_summary = "\n".join(lines).strip()
    return clean_summary, tags


def fetch_latest_papers():
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
                "title": entry.title,
                "summary_en": entry.summary,
                "published": entry.published,
                "link": entry.link,
            })
    # 公開日で新しい順にソートし、最大件数に制限
    papers.sort(key=lambda p: p["published"] or "", reverse=True)
    return papers[:MAX_PAPERS_PER_RUN]


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
        summary_clean, tags = extract_tags_from_summary(summary_ja)

        entry = {
            **paper,
            "summary_ja": summary_clean,
            "tags": tags,
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
