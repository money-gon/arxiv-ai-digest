import feedparser
import json
import os
import re
import requests
import time
from datetime import datetime, timezone, timedelta
from typing import List, Tuple

# ==========================
# arXiv設定
# ==========================
ARXIV_BASE = "http://export.arxiv.org/api/query?"
ARXIV_QUERIES = [
    # AIエージェント: cs.AI × "agent"
    "search_query=cat:cs.AI+AND+all:agent&sortBy=submittedDate&max_results=3",
    # Robotics: cs.RO
    "search_query=cat:cs.RO&sortBy=submittedDate&max_results=3",
    # ハンド模倣学習: cs.RO / cs.LG × 手・模倣・巧み・操作系キーワード
    "search_query=(cat:cs.RO+OR+cat:cs.LG)+AND+(all:hand+OR+all:imitation+OR+all:dexterous+OR+all:manipulation)&sortBy=submittedDate&max_results=3",
]

MAX_PAPERS_PER_RUN = 6
DB_FILE = "papers_db.json"

# ==========================
# Hugging Face設定
# ==========================
HF_TOKEN    = os.getenv("HF_TOKEN")
HF_CHAT_URL = "https://router.huggingface.co/v1/chat/completions"
HF_MODEL    = "Qwen/Qwen2.5-7B-Instruct"

ALLOWED_TAGS = {"AIエージェント", "Robotics", "ハンド模倣学習"}

# JST タイムゾーン（不具合②: UTC date を使うと JST と1日ずれる）
JST = timezone(timedelta(hours=9))

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

SYSTEM_PROMPT = (
    "あなたは研究論文を日本語で要約する専門家です。\n"
    "回答は必ず日本語のみで書いてください。\n"
    "【背景】【提案】【結果】【ポイント】の4セクションをこの順番で必ず書き、"
    "最後の行にタグ行を1行だけ書いてください。"
)

def build_user_prompt(abstract: str) -> str:
    return f"""以下の論文アブストラクトを、下記のフォーマットで要約してください。

=== フォーマット ===
【背景】
（2〜3文。何が問題で、なぜ重要か）

【提案】
（2〜3文。この論文で何をしたか）

【結果】
（2〜3文。実験や評価で何がわかったか）

【ポイント】
（1〜2文。一言でどんな貢献があるか）

タグ: （該当するものをカンマ区切りで。なければ「なし」）

=== タグの定義 ===
・AIエージェント  … LLM/VLMを使った自律エージェント、ツール呼び出し、プランニング、RAGなど
・Robotics        … 実機ロボットの制御・設計・シミュレーション（天文・流体など無関係分野は除く）
・ハンド模倣学習  … ロボットハンドの巧み操作、模倣学習、デモンストレーションからの学習
※ キーワードが偶然含まれていても、論文の主題でなければタグを付けないでください。

=== 執筆ルール ===
・日本語のみ（英語・中国語など混入禁止）
・専門用語には括弧で補足: 例「強化学習（RL）」
・各セクションは簡潔に（1セクション2〜3文）
・タグ行は最後に1行だけ書く（本文中に「タグ」という語を使わない）
・【ポイント】まで必ず書き切ること。途中で文を切らないこと。

=== アブストラクト ===
{abstract}
"""

def summarize_to_japanese(text: str) -> str:
    headers = {
        "Authorization": f"Bearer {HF_TOKEN}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": HF_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": build_user_prompt(text)},
        ],
        "max_tokens": 1024,
        "temperature": 0.2,
        "top_p": 0.95,
    }

    for attempt in range(3):
        try:
            response = requests.post(HF_CHAT_URL, headers=headers, json=payload, timeout=60)
            print(f"HF STATUS: {response.status_code} (attempt {attempt + 1})")

            if response.status_code != 200:
                print("HF ERROR:", response.status_code, response.text[:300])
                time.sleep(5)
                continue

            content = response.json()["choices"][0]["message"]["content"].strip()
            if not content:
                print("HF BLANK CONTENT")
                return "要約失敗"
            return content

        except requests.RequestException as e:
            print(f"HF request error (attempt {attempt + 1}):", e)
            time.sleep(5)

    return "要約失敗"

# ==========================
# タグ抽出
# ==========================
def extract_tags(summary: str) -> Tuple[str, List[str]]:
    tags: List[str] = []
    for m in re.findall(r"タグ[:：](.+)", summary):
        for t in m.split(","):
            t = t.strip()
            if t in ALLOWED_TAGS and t not in tags:
                tags.append(t)
    summary = re.sub(r"タグ[:：].*", "", summary).strip()
    return summary, tags

UNRELATED_DOMAINS = {
    "astro", "astrophys", "stellar", "supernovae", "supernova", "galaxy",
    "quantum", "fluid", "thermodynamic", "plasma", "optic", "photon",
    "chemical", "molecular", "protein", "genomic",
}

def infer_tags(title: str, abstract: str) -> List[str]:
    """LLM タグ生成失敗時のフォールバック。厳格な共起チェックで誤タグを防ぐ。"""
    combined = (title + " " + abstract).lower()

    if any(d in combined for d in UNRELATED_DOMAINS):
        return []

    tags: List[str] = []

    agent_keywords = ["autonomous agent", "llm agent", "ai agent", "tool use",
                      "planning agent", "multi-agent", "language model agent"]
    if any(kw in combined for kw in agent_keywords):
        tags.append("AIエージェント")

    if "robot" in title.lower() or "robot" in abstract.lower():
        tags.append("Robotics")

    hand_score = sum([
        "dexterous" in combined,
        "manipulation" in combined and "robot" in combined,
        "imitation learning" in combined,
        "hand" in combined and ("robot" in combined or "grasp" in combined),
        "teleoperation" in combined,
    ])
    if hand_score >= 2:
        tags.append("ハンド模倣学習")

    return tags

# ==========================
# arXiv取得
# 不具合⑤: e.id（フルURL）をそのまま ID として使用し DB と一致させる
# ==========================
def fetch() -> List[dict]:
    seen: set = set()
    papers: List[dict] = []

    for q in ARXIV_QUERIES:
        feed = feedparser.parse(ARXIV_BASE + q)
        time.sleep(1)

        for e in feed.entries:
            # e.id = "http://arxiv.org/abs/2603.04392v1" (フルURL)
            if e.id in seen:
                continue
            seen.add(e.id)

            papers.append({
                "id":         e.id,           # ← フル URL を canonical ID として使用
                "title":      e.title.strip(),
                "summary_en": clean_abstract(e.summary),
                "published":  e.published,
                "link":       e.link,
            })

    papers.sort(key=lambda x: x["published"], reverse=True)
    return papers[:MAX_PAPERS_PER_RUN]

# ==========================
# DB クリーンアップ
# ==========================

# 保持期間（日数）。この日数を超えた論文のうち saved でないものを削除する
RETENTION_DAYS = 30

SAVED_FILE = "saved.json"

def load_saved_ids() -> set:
    """
    リポジトリ内の saved.json から保存済み論文 ID を読み込む。
    ファイルが存在しない・壊れている場合は空セットを返す（安全側に倒す）。
    ID は正規化して比較するため、短形式・フルURL どちらでも一致させる。
    """
    if not os.path.exists(SAVED_FILE):
        return set()
    try:
        with open(SAVED_FILE, "r", encoding="utf-8") as f:
            raw = json.load(f)
        if not isinstance(raw, list):
            return set()
        return set(normalize_id(i) for i in raw if i)
    except (json.JSONDecodeError, TypeError):
        print("WARNING: saved.json broken. Treating all as unsaved.")
        return set()

def normalize_id(paper_id: str) -> str:
    """ID を 'http://arxiv.org/abs/XXXX' 形式に統一する。"""
    if not paper_id:
        return paper_id
    if paper_id.startswith("http://") or paper_id.startswith("https://"):
        return paper_id
    if paper_id.startswith("abs/"):
        return "http://arxiv.org/" + paper_id
    return "http://arxiv.org/abs/" + paper_id

def cleanup_db(db: list, saved_ids: set, retention_days: int, today: "date") -> tuple:
    """
    保持期間を超えた論文のうち、saved でないものを DB から削除する。

    削除条件（AND）:
      - published が retention_days 日以上前
      - saved_ids に含まれていない

    Returns:
        (cleaned_db, removed_count)
    """
    from datetime import date as date_type
    cutoff = today.toordinal() - retention_days  # ordinal で比較（タイムゾーン非依存）

    kept    = []
    removed = 0

    for p in db:
        pid = normalize_id(p.get("id", ""))

        # Saved 論文は期間に関係なく保持
        if pid in saved_ids:
            kept.append(p)
            continue

        # published を日付に変換して比較
        published_str = p.get("published", "")
        try:
            # arXiv の published 形式: "2026-03-13T12:34:56Z"
            pub_date = datetime.fromisoformat(
                published_str.replace("Z", "+00:00")
            ).date()
            if pub_date.toordinal() >= cutoff:
                kept.append(p)   # 保持期間内 → 保持
            else:
                removed += 1
                print(f"  remove (expired): {p.get('title', pid)[:60]}")
        except (ValueError, AttributeError):
            # 日付パースに失敗した論文は安全のため保持
            kept.append(p)

    return kept, removed


# ==========================
# メイン処理
# ==========================
def main():
    if not HF_TOKEN:
        print("HF_TOKEN not set")
        return

    db        = load_db()
    db_map    = {p["id"]: p for p in db}
    papers    = fetch()
    new_count = 0

    today_jst = datetime.now(JST).date()

    # ── 新規論文の追加 ────────────────────────────────────────────
    for p in papers:
        if p["id"] in db_map:
            continue

        print("summarizing:", p["title"])
        summary_raw = summarize_to_japanese(p["summary_en"])

        if summary_raw == "要約失敗":
            summary_clean = "要約生成に失敗しました。"
            tags = infer_tags(p["title"], p["summary_en"])
        else:
            summary_clean, tags = extract_tags(summary_raw)
            if not tags:
                tags = infer_tags(p["title"], p["summary_en"])

        db_map[p["id"]] = {
            "id":         p["id"],
            "title":      p["title"],
            "summary_ja": summary_clean,
            "tags":       tags,
            "link":       p["link"],
            "published":  p["published"],
            "created_at": today_jst.isoformat(),
        }
        new_count += 1

    # ── DB クリーンアップ ─────────────────────────────────────────
    saved_ids  = load_saved_ids()
    print(f"saved papers: {len(saved_ids)}")

    current_db = list(db_map.values())
    cleaned_db, removed_count = cleanup_db(current_db, saved_ids, RETENTION_DAYS, today_jst)

    print(f"cleanup: {removed_count} paper(s) removed "
          f"(older than {RETENTION_DAYS} days, not saved).")

    # ── 保存（追加 or 削除があった場合のみ書き込み）─────────────────
    if new_count > 0 or removed_count > 0:
        updated = sorted(cleaned_db, key=lambda x: x["published"], reverse=True)
        save_db(updated)
        print(f"DB updated: +{new_count} added, -{removed_count} removed. "
              f"total={len(updated)}")
    else:
        print("no changes to DB.")

if __name__ == "__main__":
    main()
