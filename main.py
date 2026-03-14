import feedparser
import json
import os
import re
import requests
import time
from datetime import date
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

MAX_PAPERS_PER_RUN = 4
DB_FILE = "papers_db.json"

# ==========================
# Hugging Face設定
# ==========================
HF_TOKEN      = os.getenv("HF_TOKEN")
HF_CHAT_URL   = "https://router.huggingface.co/v1/chat/completions"
HF_MODEL      = "Qwen/Qwen2.5-7B-Instruct"

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

# --- 改善点①: システムプロンプトとユーザープロンプトの重複を除去
#              各セクション間に空行を明示
#              max_tokens を 768 → 1024 に増加（【ポイント】の途中打ち切り防止）
#              タグ判定基準を具体化（誤タグ付け抑制）
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
        "max_tokens": 1024,   # 768 → 1024: 【ポイント】打ち切り防止
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
    """
    要約末尾の「タグ: ...」行から ALLOWED_TAGS を抽出し、行を除去して返す。
    """
    tags: List[str] = []
    for m in re.findall(r"タグ[:：](.+)", summary):
        for t in m.split(","):
            t = t.strip()
            if t in ALLOWED_TAGS and t not in tags:
                tags.append(t)

    summary = re.sub(r"タグ[:：].*", "", summary).strip()
    return summary, tags


# --- 改善点②: infer_tags のフォールバックを厳格化
#              タイトル＋アブストラクト両方にキーワードが存在する場合のみタグ付け
#              天文・物理・化学など明らかに無関係なカテゴリを除外
UNRELATED_DOMAINS = {
    "astro", "astrophys", "stellar", "supernovae", "supernova", "galaxy",
    "quantum", "fluid", "thermodynamic", "plasma", "optic", "photon",
    "chemical", "molecular", "protein", "genomic",
}

def infer_tags(title: str, abstract: str) -> List[str]:
    """
    LLM タグ生成失敗時のフォールバック。
    タイトルと要約の両方にキーワードが存在し、
    かつ無関係ドメインのキーワードが含まれない場合のみタグ付けする。
    """
    combined = (title + " " + abstract).lower()

    # 無関係ドメインのキーワードが含まれていればスキップ
    if any(d in combined for d in UNRELATED_DOMAINS):
        return []

    tags: List[str] = []
    title_l    = title.lower()
    abstract_l = abstract.lower()

    def both(kw: str) -> bool:
        """タイトルまたは要約のどちらにも（あるいは両方に）含まれるか"""
        return kw in title_l or kw in abstract_l

    # AIエージェント: agent 単独ではなく、autonomous/planning/LLM などと共起
    agent_keywords = ["autonomous agent", "llm agent", "ai agent", "tool use", "planning agent", "multi-agent"]
    if any(kw in combined for kw in agent_keywords):
        tags.append("AIエージェント")

    # Robotics: robot が title か abstract にあること
    if both("robot"):
        tags.append("Robotics")

    # ハンド模倣学習: 複数キーワードの共起を要求
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
# ==========================
def fetch() -> List[dict]:
    seen: set[str] = set()
    papers: List[dict] = []

    for q in ARXIV_QUERIES:
        feed = feedparser.parse(ARXIV_BASE + q)
        time.sleep(1)

        for e in feed.entries:
            if e.id in seen:
                continue
            seen.add(e.id)
            pid = e.id.split("/")[-1]
            papers.append({
                "id":         pid,
                "title":      e.title.strip(),
                "summary_en": clean_abstract(e.summary),
                "published":  e.published,
                "link":       e.link,
            })

    papers.sort(key=lambda x: x["published"], reverse=True)
    return papers[:MAX_PAPERS_PER_RUN]

# ==========================
# メイン処理
# ==========================
def main():
    if not HF_TOKEN:
        print("HF_TOKEN not set")
        return

    db     = load_db()
    db_map = {p["id"]: p for p in db}
    papers = fetch()
    new_count = 0

    for p in papers:
        if p["id"] in db_map:
            continue

        print("summarizing:", p["title"])
        summary_raw = summarize_to_japanese(p["summary_en"])

        if summary_raw == "要約失敗":
            summary_clean = "要約生成に失敗しました。"
            # --- 改善点③: フォールバック時もタイトルを活用
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
            "created_at": date.today().isoformat(),  # 改善点④: 追加日を記録
        }
        new_count += 1

    if new_count > 0:
        updated = sorted(db_map.values(), key=lambda x: x["published"], reverse=True)
        save_db(updated)
        print(f"added {new_count} new papers.")
    else:
        print("no new papers added.")

if __name__ == "__main__":
    main()
