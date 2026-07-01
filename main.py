import feedparser
import json
import os
import re
import requests
import time
from datetime import datetime, timezone, timedelta, date
from typing import List, Tuple, Optional

# ==========================
# arXiv設定
# ==========================
ARXIV_BASE = "http://export.arxiv.org/api/query?"
ARXIV_QUERIES = [
    # AIエージェント: cs.AI × "agent"
    "search_query=cat:cs.AI+AND+all:agent&sortBy=submittedDate&max_results=10",
    # Robotics: cs.RO
    "search_query=cat:cs.RO&sortBy=submittedDate&max_results=10",
    # ハンド模倣学習: cs.RO / cs.LG × 手・模倣・巧み・操作系
    "search_query=(cat:cs.RO+OR+cat:cs.LG)+AND+(all:hand+OR+all:imitation+OR+all:dexterous+OR+all:manipulation)&sortBy=submittedDate&max_results=10",
]
MAX_SUMMARIZE_PER_RUN = 6  # 1回のワークフローで要約する最大件数（API コスト調整用）
DB_FILE = "papers_db.json"

# ==========================
# LLM プロバイダー設定
# ==========================
# ── プロバイダー選択 ────────────────────────────────────────────
# 環境変数 LLM_PROVIDER で切り替え（デフォルト: groq）
#
# groq  : 無料枠 14,400 req/日・30 req/min。日本語品質◎。カード登録不要。
#         GROQ_API_KEY を GitHub Secrets に登録する。
#         https://console.groq.com/keys
#
# hf    : Hugging Face Router。無料枠が少なく 429 が頻発するため非推奨。
#         HF_TOKEN を GitHub Secrets に登録する。
# ─────────────────────────────────────────────────────────────────
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq")

# Groq 設定
GROQ_API_KEY  = os.getenv("GROQ_API_KEY")
GROQ_CHAT_URL = "https://api.groq.com/openai/v1/chat/completions"
# llama-3.3-70B: Groq 無料枠内で最高の日本語品質
# 他の選択肢: llama-3.1-8b-instant（高速・軽量）、gemma2-9b-it（日本語もそこそこ）
# GROQ_MODEL    = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")  # 2026/08で終了
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")


# HF 設定（フォールバック用）
HF_TOKEN    = os.getenv("HF_TOKEN")
HF_CHAT_URL = "https://router.huggingface.co/v1/chat/completions"
HF_MODEL    = os.getenv("HF_MODEL", "Qwen/Qwen3-8B")

# 実行時に使うプロバイダー情報をまとめる
def _get_provider() -> tuple:
    """(api_key, url, model_name) を返す。キー未設定なら RuntimeError。"""
    if LLM_PROVIDER == "groq":
        if not GROQ_API_KEY:
            raise RuntimeError("GROQ_API_KEY が設定されていません。GitHub Secrets を確認してください。")
        return GROQ_API_KEY, GROQ_CHAT_URL, GROQ_MODEL
    else:
        if not HF_TOKEN:
            raise RuntimeError("HF_TOKEN が設定されていません。")
        return HF_TOKEN, HF_CHAT_URL, HF_MODEL

ALLOWED_TAGS = {"AIエージェント", "Robotics", "ハンド模倣学習"}

# JST タイムゾーン
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
# Semantic Scholar による論文情報の補強
# ==========================
# arXiv API は Abstract のみ返すが、S2 は TLDR・引用数なども提供する。
# 論文の ID 部分（例: "2503.04392v1" → "2503.04392"）を使って検索する。
# - 成功時: Abstract + TLDR（あれば）を結合してより豊かなコンテキストを得る
# - 失敗時: arXiv API の Abstract のみで続行（フォールバック）
# - Rate limit: 100 req/5min (未認証)。論文ごとに 1 回のみ呼ぶ。

S2_API_BASE = "https://api.semanticscholar.org/graph/v1/paper"
S2_FIELDS   = "abstract,tldr,citationCount,year"
S2_HEADERS  = {"User-Agent": "ArXivDigestBot/1.0 (academic summarizer; github actions)"}

def _arxiv_pid(full_id: str) -> str:
    """'http://arxiv.org/abs/2503.04392v1' → '2503.04392' (バージョン除去)"""
    pid = full_id.split("/")[-1]       # "2503.04392v1"
    return re.sub(r"v\d+$", "", pid)   # "2503.04392"

def fetch_s2_info(full_id: str) -> dict:
    """
    Semantic Scholar API から追加情報を取得する。
    Returns dict with keys: abstract, tldr, citation_count (all optional)
    失敗時は空 dict を返す。
    """
    pid = _arxiv_pid(full_id)
    url = f"{S2_API_BASE}/arXiv:{pid}"
    try:
        r = requests.get(
            url,
            params={"fields": S2_FIELDS},
            headers=S2_HEADERS,
            timeout=15,
        )
        if r.status_code == 200:
            d = r.json()
            return {
                "abstract":       d.get("abstract") or "",
                "tldr":           (d.get("tldr") or {}).get("text") or "",
                "citation_count": d.get("citationCount") or 0,
            }
        # 404: S2 にまだ登録されていない新しい論文 (正常系)
        if r.status_code == 429:
            print(f"  S2 rate limited for {pid}. Skipping S2 enrichment.")
        elif r.status_code != 404:
            print(f"  S2 API {r.status_code} for {pid}: {r.text[:100]}")
    except requests.RequestException as e:
        print(f"  S2 request error for {pid}: {e}")
    return {}

def build_context(summary_en: str, s2: dict) -> str:
    """
    arXiv Abstract と S2 情報を組み合わせて LLM への入力コンテキストを構築する。
    S2 の Abstract が arXiv と異なる（より詳細な）場合は両方を使う。
    TLDR がある場合は最初のヒントとして付加する。
    """
    parts: List[str] = []

    # S2 TLDR（機械生成の英語要約）: LLM への先行ヒントとして利用
    if s2.get("tldr"):
        parts.append(f"[Key point (auto-generated): {s2['tldr']}]")

    # Abstract: S2 版が arXiv 版より長ければ S2 を優先
    s2_abstract  = s2.get("abstract", "")
    use_abstract = s2_abstract if len(s2_abstract) > len(summary_en) else summary_en
    parts.append(use_abstract)

    return "\n\n".join(parts)

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

def build_user_prompt(context: str) -> str:
    return f"""以下の論文情報を、下記のフォーマットで日本語要約してください。

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

=== 論文情報 ===
{context}
"""

def remove_thinking_block(text: str) -> str:
    """Qwen3 の <think>...</think> ブロックを除去する"""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

def summarize_to_japanese(context: str) -> str:
    api_key, chat_url, model_name = _get_provider()
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": build_user_prompt(context)},
        ],
        "max_tokens": 1024,
        "temperature": 0.2,
        "top_p": 0.95,
    }

    # 429 のバックオフは指数的に伸ばす（30→120→300秒）
    # 無料枠のレート制限は短時間待機では回復しないことが多い
    backoff = [30, 120, 300]

    for attempt in range(3):
        try:
            response = requests.post(chat_url, headers=headers, json=payload, timeout=90)
            print(f"  LLM STATUS: {response.status_code} "
                  f"(attempt {attempt + 1}, provider={LLM_PROVIDER}, model={model_name})")

            if response.status_code == 429:
                wait = backoff[attempt]
                print(f"  Rate limited. Waiting {wait}s...")
                time.sleep(wait)
                continue

            if response.status_code != 200:
                print(f"  LLM ERROR: {response.status_code} {response.text[:300]}")
                time.sleep(10)
                continue

            content = response.json()["choices"][0]["message"]["content"].strip()
            if not content:
                print("  LLM BLANK CONTENT")
                return "要約失敗"

            content = remove_thinking_block(content)  # Qwen3 対応
            return content

        except requests.RequestException as e:
            print(f"  LLM request error (attempt {attempt + 1}): {e}")
            time.sleep(10)

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
# ==========================
def fetch() -> List[dict]:
    """arXiv から論文を取得して返す。上限カットはせず全件返す。
    新規かどうかの判定と件数制限は main() で行う。"""
    seen: set = set()
    papers: List[dict] = []

    for q in ARXIV_QUERIES:
        feed = feedparser.parse(ARXIV_BASE + q)
        time.sleep(1)

        fetched_in_query = 0
        for e in feed.entries:
            if e.id in seen:
                continue
            seen.add(e.id)
            fetched_in_query += 1
            papers.append({
                "id":         e.id,
                "title":      e.title.strip(),
                "summary_en": clean_abstract(e.summary),
                "published":  e.published,
                "link":       e.link,
            })

        label = q.split("&")[0][13:]
        print(f"fetch: '{label[:50]}' got={fetched_in_query}")

    papers.sort(key=lambda x: x["published"], reverse=True)
    print(f"fetch total: {len(papers)} unique papers")
    return papers  # ← 全件返す（上限カットなし）

# ==========================
# DB クリーンアップ
# ==========================
RETENTION_DAYS = 30
SAVED_FILE     = "saved.json"

def load_saved_ids() -> set:
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
    if not paper_id:
        return paper_id
    if paper_id.startswith("http://") or paper_id.startswith("https://"):
        return paper_id
    if paper_id.startswith("abs/"):
        return "http://arxiv.org/" + paper_id
    return "http://arxiv.org/abs/" + paper_id

def cleanup_db(db: list, saved_ids: set, retention_days: int, today: date) -> Tuple[list, int]:
    cutoff  = today.toordinal() - retention_days
    kept    = []
    removed = 0

    for p in db:
        pid = normalize_id(p.get("id", ""))
        if pid in saved_ids:
            kept.append(p)
            continue
        published_str = p.get("published", "")
        try:
            pub_date = datetime.fromisoformat(
                published_str.replace("Z", "+00:00")
            ).date()
            if pub_date.toordinal() >= cutoff:
                kept.append(p)
            else:
                removed += 1
                print(f"  remove (expired): {p.get('title', pid)[:60]}")
        except (ValueError, AttributeError):
            kept.append(p)

    return kept, removed

# ==========================
# メイン処理
# ==========================
def main():
    try:
        _, _, model_name = _get_provider()  # 起動時にAPIキー設定チェック
    except RuntimeError as e:
        print(f"ERROR: {e}")
        return

    print(f"Using model: {model_name} (provider={LLM_PROVIDER})")

    db        = load_db()
    db_map    = {p["id"]: p for p in db}
    papers    = fetch()
    new_count = 0
    today_jst = datetime.now(JST).date()

    # ── 新規論文の抽出（全件チェック）──────────────────────────────
    new_papers = [p for p in papers if p["id"] not in db_map]
    skip_count = len(papers) - len(new_papers)
    print(f"  existing={skip_count}, new={len(new_papers)}, "
          f"will summarize up to {MAX_SUMMARIZE_PER_RUN}")

    # ── 要約処理（上限件数まで）────────────────────────────────────
    for p in new_papers[:MAX_SUMMARIZE_PER_RUN]:
        print(f"processing: {p['title'][:70]}")

        # S2 レート制限対策: 論文間に待機を入れる
        if new_count > 0:
            time.sleep(6)  # 3 → 6秒（S2の100req/5min制限に対して余裕を持たせる）

        s2  = fetch_s2_info(p["id"])
        ctx = build_context(p["summary_en"], s2)

        if s2.get("tldr"):
            print(f"  S2 tldr found ({len(s2['tldr'])} chars)")
        elif s2.get("abstract"):
            print(f"  S2 abstract found ({len(s2['abstract'])} chars)")
        else:
            print("  S2 not available, using arXiv abstract only")

        summary_raw = summarize_to_japanese(ctx)

        if summary_raw == "要約失敗":
            summary_clean = "要約生成に失敗しました。"
            tags = infer_tags(p["title"], p["summary_en"])
        else:
            summary_clean, tags = extract_tags(summary_raw)
            if not tags:
                tags = infer_tags(p["title"], p["summary_en"])

        entry: dict = {
            "id":         p["id"],
            "title":      p["title"],
            "summary_ja": summary_clean,
            "tags":       tags,
            "link":       p["link"],
            "published":  p["published"],
            "created_at": today_jst.isoformat(),
        }
        if s2.get("citation_count"):
            entry["citation_count"] = s2["citation_count"]

        db_map[p["id"]] = entry
        new_count += 1

        # arXiv 利用規約に配慮した待機
        time.sleep(2)

    # ── DB クリーンアップ ─────────────────────────────────────────
    saved_ids  = load_saved_ids()
    print(f"saved papers: {len(saved_ids)}")

    current_db = list(db_map.values())
    cleaned_db, removed_count = cleanup_db(current_db, saved_ids, RETENTION_DAYS, today_jst)
    print(f"cleanup: {removed_count} paper(s) removed (older than {RETENTION_DAYS} days, not saved).")

    # ── 書き込み（変更があった場合のみ）─────────────────────────────
    if new_count > 0 or removed_count > 0:
        updated = sorted(cleaned_db, key=lambda x: x["published"], reverse=True)
        save_db(updated)
        print(f"DB updated: +{new_count} added, -{removed_count} removed. total={len(updated)}")
    else:
        print("no changes to DB.")

if __name__ == "__main__":
    main()
