# arxiv-ai-digest

Fetches latest arXiv papers (AI, robotics, ML) and generates Japanese summaries using Hugging Face inference. Results are stored in `papers_db.json` and shown on GitHub Pages.

## Live site (GitHub Pages)

Enable **Settings → Pages** (source: branch **main**, folder **/ (root)**).

- **URL:** `https://<your-username>.github.io/arxiv-ai-digest/`

### Mobile access

Bookmark the URL or use **Add to Home Screen** (iOS Safari / Android Chrome). The site uses a web app manifest so the home-screen icon shows as **arXiv Digest**.

### Cross-device saved list

To share the same "Saved" list across devices and browsers, open the **Settings** panel on the site and set **Repository** (e.g. `username/arxiv-ai-digest`) and **GitHub Token**.

- **saved.json** in the repo holds the list of saved paper IDs. Pages serves it, so every device reads the same list.
- **Writing** (when you click Save) uses the GitHub API and requires a token. Store the token in Settings on each device where you want to update the shared list; it is kept only in that browser’s localStorage.
- **Security:** Do not share your token. If it leaks, revoke it on GitHub and create a new one, then update Settings on each device.

#### GitHub token (Fine-grained)

Use a **Fine-grained** personal access token with minimal permissions:

1. [GitHub → Settings → Developer settings → Personal access tokens → Fine-grained tokens](https://github.com/settings/tokens?type=beta) → **Generate new token**.
2. **Token name:** Any label (e.g. `arxiv-ai-digest saved`). Does not affect the API.
3. **Repository access:** Only this repository (e.g. your `arxiv-ai-digest`).
4. **Permissions → Repository permissions:** **Contents** → **Read and write**. (Metadata may be set to Read-only automatically; that is fine.)
5. **Expiration:** 30 days is fine; you will need to create a new token and update Settings on each device when it expires. For less maintenance, choose 90 days, 1 year, or No expiration.

Then paste the token into the site’s Settings panel and save.

## Setup

```bash
pip install -r requirements.txt
```

Set your Hugging Face token:

```bash
export HF_TOKEN=your_token_here   # Linux/macOS
# or
$env:HF_TOKEN = "your_token_here"  # PowerShell
```

Create a token at [Hugging Face → Settings → Access Tokens](https://huggingface.co/settings/tokens) with **Inference** permission.

## Usage

```bash
python main.py
```

- Fetches papers from three tag-based arXiv queries: **AIエージェント** (cs.AI + agent), **Robotics** (cs.RO), **ハンド模倣学習** (hand / imitation / dexterous). Results are merged, sorted by date, and limited to 6 per run.
- Generates Japanese summaries and tags via the HF API. Appends new entries to `papers_db.json` (existing papers are skipped).

## Automation

A GitHub Action runs daily (07:00 JST) and on manual trigger:

- **Actions** → **daily arXiv summary** → **Run workflow**.

The workflow updates `papers_db.json` on the `main` branch. Set the repository secret **HF_TOKEN** under **Settings → Secrets and variables → Actions**.

## Output

**papers_db.json** — one object per paper: `id`, `title`, `link`, `published`, `summary_en`, `summary_ja`, `tags`, `saved`, `created_at`.

**saved.json** — list of saved paper IDs when using cross-device Settings; otherwise the list is stored only in the browser (localStorage).
