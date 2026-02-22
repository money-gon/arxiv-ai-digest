# arxiv-ai-digest

Fetches latest arXiv papers (AI, robotics, ML) and generates Japanese summaries using Hugging Face inference. Results are stored in `papers_db.json` and shown on GitHub Pages.

## Live site (GitHub Pages)

After enabling Pages in **Settings → Pages** (source: branch **main**, folder **/ (root)**):

- **URL:** `https://<your-username>.github.io/arxiv-ai-digest/`

### Access from smartphone

- **Bookmark:** Open the URL in the browser and add it to bookmarks.
- **Home screen (iOS):** Safari → Share → **Add to Home Screen** → name it e.g. "arXiv Digest" and add.
- **Home screen (Android):** Chrome → menu (⋮) → **Add to Home screen** or **Install app**.

Then open the site with one tap from your home screen. The page is responsive; a web app manifest is included so the home-screen icon shows as **arXiv Digest**.

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

- Fetches up to 5 newest papers from arXiv (categories: cs.AI, cs.RO, cs.LG).
- Generates Japanese summaries (background, method, results, impact) via the HF API.
- Appends new entries to `papers_db.json` (existing papers are skipped).

## Automation

A GitHub Action runs daily (07:00 JST) and on manual trigger:

- **Actions** → **daily arXiv summary** → **Run workflow**.

The workflow updates `papers_db.json` on the `main` branch. Ensure the repository secret **HF_TOKEN** is set under **Settings → Secrets and variables → Actions**.

## Output

`papers_db.json` contains one object per paper with:

- `id`, `title`, `link`, `published`
- `summary_en` — original abstract
- `summary_ja` — Japanese summary
- `saved`, `created_at`
