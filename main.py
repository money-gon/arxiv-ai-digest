import feedparser
from datetime import datetime

ARXIV_URL = (
    "http://export.arxiv.org/api/query?"
    "search_query=cat:cs.AI+OR+cat:cs.RO+OR+cat:cs.LG&"
    "sortBy=submittedDate&max_results=5"
)

def fetch_latest_papers():
    feed = feedparser.parse(ARXIV_URL)
    papers = []

    for entry in feed.entries:
        paper = {
            "id": entry.id,
            "title": entry.title,
            "summary": entry.summary,
            "published": entry.published,
            "link": entry.link
        }
        papers.append(paper)

    return papers


def main():
    papers = fetch_latest_papers()

    print(f"\n=== arXiv Latest Papers ({datetime.now()}) ===\n")

    for i, paper in enumerate(papers, 1):
        print(f"{i}. {paper['title']}")
        print(f"   Published: {paper['published']}")
        print(f"   Link: {paper['link']}\n")


if __name__ == "__main__":
    main()