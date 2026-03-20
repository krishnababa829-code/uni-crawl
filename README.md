# uni-crawl

Advanced async web scraping and AI data extraction framework built exclusively for free, ephemeral Jupyter environments (Google Colab & Kaggle).
Features


Async-first: Pure asyncio with nest_asyncio patching for notebook compatibility

Stealth browsing: Playwright + stealth evasion for dynamic JS-rendered pages

Multi-page spidering: Concurrent internal-link traversal with depth/page limits

AI-ready NLP pipeline: HTML-to-Markdown, citation extraction, multi-strategy chunking

Semantic filtering: BM25 (lexical) and cosine-similarity (embedding) chunk ranking

LLM structured extraction: Pydantic schema enforcement with local HuggingFace models

Fallback extraction: CSS selector and XPath extractors that bypass LLMs entirely

Ephemeral storage: All caches and sessions default to /tmp/ directories

Quick Start (Google Colab)


!pip install opus-crawl
!playwright install chromium

from opus_crawl import BrowserEngine, NLPProcessor, LLMExtractor

# Scrape
engine = BrowserEngine()
pages = await engine.crawl("https://example.com", max_depth=1, max_pages=10)

# Process
processor = NLPProcessor()
for page in pages:
    md = processor.to_markdown(page.html)
    chunks = processor.chunk(md, strategy="sentence", max_tokens=512)
    print(chunks[:2])

# Extract structured data
from pydantic import BaseModel

class Article(BaseModel):
    title: str
    summary: str
    tags: list[str]

extractor = LLMExtractor(model_name="google/flan-t5-base")
result = await extractor.extract(chunks, Article)
print(result)


