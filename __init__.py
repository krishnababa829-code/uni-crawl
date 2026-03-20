"""opus-crawl: Advanced async web scraping & AI extraction for Jupyter environments.

Designed to run natively inside Google Colab and Kaggle notebooks.
Uses nest_asyncio to patch the running event loop so all async code
works transparently from a notebook cell.

Quick start
-----------
>>> from opus_crawl import BrowserEngine, NLPProcessor, LLMExtractor
>>> engine = BrowserEngine()
>>> pages = await engine.crawl("https://example.com", max_depth=2)
>>> processor = NLPProcessor()
>>> chunks = processor.process(pages[0].html)
"""

from __future__ import annotations

import nest_asyncio as _nest_asyncio

_nest_asyncio.apply()

__version__ = "0.1.0"
__all__ = [
    "BrowserEngine",
    "NLPProcessor",
    "LLMExtractor",
    "WordOverlapChunker",
    "SentenceChunker",
    "DelimiterChunker",
    "BM25Chunker",
    "CosineSimilarityChunker",
    "CSSExtractor",
    "XPathExtractor",
]

from opus_crawl.browser_engine import BrowserEngine
from opus_crawl.nlp_processor import (
    BM25Chunker,
    CosineSimilarityChunker,
    DelimiterChunker,
    NLPProcessor,
    SentenceChunker,
    WordOverlapChunker,
)
from opus_crawl.llm_extractor import CSSExtractor, LLMExtractor, XPathExtractor
