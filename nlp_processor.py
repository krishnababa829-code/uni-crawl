"""nlp_processor.py - Convert raw DOM into LLM-friendly text chunks.

This module sits between the browser engine (raw HTML) and the LLM
extractor (structured output).  It provides:

1. **HTML -> Markdown** conversion with noise stripping.
2. **Citation extraction** - page links become numbered references.
3. **Chunking strategies** (Strategy pattern):
   - Word-count overlap
   - Sentence-level (regex)
   - Custom delimiter
4. **Semantic filtering**:
   - BM25 lexical ranking
   - Cosine-similarity ranking via ``sentence-transformers``
"""

from __future__ import annotations

import abc
import math
import re
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

from bs4 import BeautifulSoup, Tag
from markdownify import markdownify as md  # type: ignore[import-untyped]

# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------


@dataclass
class Citation:
    """A numbered reference extracted from the page."""

    index: int
    text: str
    url: str


@dataclass
class ProcessedPage:
    """Result of the full NLP pipeline for a single page."""

    markdown: str
    citations: list[Citation] = field(default_factory=list)
    chunks: list[str] = field(default_factory=list)


# ===================================================================
# HTML -> Markdown
# ===================================================================

# Tags considered "noise" by default.
_DEFAULT_STRIP_TAGS: set[str] = {
    "nav",
    "footer",
    "header",
    "aside",
    "script",
    "style",
    "noscript",
    "svg",
    "form",
    "button",
    "iframe",
}


def strip_noise(
    html: str,
    *,
    strip_tags: Optional[set[str]] = None,
    strip_ids: Optional[set[str]] = None,
    strip_classes: Optional[set[str]] = None,
) -> str:
    """Remove noisy DOM elements and return cleaned HTML.

    Parameters
    ----------
    html : str
        Raw page HTML.
    strip_tags : set[str] | None
        Tag names to remove.  Merged with the built-in defaults.
    strip_ids : set[str] | None
        Element ``id`` values to remove (exact match).
    strip_classes : set[str] | None
        CSS class names to remove (any element containing the class).
    """
    tags = _DEFAULT_STRIP_TAGS | (strip_tags or set())
    soup = BeautifulSoup(html, "lxml")

    # Remove by tag name
    for tag_name in tags:
        for el in soup.find_all(tag_name):
            el.decompose()

    # Remove by id
    if strip_ids:
        for sid in strip_ids:
            el = soup.find(id=sid)
            if el and isinstance(el, Tag):
                el.decompose()

    # Remove by class
    if strip_classes:
        for cls in strip_classes:
            for el in soup.find_all(class_=cls):
                if isinstance(el, Tag):
                    el.decompose()

    return str(soup)


def html_to_markdown(
    html: str,
    *,
    strip_tags: Optional[set[str]] = None,
    strip_ids: Optional[set[str]] = None,
    strip_classes: Optional[set[str]] = None,
    heading_style: str = "ATX",
) -> str:
    """Convert HTML to clean Markdown.

    Strips noise first, then delegates to ``markdownify``.
    """
    cleaned = strip_noise(
        html,
        strip_tags=strip_tags,
        strip_ids=strip_ids,
        strip_classes=strip_classes,
    )
    text: str = md(cleaned, heading_style=heading_style, strip=["img"])
    # Collapse excessive blank lines
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text


# ===================================================================
# Citation extraction
# ===================================================================


def extract_citations(html: str) -> tuple[str, list[Citation]]:
    """Replace inline ``<a>`` links with numbered references.

    Returns the modified Markdown **and** the citation list.

    Example output
    --------------
    ``Some text [1] more text [2]``

    References:
    [1] https://example.com/page1
    [2] https://example.com/page2
    """
    soup = BeautifulSoup(html, "lxml")
    citations: list[Citation] = []
    seen_urls: dict[str, int] = {}

    for anchor in soup.find_all("a", href=True):
        href: str = anchor["href"]
        if href.startswith(("#", "javascript:")):
            continue
        link_text = anchor.get_text(strip=True) or href

        if href in seen_urls:
            idx = seen_urls[href]
        else:
            idx = len(citations) + 1
            seen_urls[href] = idx
            citations.append(Citation(index=idx, text=link_text, url=href))

        anchor.replace_with(f"{link_text} [{idx}]")

    markdown = md(str(soup), heading_style="ATX", strip=["img"])
    markdown = re.sub(r"\n{3,}", "\n\n", markdown).strip()

    if citations:
        ref_block = "\n\n---\n**References**\n"
        for c in citations:
            ref_block += f"[{c.index}] {c.url}\n"
        markdown += ref_block

    return markdown, citations


# ===================================================================
# Chunking strategies (Strategy pattern)
# ===================================================================


class ChunkingStrategy(abc.ABC):
    """Abstract base for all chunking strategies."""

    @abc.abstractmethod
    def chunk(self, text: str) -> list[str]:
        """Split *text* into a list of chunks."""


class WordOverlapChunker(ChunkingStrategy):
    """Split by word count with configurable overlap.

    Parameters
    ----------
    max_words : int
        Maximum words per chunk.
    overlap_words : int
        Number of trailing words from the previous chunk prepended to the
        next chunk for context continuity.
    """

    def __init__(self, max_words: int = 300, overlap_words: int = 50) -> None:
        if overlap_words >= max_words:
            raise ValueError("overlap_words must be less than max_words")
        self._max = max_words
        self._overlap = overlap_words

    def chunk(self, text: str) -> list[str]:
        words = text.split()
        if not words:
            return []
        chunks: list[str] = []
        start = 0
        while start < len(words):
            end = start + self._max
            chunks.append(" ".join(words[start:end]))
            start += self._max - self._overlap
        return chunks


class SentenceChunker(ChunkingStrategy):
    """Split text into sentence-level chunks grouped up to *max_sentences*.

    Uses a regex splitter that handles common abbreviations.
    """

    # Sentence boundary: period / question / exclamation followed by
    # whitespace and an uppercase letter (or end of string).
    _SENT_RE = re.compile(
        r"(?<=[.!?])\s+(?=[A-Z\u00C0-\u024F])"
    )

    def __init__(self, max_sentences: int = 5) -> None:
        self._max = max_sentences

    def chunk(self, text: str) -> list[str]:
        sentences = self._SENT_RE.split(text)
        sentences = [s.strip() for s in sentences if s.strip()]
        if not sentences:
            return []
        chunks: list[str] = []
        for i in range(0, len(sentences), self._max):
            group = sentences[i : i + self._max]
            chunks.append(" ".join(group))
        return chunks


class DelimiterChunker(ChunkingStrategy):
    """Split on a user-supplied delimiter string.

    Parameters
    ----------
    delimiter : str
        The string to split on (e.g. ``"\n\n"`` for paragraphs,
        ``"---"`` for horizontal rules).
    strip_empty : bool
        Drop empty chunks after splitting.
    """

    def __init__(self, delimiter: str = "\n\n", *, strip_empty: bool = True) -> None:
        self._delim = delimiter
        self._strip_empty = strip_empty

    def chunk(self, text: str) -> list[str]:
        parts = text.split(self._delim)
        if self._strip_empty:
            parts = [p.strip() for p in parts if p.strip()]
        return parts


# ===================================================================
# Semantic filtering
# ===================================================================


class BM25Chunker:
    """Rank and filter chunks using Okapi BM25 (lexical).

    Parameters
    ----------
    top_k : int
        Number of top-ranked chunks to return.
    """

    def __init__(self, top_k: int = 5) -> None:
        self._top_k = top_k

    def filter(
        self,
        chunks: Sequence[str],
        query: str,
    ) -> list[str]:
        """Return the *top_k* chunks most relevant to *query*."""
        from rank_bm25 import BM25Okapi  # type: ignore[import-untyped]

        if not chunks:
            return []

        tokenized_corpus = [c.lower().split() for c in chunks]
        bm25 = BM25Okapi(tokenized_corpus)
        tokenized_query = query.lower().split()
        scores = bm25.get_scores(tokenized_query)

        ranked_indices = sorted(
            range(len(scores)), key=lambda i: scores[i], reverse=True
        )
        top_indices = ranked_indices[: self._top_k]
        # Preserve original order for readability.
        top_indices.sort()
        return [chunks[i] for i in top_indices]


class CosineSimilarityChunker:
    """Rank and filter chunks using embedding cosine similarity.

    Uses ``sentence-transformers`` with a lightweight model suitable for
    Colab / Kaggle free-tier GPUs.

    Parameters
    ----------
    model_name : str
        HuggingFace model identifier.
    top_k : int
        Number of top-ranked chunks to return.
    device : str | None
        PyTorch device string.  ``None`` = auto-detect.
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        top_k: int = 5,
        device: Optional[str] = None,
    ) -> None:
        self._model_name = model_name
        self._top_k = top_k
        self._device = device
        self._model: Any = None  # lazy-loaded

    def _ensure_model(self) -> Any:
        if self._model is None:
            from sentence_transformers import SentenceTransformer  # type: ignore[import-untyped]

            self._model = SentenceTransformer(
                self._model_name, device=self._device
            )
        return self._model

    @staticmethod
    def _cosine(a: Any, b: Any) -> float:
        """Cosine similarity between two 1-D numpy arrays."""
        import numpy as np  # type: ignore[import-untyped]

        dot = float(np.dot(a, b))
        norm_a = float(np.linalg.norm(a))
        norm_b = float(np.linalg.norm(b))
        if norm_a == 0.0 or norm_b == 0.0:
            return 0.0
        return dot / (norm_a * norm_b)

    def filter(
        self,
        chunks: Sequence[str],
        query: str,
    ) -> list[str]:
        """Return the *top_k* chunks most semantically similar to *query*."""
        if not chunks:
            return []

        model = self._ensure_model()
        query_emb = model.encode(query, convert_to_numpy=True)
        chunk_embs = model.encode(list(chunks), convert_to_numpy=True)

        scores = [
            self._cosine(query_emb, chunk_embs[i])
            for i in range(len(chunks))
        ]

        ranked_indices = sorted(
            range(len(scores)), key=lambda i: scores[i], reverse=True
        )
        top_indices = ranked_indices[: self._top_k]
        top_indices.sort()
        return [chunks[i] for i in top_indices]


# ===================================================================
# Facade
# ===================================================================

# Registry of built-in chunking strategies.
_STRATEGY_REGISTRY: dict[str, type[ChunkingStrategy]] = {
    "word": WordOverlapChunker,
    "sentence": SentenceChunker,
    "delimiter": DelimiterChunker,
}


class NLPProcessor:
    """High-level facade that chains noise removal, markdown conversion,
    citation extraction, and chunking into a single call.

    Parameters
    ----------
    strip_tags : set[str] | None
        Extra HTML tags to strip (merged with defaults).
    strip_ids : set[str] | None
        Element IDs to strip.
    strip_classes : set[str] | None
        CSS classes to strip.
    default_strategy : str
        Default chunking strategy name (``word``, ``sentence``, or
        ``delimiter``).
    strategy_kwargs : dict | None
        Keyword arguments forwarded to the default chunking strategy
        constructor.
    """

    def __init__(
        self,
        *,
        strip_tags: Optional[set[str]] = None,
        strip_ids: Optional[set[str]] = None,
        strip_classes: Optional[set[str]] = None,
        default_strategy: str = "word",
        strategy_kwargs: Optional[dict[str, Any]] = None,
    ) -> None:
        self._strip_tags = strip_tags
        self._strip_ids = strip_ids
        self._strip_classes = strip_classes

        cls = _STRATEGY_REGISTRY.get(default_strategy)
        if cls is None:
            raise ValueError(
                f"Unknown strategy '{default_strategy}'. "
                f"Choose from: {list(_STRATEGY_REGISTRY)}"
            )
        self._chunker: ChunkingStrategy = cls(**(strategy_kwargs or {}))

    # -- convenience wrappers -------------------------------------------

    def to_markdown(self, html: str) -> str:
        """Convert raw HTML to clean Markdown."""
        return html_to_markdown(
            html,
            strip_tags=self._strip_tags,
            strip_ids=self._strip_ids,
            strip_classes=self._strip_classes,
        )

    def to_markdown_with_citations(self, html: str) -> tuple[str, list[Citation]]:
        """Convert HTML to Markdown with numbered citation references."""
        cleaned = strip_noise(
            html,
            strip_tags=self._strip_tags,
            strip_ids=self._strip_ids,
            strip_classes=self._strip_classes,
        )
        return extract_citations(cleaned)

    def chunk(
        self,
        text: str,
        *,
        strategy: Optional[str] = None,
        strategy_kwargs: Optional[dict[str, Any]] = None,
    ) -> list[str]:
        """Chunk *text* using the given (or default) strategy.

        Parameters
        ----------
        text : str
            Plain text or Markdown to chunk.
        strategy : str | None
            Override the default strategy for this call.
        strategy_kwargs : dict | None
            Override constructor kwargs for the strategy.
        """
        if strategy is not None:
            cls = _STRATEGY_REGISTRY.get(strategy)
            if cls is None:
                raise ValueError(
                    f"Unknown strategy '{strategy}'. "
                    f"Choose from: {list(_STRATEGY_REGISTRY)}"
                )
            chunker: ChunkingStrategy = cls(**(strategy_kwargs or {}))
        else:
            chunker = self._chunker
        return chunker.chunk(text)

    def process(
        self,
        html: str,
        *,
        with_citations: bool = True,
        strategy: Optional[str] = None,
        strategy_kwargs: Optional[dict[str, Any]] = None,
    ) -> ProcessedPage:
        """Run the full pipeline: clean -> markdown -> cite -> chunk."""
        if with_citations:
            markdown, citations = self.to_markdown_with_citations(html)
        else:
            markdown = self.to_markdown(html)
            citations = []

        chunks = self.chunk(markdown, strategy=strategy, strategy_kwargs=strategy_kwargs)

        return ProcessedPage(
            markdown=markdown,
            citations=citations,
            chunks=chunks,
        )

    @staticmethod
    def register_strategy(name: str, cls: type[ChunkingStrategy]) -> None:
        """Register a custom chunking strategy globally."""
        _STRATEGY_REGISTRY[name] = cls
