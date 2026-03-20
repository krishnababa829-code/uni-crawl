"""llm_extractor.py - Structured output engine.

Converts LLM-friendly text chunks into validated Pydantic models using
either a local HuggingFace model **or** deterministic CSS / XPath
fallback extractors.

Key capabilities
----------------
* **Schema enforcement** via Pydantic ``BaseModel`` subclasses.
* **Local LLM pipeline** using ``transformers`` with automatic GPU
  detection (``device_map="auto"``, ``torch.bfloat16``).
* **JSON cleaning** - strips Markdown fences, repairs truncated JSON.
* **Fallback extraction** - ``CSSExtractor`` and ``XPathExtractor``
  bypass the LLM entirely for deterministic scraping.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Any, Optional, Sequence, Type, TypeVar, Union

from pydantic import BaseModel, ValidationError

logger = logging.getLogger("opus_crawl.llm_extractor")

T = TypeVar("T", bound=BaseModel)

# ===================================================================
# JSON cleaning utilities
# ===================================================================

# Matches ```json ... ``` or ``` ... ``` fences.
_FENCE_RE = re.compile(
    r"```(?:json)?\s*\n?(.*?)\n?\s*```",
    re.DOTALL,
)

# Matches a top-level JSON object or array.
_JSON_OBJECT_RE = re.compile(
    r"(\{[\s\S]*\}|\[[\s\S]*\])",
)


def clean_json(raw: str) -> str:
    """Extract and clean a JSON string from potentially messy LLM output.

    Handles:
    - Markdown code fences (````json ... ````)
    - Leading / trailing prose around the JSON body
    - Truncated JSON (attempts brace / bracket balancing)

    Parameters
    ----------
    raw : str
        Raw text from the LLM.

    Returns
    -------
    str
        Cleaned JSON string (may still be invalid if the LLM
        hallucinated badly).
    """
    # 1. Try to extract from fenced block.
    fence_match = _FENCE_RE.search(raw)
    if fence_match:
        raw = fence_match.group(1).strip()

    # 2. Try to isolate the JSON object / array.
    obj_match = _JSON_OBJECT_RE.search(raw)
    if obj_match:
        raw = obj_match.group(1).strip()

    # 3. Attempt brace / bracket balancing for truncated output.
    raw = _balance_braces(raw)

    return raw


def _balance_braces(text: str) -> str:
    """Append missing closing braces / brackets to truncated JSON."""
    stack: list[str] = []
    in_string = False
    escape = False

    for ch in text:
        if escape:
            escape = False
            continue
        if ch == "\\":
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch in ("{{", "["):
            stack.append(ch)
        elif ch == "}}" and stack and stack[-1] == "{{":
            stack.pop()
        elif ch == "]" and stack and stack[-1] == "[":
            stack.pop()

    # Close any unclosed openers.
    closers = {"{": "}", "[": "]"}
    while stack:
        opener = stack.pop()
        text += closers.get(opener, "")

    return text


# ===================================================================
# LLM Extractor
# ===================================================================


class LLMExtractor:
    """Extract structured data from text chunks using a local HuggingFace model.

    Parameters
    ----------
    model_name : str
        HuggingFace model identifier.  Defaults to ``google/flan-t5-base``.
    device_map : str
        Passed to ``transformers.pipeline``.  ``"auto"`` uses the T4 GPU
        on Colab when available.
    torch_dtype : str
        String representation of the dtype (``"bfloat16"``, ``"float16"``,
        ``"float32"``).  Converted to the matching ``torch.dtype``.
    max_new_tokens : int
        Maximum tokens the model may generate per call.
    temperature : float
        Sampling temperature.  ``0.0`` = greedy.
    """

    def __init__(
        self,
        model_name: str = "google/flan-t5-base",
        *,
        device_map: str = "auto",
        torch_dtype: str = "bfloat16",
        max_new_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> None:
        self._model_name = model_name
        self._device_map = device_map
        self._torch_dtype_str = torch_dtype
        self._max_new_tokens = max_new_tokens
        self._temperature = temperature
        self._pipeline: Any = None  # lazy

    # -- lazy init ------------------------------------------------------

    def _ensure_pipeline(self) -> Any:
        """Build the ``transformers`` text2text-generation pipeline once."""
        if self._pipeline is not None:
            return self._pipeline

        import torch
        from transformers import pipeline as hf_pipeline  # type: ignore[import-untyped]

        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        torch_dtype = dtype_map.get(self._torch_dtype_str, torch.bfloat16)

        self._pipeline = hf_pipeline(
            "text2text-generation",
            model=self._model_name,
            device_map=self._device_map,
            torch_dtype=torch_dtype,
        )
        logger.info(
            "Loaded model %s (dtype=%s, device_map=%s)",
            self._model_name,
            self._torch_dtype_str,
            self._device_map,
        )
        return self._pipeline

    # -- prompt construction -------------------------------------------

    @staticmethod
    def _build_prompt(text: str, schema: Type[T]) -> str:
        """Build an instruction prompt that asks the model to output JSON
        conforming to the Pydantic schema."""
        schema_json = json.dumps(schema.model_json_schema(), indent=2)
        return (
            "You are a precise data extraction assistant.\n"
            "Extract structured data from the following text and return "
            "ONLY valid JSON matching this schema:\n\n"
            f"```json\n{schema_json}\n```\n\n"
            f"Text:\n{text}\n\n"
            "Respond with ONLY the JSON object, no explanation."
        )

    # -- single-chunk extraction ---------------------------------------

    def _extract_single_sync(
        self,
        text: str,
        schema: Type[T],
    ) -> T:
        """Run extraction on a single chunk (synchronous)."""
        pipe = self._ensure_pipeline()
        prompt = self._build_prompt(text, schema)

        gen_kwargs: dict[str, Any] = {
            "max_new_tokens": self._max_new_tokens,
        }
        if self._temperature > 0:
            gen_kwargs["temperature"] = self._temperature
            gen_kwargs["do_sample"] = True
        else:
            gen_kwargs["do_sample"] = False

        outputs = pipe(prompt, **gen_kwargs)
        raw_text: str = outputs[0]["generated_text"]
        cleaned = clean_json(raw_text)

        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Model output is not valid JSON after cleaning: {cleaned!r}"
            ) from exc

        return schema.model_validate(data)

    # -- public async API -----------------------------------------------

    async def extract(
        self,
        chunks: Union[str, Sequence[str]],
        schema: Type[T],
        *,
        merge: bool = True,
    ) -> Union[T, list[T]]:
        """Extract structured data from one or more text chunks.

        Parameters
        ----------
        chunks : str | Sequence[str]
            A single text block or a list of chunks.
        schema : Type[BaseModel]
            Pydantic model class defining the extraction target.
        merge : bool
            If ``True`` and multiple chunks are provided, attempt to
            merge results into a single model instance by taking the
            first non-empty value for each field.  If ``False``, return
            a list of model instances.

        Returns
        -------
        T | list[T]
            Validated Pydantic model instance(s).
        """
        if isinstance(chunks, str):
            chunks = [chunks]

        loop = asyncio.get_event_loop()
        results: list[T] = []

        for chunk in chunks:
            try:
                result = await loop.run_in_executor(
                    None, self._extract_single_sync, chunk, schema
                )
                results.append(result)
            except (ValueError, ValidationError) as exc:
                logger.warning("Extraction failed for chunk: %s", exc)
                continue

        if not results:
            raise ValueError("Extraction failed for all chunks.")

        if merge and len(results) > 1:
            return self._merge_results(results, schema)

        return results[0] if len(results) == 1 else results

    @staticmethod
    def _merge_results(results: list[T], schema: Type[T]) -> T:
        """Merge multiple extraction results into one by picking the first
        non-empty value for each field."""
        merged: dict[str, Any] = {}
        field_names = list(schema.model_fields.keys())

        for field_name in field_names:
            for result in results:
                value = getattr(result, field_name)
                if value is not None and value != "" and value != []:
                    merged[field_name] = value
                    break
            else:
                # Use the value from the first result as fallback.
                merged[field_name] = getattr(results[0], field_name)

        return schema.model_validate(merged)


# ===================================================================
# Fallback extractors (no LLM required)
# ===================================================================


class CSSExtractor:
    """Extract data from HTML using CSS selectors.

    Parameters
    ----------
    schema : dict[str, str]
        Mapping of output field names to CSS selectors.
        Each selector should match a single element whose text content
        becomes the field value.  Prefix with ``@attr:`` to extract an
        attribute instead (e.g. ``@href:a.link``).

    Example
    -------
    >>> ext = CSSExtractor({"title": "h1.main", "link": "@href:a.primary"})
    >>> ext.extract(html)
    {"title": "Hello World", "link": "https://example.com"}
    """

    # Pattern: @attrname:selector
    _ATTR_RE = re.compile(r"^@(\w+):(.+)$")

    def __init__(self, schema: dict[str, str]) -> None:
        self._schema = schema

    def extract(self, html: str) -> dict[str, Any]:
        """Run CSS extraction and return a plain dict."""
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(html, "lxml")
        result: dict[str, Any] = {}

        for field_name, selector in self._schema.items():
            attr_match = self._ATTR_RE.match(selector)
            if attr_match:
                attr_name, css = attr_match.group(1), attr_match.group(2)
                el = soup.select_one(css.strip())
                result[field_name] = el.get(attr_name, "") if el else ""
            else:
                el = soup.select_one(selector)
                result[field_name] = el.get_text(strip=True) if el else ""

        return result

    def extract_many(self, html: str) -> list[dict[str, Any]]:
        """Extract multiple records when selectors match many elements.

        Uses the first selector to determine the number of records,
        then zips all selectors by index.
        """
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(html, "lxml")
        first_key = next(iter(self._schema))
        first_selector = self._schema[first_key]

        attr_match = self._ATTR_RE.match(first_selector)
        if attr_match:
            _, css = attr_match.group(1), attr_match.group(2)
            anchors = soup.select(css.strip())
        else:
            anchors = soup.select(first_selector)

        count = len(anchors)
        records: list[dict[str, Any]] = [{} for _ in range(count)]

        for field_name, selector in self._schema.items():
            attr_match = self._ATTR_RE.match(selector)
            if attr_match:
                attr_name, css = attr_match.group(1), attr_match.group(2)
                elements = soup.select(css.strip())
                for i in range(min(count, len(elements))):
                    records[i][field_name] = elements[i].get(attr_name, "")
            else:
                elements = soup.select(selector)
                for i in range(min(count, len(elements))):
                    records[i][field_name] = elements[i].get_text(strip=True)

        return records

    def extract_validated(
        self,
        html: str,
        model: Type[T],
    ) -> T:
        """Extract and validate against a Pydantic model."""
        data = self.extract(html)
        return model.model_validate(data)


class XPathExtractor:
    """Extract data from HTML using XPath expressions.

    Parameters
    ----------
    schema : dict[str, str]
        Mapping of output field names to XPath expressions.

    Example
    -------
    >>> ext = XPathExtractor({"title": "//h1/text()", "link": "//a/@href"})
    >>> ext.extract(html)
    {"title": "Hello World", "link": "https://example.com"}
    """

    def __init__(self, schema: dict[str, str]) -> None:
        self._schema = schema

    def extract(self, html: str) -> dict[str, Any]:
        """Run XPath extraction and return a plain dict."""
        from lxml import etree  # type: ignore[import-untyped]

        tree = etree.HTML(html)
        if tree is None:
            return {field: "" for field in self._schema}

        result: dict[str, Any] = {}
        for field_name, xpath in self._schema.items():
            matches = tree.xpath(xpath)
            if not matches:
                result[field_name] = ""
            elif len(matches) == 1:
                result[field_name] = (
                    str(matches[0]).strip()
                    if isinstance(matches[0], str)
                    else matches[0].text_content().strip()
                    if hasattr(matches[0], "text_content")
                    else str(matches[0])
                )
            else:
                result[field_name] = [
                    str(m).strip()
                    if isinstance(m, str)
                    else m.text_content().strip()
                    if hasattr(m, "text_content")
                    else str(m)
                    for m in matches
                ]

        return result

    def extract_many(self, html: str) -> list[dict[str, Any]]:
        """Extract multiple records by treating each XPath result as a list."""
        from lxml import etree  # type: ignore[import-untyped]

        tree = etree.HTML(html)
        if tree is None:
            return []

        columns: dict[str, list[str]] = {}
        max_len = 0

        for field_name, xpath in self._schema.items():
            matches = tree.xpath(xpath)
            values: list[str] = []
            for m in matches:
                if isinstance(m, str):
                    values.append(m.strip())
                elif hasattr(m, "text_content"):
                    values.append(m.text_content().strip())
                else:
                    values.append(str(m))
            columns[field_name] = values
            max_len = max(max_len, len(values))

        records: list[dict[str, Any]] = []
        for i in range(max_len):
            row: dict[str, Any] = {}
            for field_name, values in columns.items():
                row[field_name] = values[i] if i < len(values) else ""
            records.append(row)

        return records

    def extract_validated(
        self,
        html: str,
        model: Type[T],
    ) -> T:
        """Extract and validate against a Pydantic model."""
        data = self.extract(html)
        return model.model_validate(data)
