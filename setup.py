"""Setup script for opus-crawl."""

from setuptools import find_packages, setup

setup(
    name="opus-crawl",
    version="0.1.0",
    description=(
        "Advanced async web scraping and AI data extraction "
        "framework for Jupyter environments (Colab / Kaggle)."
    ),
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    author="organicsol",
    license="MIT",
    python_requires=">=3.9",
    packages=find_packages(),
    install_requires=[
        "nest_asyncio>=1.5.0",
        "playwright>=1.40.0",
        "playwright-stealth>=1.0.6",
        "pydantic>=2.0",
        "transformers>=4.35.0",
        "sentence-transformers>=2.2.0",
        "rank_bm25>=0.2.2",
        "lxml>=4.9.0",
        "beautifulsoup4>=4.12.0",
        "markdownify>=0.11.0",
        "torch>=2.0.0",
        "accelerate>=0.25.0",
    ],
    extras_require={
        "dev": ["pytest", "pytest-asyncio", "ruff", "mypy"],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Internet :: WWW/HTTP :: Indexing/Search",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
