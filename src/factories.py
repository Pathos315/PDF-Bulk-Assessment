"""`factories.py` is a module for constructing and executing scientific data scrapers.

This module contains functions and classes for scraping scientific data from various sources,
including the Dimensions.ai API and local directories.
It also includes functions for serializing and staging the scraped data.
"""

from __future__ import annotations

from functools import partial
from pathlib import Path
from typing import NoReturn

from src.config import config
from src.docscraper import DocScraper
from src.downloaders import BulkPDFScraper, ImagesDownloader
from src.fetch import SciScraper, ScrapeFetcher, StagingFetcher
from src.log import logger
from src.serials import (
    serialize_from_csv,
    serialize_from_directory,
    serialize_from_txt,
)
from src.stagers import stage_from_series, stage_with_reference
from src.webscrapers import (
    DimensionsScraper,
    GoogleScholarScraper,
    SemanticWebScraper,
)


# Create instances of scrapers
SCRAPERS: dict[str, ScrapeFetcher] = {
    "pdf_lookup": ScrapeFetcher(
        DocScraper(
            Path(config.target_words).resolve(),
            Path(config.bycatch_words).resolve(),
        ),
        serialize_from_directory,
    ),
    "csv_lookup": ScrapeFetcher(
        SemanticWebScraper(
            config.semantic_scholar_url,
            config.sleep_interval,
        ),
        serialize_from_csv,
    ),
    "abstract_lookup": ScrapeFetcher(
        DocScraper(
            Path(config.target_words).resolve(),
            Path(config.bycatch_words).resolve(),
            is_pdf=False,
        ),
        partial(
            serialize_from_csv,
            column="abstract",
        ),
        title_serializer=partial(
            serialize_from_csv,
            column="title",
        ),
    ),
}


# Create instances of stagers
STAGERS: dict[str, StagingFetcher] = {
    "abstracts": StagingFetcher(
        DocScraper(
            Path(config.target_words).resolve(),
            Path(config.bycatch_words).resolve(),
            False,
        ),
        stage_from_series,
    ),
    "download": StagingFetcher(
        BulkPDFScraper(config.downloader_url),
        partial(stage_from_series, column="doi"),
    ),
    "citations": StagingFetcher(
        SemanticWebScraper(
            config.semantic_scholar_url,
            config.sleep_interval,
        ),
        stage_with_reference,
    ),
    "references": StagingFetcher(
        SemanticWebScraper(
            config.semantic_scholar_url,
            config.sleep_interval,
        ),
        partial(stage_with_reference, column_x="references"),
    ),
    "images": StagingFetcher(
        ImagesDownloader(url=""),
        partial(stage_with_reference, column_x="figures"),
    ),
    "pdf_expanded": StagingFetcher(
        SemanticWebScraper(
            config.semantic_scholar_url,
            config.sleep_interval,
        ),
        partial(stage_from_series, column="doi_from_pdf"),
    ),
}


# Create instances of scraper factories for different scraper types
SCISCRAPERS: dict[str, SciScraper | NoReturn] = {
    "directory": SciScraper(SCRAPERS["pdf_lookup"], STAGERS["pdf_expanded"]),
    "wordscore": SciScraper(SCRAPERS["csv_lookup"], STAGERS["abstracts"]),
    "citations": SciScraper(SCRAPERS["csv_lookup"], STAGERS["citations"]),
    "references": SciScraper(SCRAPERS["csv_lookup"], STAGERS["references"]),
    "download": SciScraper(SCRAPERS["csv_lookup"], STAGERS["download"]),
    "images": SciScraper(SCRAPERS["csv_lookup"], STAGERS["images"]),
    "fastscore": SciScraper(SCRAPERS["abstract_lookup"], None),
}


def read_factory() -> SciScraper:
    """
    Constructs an exporter factory based on the user's preference.

    Returns
    ------
    Sciscraper: An instance of Sciscraper, from which the program is run.
    """

    while True:
        scrape_process = input(
            f"Enter desired data scraping process ({', '.join(SCISCRAPERS)}): "
        )
        try:
            return SCISCRAPERS[scrape_process]
        except KeyError:
            logger.error(
                "Unknown data scraping process option: %s.", scrape_process
            )
