"""`factories.py` is a module for constructing and executing scientific data scrapers.

This module contains functions and classes for scraping scientific data from various sources,
including the Dimensions.ai API and local directories.
It also includes functions for serializing and staging the scraped data.
"""

from __future__ import annotations

from functools import partial
from pathlib import Path

from src.config import config
from src.docscraper import DocScraper
from src.downloaders import BulkPDFScraper, ImagesDownloader
from src.fetch import SciScraper, ScrapeFetcher, StagingFetcher
from src.log import logger
from src.serials import (
    serialize_from_csv,
    serialize_from_directory,
)
from src.stagers import (
    stage_from_series,
    stage_with_reference,
)
from src.webscrapers import (
    ORCHIDScraper,
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
    "authors": StagingFetcher(
        ORCHIDScraper(
            config.orcid_url,
            config.sleep_interval,
        ),
        partial(
            stage_with_reference,
            column_x="author_list",
        ),
    ),
    "citations": StagingFetcher(
        SemanticWebScraper(
            config.semantic_scholar_url,
            config.sleep_interval,
        ),
        stage_with_reference,
    ),
    "download": StagingFetcher(
        BulkPDFScraper(config.downloader_url),
        partial(stage_from_series, column="doi"),
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
    "references": StagingFetcher(
        SemanticWebScraper(
            config.semantic_scholar_url,
            config.sleep_interval,
        ),
        partial(stage_with_reference, column_x="references"),
    ),
}

csv_lookup = SCRAPERS["csv_lookup"]
# Create instances of scraper factories for different scraper types
SCISCRAPERS: dict[str, SciScraper] = {
    "citations": SciScraper(csv_lookup, STAGERS["citations"]),
    "csv": SciScraper(csv_lookup, None),
    "directory": SciScraper(SCRAPERS["pdf_lookup"], STAGERS["pdf_expanded"]),
    "download": SciScraper(csv_lookup, STAGERS["download"]),
    "fastscore": SciScraper(SCRAPERS["abstract_lookup"], None),
    "images": SciScraper(csv_lookup, STAGERS["images"]),
    "orcid": SciScraper(csv_lookup, STAGERS["authors"]),
    "references": SciScraper(csv_lookup, STAGERS["references"]),
    "wordscore": SciScraper(csv_lookup, STAGERS["abstracts"]),
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
