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


class Scraper:
    pdf_lookup = ScrapeFetcher(
        DocScraper(
            Path(config.target_words).resolve(),
            Path(config.bycatch_words).resolve(),
        ),
        serialize_from_directory,
    )
    csv_lookup = ScrapeFetcher(
        SemanticWebScraper(
            config.semantic_scholar_url,
        ),
        serialize_from_csv,
    )
    abstract_lookup = ScrapeFetcher(
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
    )


class Stager:
    abstracts = StagingFetcher(
        DocScraper(
            Path(config.target_words).resolve(),
            Path(config.bycatch_words).resolve(),
            False,
        ),
        stage_from_series,
    )
    authors = StagingFetcher(
        ORCHIDScraper(
            config.orcid_url,
        ),
        partial(
            stage_with_reference,
            column_x="author_list",
        ),
    )
    citations = StagingFetcher(
        SemanticWebScraper(
            config.semantic_scholar_url,
        ),
        stage_with_reference,
    )
    download = StagingFetcher(
        BulkPDFScraper(config.downloader_url),
        partial(stage_from_series, column="doi"),
    )
    images = StagingFetcher(
        ImagesDownloader(url=""),
        partial(stage_with_reference, column_x="figures"),
    )
    pdf_expanded = StagingFetcher(
        SemanticWebScraper(
            config.semantic_scholar_url,
        ),
        partial(stage_from_series, column="doi_from_pdf"),
    )
    references = StagingFetcher(
        SemanticWebScraper(
            config.semantic_scholar_url,
        ),
        partial(stage_with_reference, column_x="references"),
    )


CSV = Scraper.csv_lookup
SCISCRAPERS: dict[str, SciScraper] = {
    "citations": SciScraper(CSV, Stager.citations),
    "csv": SciScraper(CSV, None),
    "directory": SciScraper(Scraper.pdf_lookup, Stager.pdf_expanded),
    "download": SciScraper(CSV, Stager.download),
    "fastscore": SciScraper(Scraper.abstract_lookup, None),
    "images": SciScraper(CSV, Stager.images),
    "orcid": SciScraper(CSV, Stager.authors),
    "references": SciScraper(CSV, Stager.references),
    "wordscore": SciScraper(CSV, Stager.abstracts),
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
