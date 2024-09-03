from dataclasses import dataclass, field, fields
from abc import ABC
from typing import Any


@dataclass(frozen=True)
class ScrapeResult(ABC): ...


@dataclass(frozen=True)
class WebScrapeResult(ScrapeResult):
    """Represents a result from a scrape to be passed back to the dataframe."""

    title: str = "N/A"
    pub_date: str = "N/A"
    doi: str = "N/A"
    internal_id: str = "N/A"
    journal_title: str = "N/A"
    times_cited: int = 0
    author_list: list[str] = field(default_factory=list)
    citations: list[str] = field(default_factory=list)
    references: list[str] = field(default_factory=list)
    keywords: list[str] | None = field(default_factory=list)
    figures: list[str] | None = field(default_factory=list)
    biblio: str = ""
    abstract: str = ""


@dataclass(frozen=True)
class DocumentResult(ScrapeResult):
    """DocumentResult contains the WordscoreCalculator\
    scoring relevance, and two lists, each with\
    the three most frequent target and bycatch words respectively.\
    This gets passed back to a pandas dataframe.\
    """

    doi_from_pdf: str = "N/A"
    matching_terms: int = 0
    bycatch_terms: int = 0
    total_word_count: int = 0
    wordscore: float = 0.0
    target_terms_top_3: list[tuple[str, int]] = field(default_factory=list)
    bycatch_terms_top_3: list[tuple[str, int]] = field(default_factory=list)
    paper_parentheticals: list[Any] = field(default_factory=list)


@dataclass(frozen=True)
class DownloadReceipt(ScrapeResult):
    """
    A representation of the receipt describing whether
    or not the download was successful, and,
    if so, where the ensuing file may be found.

    Attributes
    ---------
    downloader : str
        The class name of the downloader
        (e.g. 'BulkPDFDownloader, ImageDownloader')
    success : bool
        If the download was successful or not. Defaults to False.
    filepath : str
        Where the file is located if downloaded. Defaults to 'N/A'.
    """

    downloader: str = "N/A"
    success: bool = False
    filepath: str = "N/A"


@dataclass(frozen=True)
class DOIFromPDFResult(ScrapeResult):
    "A data class containing the extracted identifier, and its type."

    identifier: str = ""
    identifier_type: str = ""
    validation_info: str | bool = True
