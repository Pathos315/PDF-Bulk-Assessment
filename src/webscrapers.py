from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from json import loads
from time import sleep
from typing import TYPE_CHECKING, Any, Dict, List
from typing_extensions import deprecated
from urllib.parse import urlencode

from requests import Response, Session
from selectolax.parser import HTMLParser, Node

from src.config import DIMENSIONS_AI_MAPPING, SEMANTIC_SCHOLAR_MAPPING, config
from src.log import logger

if TYPE_CHECKING:
    from collections.abc import Generator

    from numpy import _SupportsItem


client = Session()


@dataclass(frozen=True)
class WebScrapeResult:
    """Represents a result from a scrape to be passed back to the dataframe."""

    title: str
    pub_date: str
    doi: str
    internal_id: str | None
    journal_title: str | None
    times_cited: int | None
    author_list: list[str] = field(default_factory=list)
    citations: list[str] = field(default_factory=list)
    references: list[str] = field(default_factory=list)
    keywords: list[str] | None = field(default_factory=list)
    figures: list[str] | None = field(default_factory=list)
    biblio: str | None = None
    abstract: str | None = None


@dataclass
class WebScraper(ABC):
    """Abstract representation of a webscraper dataclass."""

    url: str
    sleep_val: float

    @abstractmethod
    def obtain(
        self, search_text: str
    ) -> Generator[WebScrapeResult, None, None] | None:
        """
        obtain takes the requested identifier string, `search_text`
        normally a digital object identifier (DOI), or similar,
        and requests it from the provided citational dataset,
        which returns bibliographic data on the paper(s) requested.

        Parameters
        ----------
        search_text : str
            the pubid or digital object identifier
            (i.e. DOI) of the paper in question

        Returns
        -------
        WebScrapeResult
            a dataclass containing the requested information,
            which gets sent back to a dataframe.
        """

    def get_item(
        self, data: dict[str, Any], key: str, subkey: str | None = None
    ) -> Any:
        """Retrieves an item from the parsed data, with optional subkey."""
        try:
            if subkey:
                return data[key][subkey]
            return data[key]
        except KeyError:
            logger.warning(f"Key '{key}' not found in response data")
            return None


@dataclass
class SemanticWebScraper(WebScraper):
    url: str
    sleep_val: float

    def obtain(
        self,
        search_text: str,
    ) -> Generator[WebScrapeResult, None, None] | None:
        """
        Fetches and parses articles from Semantic Scholar based on the search_text.
        """
        logger.debug("Searching for: %s", search_text)
        sleep(self.sleep_val)
        fields: str = (
            "url,paperId,year,authors,externalIds,\
            title,publicationDate,abstract,citationCount,\
            journal,fieldsOfStudy,citations,references"
        )
        url = f"{self.url}search/match?query={search_text}&fields={fields}"

        try:
            response = client.get(url)
            logger.debug("Response status: %s", response.status_code)

            if not response.ok:
                logger.error(
                    "An error occurred for %s. Status: %s, Response: %s",
                    search_text,
                    response.status_code,
                    response.text,
                )
                return None

            data = loads(response.text)

            if not data.get("data"):
                logger.warning("No results found for query: %s", search_text)
                return None

            paper_data = data["data"][0]
            citation_titles: list[str] = [
                citation["title"]
                for citation in paper_data.get("citations", [])
            ]
            reference_titles: list[str] = [
                reference["title"]
                for reference in paper_data.get("references", [])
            ]

            result = WebScrapeResult(
                title=paper_data.get("title", "N/A"),
                pub_date=paper_data.get("publicationDate"),
                doi=paper_data.get("externalIds", {}).get("DOI", "N/A"),
                internal_id=paper_data.get("paperId", "N/A"),
                abstract=paper_data.get("abstract", "N/A"),
                times_cited=paper_data.get("citationCount", 0),
                citations=citation_titles,
                references=reference_titles,
                journal_title=paper_data.get("journal", {}).get("name"),
                keywords=paper_data.get("fieldsOfStudy", []),
                author_list=self.get_authors(paper_data),
            )

            yield result  # type: ignore

        except json.JSONDecodeError:
            logger.error(
                "Failed to parse JSON response for query: %s", search_text
            )
        except Exception as e:
            logger.exception(
                "Unexpected error occurred for query: %s. Error: %s",
                search_text,
                str(e),
            )

        return None

    def get_authors(self, paper_data: dict) -> list[str]:
        """
        Extracts author names from the paper data.
        """
        return [
            author.get("name", "N/A")
            for author in paper_data.get("authors", [])
        ]


@dataclass
@deprecated("Use Semantic Scraper instead")
class GoogleScholarScraper(WebScraper):
    """
    Representation of a webscraper that makes requests to Google Scholar.
    """

    start_year: int
    end_year: int
    publication_type: str
    num_articles: int

    def obtain(
        self, search_text: str
    ) -> Generator[WebScrapeResult, None, None]:
        """
        Fetches and parses articles from Google Scholar based on the search_text and
        pre-defined criteria such as publication_type, date range, etc.
        Yields WebScrapeResult objects as they are parsed.
        """
        publication_type_mapping = {
            "all": "",
            "j": "source:journals",
            "b": "source:books",
            "c": "source:conferences",
        }
        publication_type = publication_type_mapping.get(
            self.publication_type, ""
        )

        results_yielded = 0
        num_pages = (self.num_articles - 1) // 10 + 1

        for page in range(num_pages):
            start = page * 10
            params = {
                "q": search_text,
                "hl": "en",
                "as_ylo": self.start_year,
                "as_yhi": self.end_year,
                "as_vis": "0",
                "start": start,
            }
            if publication_type:
                params["as_sdt"] = "0,5"
                params["as_vis"] = "1"
                params[publication_type] = ""

            url = f"{self.url}?{urlencode(params)}"
            print(url)

            sleep(self.sleep_val)
            response = client.get(url)

            if not response.ok:
                logger.error(
                    f"An error occurred for {search_text} on page {page + 1}"
                )
                continue

            html = HTMLParser(response.text)
            print(html)
            article_results = html.css("div")
            print(article_results)

            for result in article_results:
                print(result.text())
                parsed_result = self._parse_result(result, search_text)
                print(parsed_result)
                if parsed_result:
                    yield parsed_result
                    results_yielded += 1
                    if results_yielded >= self.num_articles:
                        return

            if results_yielded >= self.num_articles:
                return

    def _parse_result(self, result: Node, search_text: str) -> WebScrapeResult:
        title_node = result.css_first("h3")
        print(title_node)
        title = title_node.text(strip=True) if title_node else "N/A"

        url_node = title_node.css_first("a") if title_node else None
        article_url: str = str(
            url_node.attributes.get("href", "N/A") if url_node else "N/A"
        )

        abstract = self._find_element_text(result, class_name="gs_rs")

        citation_node = result.css_first(".gs_fl a")
        times_cited = (
            self._extract_number(citation_node.text()) if citation_node else 0
        )

        pub_info = result.css_first(".gs_a")
        publication_year = (
            self._extract_year(pub_info.text()) if pub_info else "N/A"
        )
        journal_title = (
            self._extract_journal(pub_info.text()) if pub_info else None
        )

        return WebScrapeResult(
            title=title,
            pub_date=publication_year,
            doi=article_url,
            internal_id=self.publication_type,
            abstract=abstract,
            times_cited=times_cited,
            journal_title=journal_title,
            keywords=[search_text],
        )

    def _find_element_text(self, result: Node, class_name: str) -> str:
        element = result.css_first(f".{class_name}")
        return element.text(strip=True) if element else ""

    @staticmethod
    def _extract_number(text: str) -> int:
        number_match = re.search(r"\d+", text)
        return int(number_match.group()) if number_match else 0

    @staticmethod
    def _extract_year(text: str) -> str:
        year_match = re.search(r"\b\d{4}\b", text)
        return year_match.group() if year_match else "N/A"

    @staticmethod
    def _extract_journal(text: str) -> str | None:
        parts = text.split("-")
        return parts[1].strip() if len(parts) > 1 else None


@dataclass
@deprecated("Use SemanticWebScraper instead.")
class DimensionsScraper(WebScraper):
    """
    Representation of a webscraper that makes requests to dimensions.ai.
    """

    query_subset_citations: bool = False

    def obtain(self, search_text: str) -> WebScrapeResult | None:
        querystring = self.create_querystring(search_text)
        response = self.get_docs(querystring)
        logger.debug(
            "search_text=%s, scraper=%r, status_code=%s",
            search_text,
            self,
            response.status_code,
        )

        if response.status_code != 200:
            return None

        data = self.enrich_response(response)
        return WebScrapeResult(**data)

    def get_docs(self, querystring: dict[Any, Any]) -> Response:
        sleep(self.sleep_val)
        return client.get(self.url, params=querystring)

    def enrich_response(self, response: Response) -> dict[str, Any]:
        api_mapping = DIMENSIONS_AI_MAPPING

        getters: dict[str, tuple[str, WebScraper]] = {
            "biblio": (
                "doi",
                CitationScraper(
                    config.citation_crosscite_url,
                    sleep_val=0.1,
                ),
            ),
            "abstract": (
                "internal_id",
                OverviewScraper(
                    config.abstract_getting_url,
                    sleep_val=0.1,
                ),
            ),
        }
        response_data = loads(response.text)
        item = self.get_item(response_data, "docs")
        data = {key: item.get(value) for (key, value) in api_mapping.items()}
        for key, getter in getters.items():
            data[key] = self.get_extra_variables(data, *getter)
        return data

    def get_extra_variables(
        self, data: dict[str, Any], query: str, getter: WebScraper
    ) -> Generator[WebScrapeResult, None, None] | None:
        """get_extra_variables queries
        subsidiary scrapers to get
        additional data

        Parameters
        ----------
        data : dict
            the dict from the initial scrape
        getter : WebScraper
            the subsidiary scraper that
            will obtain additional information
        query : str
            the existing `data` to be queried.
        """
        try:
            return getter.obtain(data[query])
        except (KeyError, TypeError) as e:
            logger.error(
                "func_repr=%r, query=%s, error=%s, action_undertaken=%s",
                getter,
                query,
                e,
                "Returning None",
            )
            return None

    def create_querystring(self, search_text: str) -> dict[str, str]:
        return (
            {"or_subset_publication_citations": search_text}
            if self.query_subset_citations
            else {
                "search_mode": "content",
                "search_text": search_text,
                "search_type": "kws",
                "search_field": (
                    "doi" if search_text.startswith("10.") else "text_search"
                ),
            }
        )


@deprecated("Unnecessary")
class Style(Enum):
    """An enum that represents
    different academic writing styles.

    Parameters
    ----------
    Style : Enum
        A given academic writing style
    """

    APA = "apa"
    MLA = "modern-language-association"
    CHI = "chicago-fullnote-bibliography"


@dataclass
@deprecated("Unnecessary")
class CitationScraper(WebScraper):
    """
    CitationsScraper is a webscraper made exclusively for generating citations
    for requested papers.

    Attributes
    --------
    style : Style
        An Enum denoting a specific kind of writing style.
        Defaults to "apa".
    lang : str
        A string denoting which language will be requested.
        Defaults to "en-US".
    """

    style: Style = Style.APA
    lang: str = "en-US"

    def obtain(self, search_text: str) -> str | None:  # type: ignore[override]
        querystring = self.create_querystring(search_text)
        response = client.get(self.url, params=querystring)
        logger.debug(
            "search_text=%s, scraper=%r, status_code=%s",
            search_text,
            self,
            response.status_code,
        )
        return response.text if response.status_code == 200 else None

    def create_querystring(self, search_text: str) -> dict[str, Any]:
        return {
            "doi": search_text,
            "style": self.style.value,
            "lang": self.lang,
        }


@dataclass
@deprecated("Unnecessary")
class OverviewScraper(WebScraper):
    """
    OverviewScraper is a webscraper made exclusively
    for getting abstracts to papers
    within the dimensions.ai website.
    """

    def obtain(self, search_text: str) -> str | None:  # type: ignore[override]
        url = f"{self.url}/{search_text}/abstract.json"
        response = client.get(url)
        logger.debug(
            "search_text=%s, scraper=%r, status_code=%s",
            search_text,
            self,
            response.status_code,
        )

        return (
            self.get_item(
                response.json(),
                "docs",
                "abstract",
            )
            if response.status_code == 200
            else None
        )


# TODO: Figure out how to make requests to SemanticScholar without causing 429 Errors.
# Possibility of a post request according to their API?
@dataclass
@deprecated("Unnecessary")
class SemanticFigureScraper(WebScraper):
    """Scraper that queries
    semanticscholar.org for graphs and charts
    from the paper in question.
    """

    def obtain(self, search_text: str) -> list[str | None] | None:  # type: ignore[override]
        paper_url = self.find_paper_url(search_text)
        if paper_url is None:
            return None
        response = client.get(paper_url)
        logger.debug(
            "paper_url=%s, scraper=%r, status_code=%s",
            paper_url,
            self,
            response.status_code,
        )
        return (
            self.parse_html_tree(response.text)
            if response.status_code == 200
            else None
        )

    def find_paper_url(self, search_text: str) -> str | None:
        paper_searching_url = self.url + urlencode(
            {"query": search_text, "fields": "url", "limit": 1}
        )
        logger.info(paper_searching_url)
        paper_searching_response = client.get(paper_searching_url)
        logger.info(paper_searching_response)
        paper_info: dict[str, Any] = loads(paper_searching_response.text)
        logger.info(paper_info)
        try:
            paper_url: str | None = paper_info["data"][0]["url"]
            logger.debug("\n%s\n", paper_url)
        except IndexError as e:
            logger.error(
                "error=%s, action_undertaken=%s",
                e,
                "Returning None",
            )
            paper_url = None
        return paper_url

    def parse_html_tree(self, response_text: str) -> list[Any] | None:
        tree = HTMLParser(response_text)
        images: list[Node] = tree.css(
            "li.figure-list__figure > a > figure > div > img"
        )
        return (
            [image.attributes.get("src") for image in images]
            if images
            else None
        )
