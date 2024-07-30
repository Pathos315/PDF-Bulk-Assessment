from __future__ import annotations

from contextlib import suppress
import json
import xml.etree.ElementTree as ET
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from json import loads
from time import sleep
from typing import TYPE_CHECKING, Any, Optional
from typing_extensions import deprecated
from urllib.parse import urlencode, quote_plus

from requests import Response, Session
from selectolax.parser import HTMLParser, Node

from src.config import DIMENSIONS_AI_MAPPING, config
from src.log import logger

if TYPE_CHECKING:
    from collections.abc import Generator


client = Session()


@dataclass(frozen=True)
class WebScrapeResult:
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

    @abstractmethod
    def format_request(self, **kwargs) -> Any:
        pass

    @abstractmethod
    def process_response(self, response: Any, **kwargs) -> Any:
        pass

    def make_request(
        self,
        url: str,
        method: str = "GET",
        **kwargs,
    ) -> Response | None:
        sleep(self.sleep_val)
        response = client.request(method, url, **kwargs)
        logger.debug(
            "response=%r, scraper=%r, status_code=%s",
            response,
            self,
            response.status_code,
        )
        return response if response.ok else None

    def get_item(
        self,
        data: dict[str, Any],
        key: str,
        subkey: Optional[str | int] = None,
    ) -> Any:
        """Retrieves an item from the parsed data, with optional subkey."""
        try:
            return data[key][subkey] if subkey else data[key]
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
        url = self.format_request(search_text)
        response = self.make_request(url=url)
        return (
            self.process_response(search_text, response) if response else None
        )

    def process_response(
        self, search_text: str, response: Response
    ) -> Generator[WebScrapeResult, None, None] | None:
        try:

            data = loads(response.text)

            if not data["data"]:
                return None

            paper_data = self.get_item(data, "data", 0)
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

    def format_request(self, search_text: str) -> str:
        fields: str = (
            "url,paperId,year,authors,externalIds,title,publicationDate,abstract,citationCount,journal,fieldsOfStudy,citations,references"
        )
        url = f"{self.url}search/match?query={search_text}&fields={fields}"
        return url

    def get_authors(self, paper_data: dict) -> list[str]:
        """
        Extracts author names from the paper data.
        """
        return [
            author.get("name", "N/A")
            for author in paper_data.get("authors", [])
        ]


@dataclass
class ORCHIDScraper(WebScraper):
    namespace: dict[str, str] = field(
        default_factory=lambda: {
            "es": "http://www.orcid.org/ns/expanded-search"
        }
    )

    def obtain(
        self, search_terms
    ) -> Generator[WebScrapeResult, None, None] | None:
        full_url = self.format_request(search_terms)
        response = self.make_request(full_url)
        if not response:
            return None
        orcid_id = self.parse_xml_response(response.text)
        if orcid_id is None:
            return
        extended_response = self.get_extended_response(orcid_id)
        if extended_response is None:
            return
        yield from self.parse_orcid_json(extended_response)

    def get_extended_response(self, orcid_id: str) -> str | None:
        extended_page_querystring = {
            "offset": "0",
            "sort": "date",
            "sortAsc": "false",
            "pageSize": "75",
        }
        extended_url = f"https://orcid.org/{orcid_id}/worksExtendedPage.json"
        extended_response = self.make_request(
            extended_url, params=extended_page_querystring
        )
        return extended_response.text if extended_response.ok else None  # type: ignore

    def parse_xml_response(self, response_text: str):
        root = ET.fromstring(response_text).find(
            "es:expanded-result", self.namespace
        )
        return root.find("es:orcid-id", self.namespace).text  # type: ignore

    def format_request(self, search_terms: str) -> str:
        querystring = {
            "q": '{!edismax qf="given-and-family-names^50.0 family-name^10.0 given-names^10.0 credit-name^10.0 other-names^5.0 text^1.0" pf="given-and-family-names^50.0" bq="current-institution-affiliation-name:[* TO *]^100.0 past-institution-affiliation-name:[* TO *]^70" mm=1}'
            + search_terms,
            "start": "0",
            "rows": "1",
        }
        encoded_params = urlencode(querystring, quote_via=quote_plus)
        return f"{self.url}?{encoded_params}"

    def parse_orcid_json(
        self, json_data: str
    ) -> Generator[WebScrapeResult, None, None]:
        data = loads(json_data)

        for group in data["groups"]:
            yield from self.process_response(group)

    def process_response(
        self, group: dict
    ) -> Generator[WebScrapeResult, None, None]:
        for work in group["works"]:
            title = work["title"]["value"]
            try:
                pub_date = work["publicationDate"]["year"]
            except TypeError:
                pub_date = "N/A"
            doi = next(
                (
                    id["externalIdentifierId"]["value"]
                    for id in work["workExternalIdentifiers"]
                    if id["externalIdentifierType"]["value"] == "doi"
                ),
                "N/A",
            )
            internal_id = work["putCode"]["value"]
            journal_title = (
                work["journalTitle"]["value"]
                if work["journalTitle"]
                else "N/A"
            )

            author_list = [
                contributor["creditName"]["content"]
                for contributor in work["contributorsGroupedByOrcid"]
                if contributor["creditName"]
            ]

            yield WebScrapeResult(
                title=title,
                pub_date=pub_date,
                doi=doi,
                internal_id=internal_id,
                journal_title=journal_title,
                times_cited=0,
                author_list=author_list,
                citations=[],
                keywords=None,
                figures=None,
                biblio="",
                abstract="",
            )
