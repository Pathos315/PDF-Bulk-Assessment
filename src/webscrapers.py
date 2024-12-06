from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from json import loads
from time import sleep
from typing import TYPE_CHECKING, Any, Optional
from urllib.parse import quote_plus, urlencode

from requests import Response, Session

from src.log import logger
from src.scraperesults import WebScrapeResult

if TYPE_CHECKING:
    from collections.abc import Generator


client = Session()


def make_request(
    url: str,
    method: str = "GET",
    sleep_val: float = 1.0,
    **kwargs,
) -> Response:
    sleep(sleep_val)
    response = client.request(method, url, **kwargs)
    logger.debug(
        "response=%r, scraper=%r, status_code=%s",
        response,
        response.status_code,
    )
    if response.status_code != 200:
        raise ValueError("Response not found")
    return response


def get_item(
    data: dict[str, Any],
    key: str,
    subkey: Optional[str | int] = None,
) -> Any:
    """Retrieves an item from the parsed data, with optional subkey."""
    try:
        if not subkey:
            return data[key]
        return data[key][subkey]
    except KeyError:
        logger.warning(f"Key '{key}' not found in response data")


@dataclass
class SemanticWebScraper:
    url: str

    def obtain(
        self,
        search_text: str,
    ) -> Generator[WebScrapeResult, None, None]:
        """
        Fetches and parses articles from Semantic Scholar based on the search_text.
        """
        url = self.format_request(search_text)
        response = make_request(url)
        return self.process_response(search_text, response)

    def process_response(
        self, search_text: str, response: Response
    ) -> Generator[WebScrapeResult, None, None]:
        try:

            data = loads(response.text)

            if not data["data"]:
                raise ValueError("Data not found.")

            paper_data = get_item(data, "data")[0]
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

            yield result

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
class ORCHIDScraper:
    url: str
    namespace: dict[str, str] = field(
        default_factory=lambda: {
            "es": "http://www.orcid.org/ns/expanded-search"
        }
    )

    def obtain(self, search_terms) -> Generator[WebScrapeResult, None, None]:
        full_url = self.format_request(search_terms)
        response = make_request(full_url)
        orcid_id = self.parse_xml_response(response.text)
        extended_response = self.get_extended_response(orcid_id)
        yield from self.parse_orcid_json(extended_response)

    def get_extended_response(self, orcid_id: str) -> str:
        extended_page_querystring = {
            "offset": "0",
            "sort": "date",
            "sortAsc": "false",
            "pageSize": "75",
        }
        extended_url = f"https://orcid.org/{orcid_id}/worksExtendedPage.json"
        extended_response = make_request(
            extended_url,
            params=extended_page_querystring,
        )
        return extended_response.text

    def parse_xml_response(self, response_text: str) -> str:
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
        self, group: dict[Any, Any]
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

            result = WebScrapeResult(
                title=title,
                pub_date=pub_date,
                doi=doi,
                internal_id=internal_id,
                journal_title=journal_title,
                times_cited=0,
                author_list=author_list,
                citations=[""],
                keywords=[""],
                figures=[""],
                biblio="",
                abstract="",
            )
            yield result
