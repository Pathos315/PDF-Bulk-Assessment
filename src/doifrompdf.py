from __future__ import annotations

from pathlib import Path
from typing import Any

import pdfplumber
from feedparser import FeedParserDict
from feedparser import (
    parse as feedparse,
)  # type: ignore[import-untyped, unused-ignore]
from googlesearch import search  # type: ignore[import-untyped, unused-ignore]

from src.config import FilePath, config
from src.doi_regex import IDENTIFIER_PATTERNS, extract_identifier
from src.log import logger
from src.scraperesults import DOIFromPDFResult
from src.webscrapers import client


def doi_from_pdf(file: FilePath, preprint: str) -> DOIFromPDFResult | None:
    """
    Extracts a DOI from a PDF file using a set of heuristics.

    :param FilePath file: The path to the PDF file.
    :param str preprint: A preprint identifier, such as a manuscript ID, or arXiv ID.

    :returns: A data class containing the extracted DOI, if any, and its type.
    """
    metadata: dict[Any, Any] = extract_metadata(file)
    title: str = metadata.get("Title", Path(file).stem)
    handlers: dict[Any, Any] = {
        find_identifier_in_metadata: (metadata,),
        find_identifier_in_pdf_info: (metadata,),
        find_identifier_in_text: (title, IDENTIFIER_PATTERNS, True),
        find_identifier_in_text: (preprint, IDENTIFIER_PATTERNS),
        find_identifier_by_googling_first_n_characters_in_pdf: (preprint,),
    }
    handler_comprehension = (
        handler(*args) for handler, args in handlers.items()
    )
    filtered_handler: filter[Any] = filter(None, handler_comprehension)
    return next(filtered_handler, None)


def find_identifier_in_metadata(
    metadata: dict[str, Any],
) -> DOIFromPDFResult | None:
    """
    Searches for a valid identifier (e.g., DOI, arXiv ID) within the given metadata dictionary.
    Prioritizes certain keys for a more efficient search.

    :param dict metadata: A dictionary containing metadata key-value pairs.

    :rtype: DOIFromPDFResult | None
    :returns: A data class containing the identifier and its type if a valid identifier is found; otherwise, None.
    """

    logger.info(
        "Method #1: Looking for a valid identifier in the document metadata..."
    )

    for key in config.priority_keys:
        if not (initial_result := metadata.get(key)):
            continue
        logger.info(f"Identifier found using Method #1 {initial_result}")
        return DOIFromPDFResult(identifier=initial_result, identifier_type=key)
    logger.info(
        "Could not find a valid identifier in the most likely metadata keys."
    )
    return None


def find_identifier_in_pdf_info(
    metadata: dict[str, str],
) -> DOIFromPDFResult | None:
    """
    Try to find a valid DOI in the values of the 'document information' dictionary.

    :param dict metadata: A dictionary containing metadata key-value pairs.
    :rtype: DOIFromPDFResult | None
    :returns: A dictionary with identifier and other info (see above)
    """
    values_to_search = (
        value for key, value in metadata.items() if key != "/wps-journaldoi"
    )

    for value in values_to_search:
        result = find_identifier_in_text(value)
        if not (result and result.identifier):
            continue
        logger.info(
            f"A valid {result.identifier_type} was found in the document info labelled '{value}'."
        )
        return result
    logger.info(f"No valid identifier found in the metadata key: '{value}'.")
    return None


def extract_metadata(file: FilePath) -> dict[Any, Any]:
    """
    Extracts metadata from a PDF file using the pdfplumber library.

    :param FilePath file: The path to the PDF file.

    :rtype: dict[Any, Any]
    :returns: A dictionary containing the metadata key-value pairs.
    """
    with pdfplumber.open(file) as pdf:
        metadata: dict[Any, Any] = pdf.metadata
        logger.debug(metadata)
    return metadata


def find_identifier_in_text(
    text: str,
    title_search: bool = False,
) -> DOIFromPDFResult | None:
    """
    Searches for a valid identifier (e.g., DOI or arXiv ID) within a text.

    :param str text: Text to be analyzed.
    :param bool title_search: Flag indicating whether the search is for a title.

    :rtype: DOIFromPDFResult | None
    :returns: A data class containing the identifier and its type if a valid identifier is found; otherwise, None.

    """
    search_type = "title" if title_search else "text"

    for id_type, pattern in IDENTIFIER_PATTERNS.items():
        logger.info(
            f"Searching for a valid {id_type.upper()} in the document {search_type}..."
        )
        for sub_pattern in pattern:
            matches = sub_pattern.findall(text)
        if not matches:
            logger.info(
                f"No valid {id_type.upper()} found in the document {search_type}."
            )
        identifier = matches[0]
        logger.debug(f"Potential {id_type.upper()} found: {identifier}")

        validation = validate_identifier(identifier, id_type)
        identifier = (
            extract_identifier(identifier) if id_type == "doi" else identifier
        )

    return DOIFromPDFResult(identifier, id_type, validation)


def validate_identifier(identifier: str, id_type: str) -> str | None:
    """
    Validate an identifier by querying appropriate URLs based on the identifier type.

    :params str identifier: The identifier to be validated.
    :params str id_type: Type of the identifier ('arxiv' or 'doi').
    :rtype: Any
    :returns: A string representation of the validation result, or None if validation fails.
    """
    try:
        return (
            validate_arxiv(identifier)
            if id_type == "arxiv"
            else validate_doi(identifier)
        )
    except Exception as e:
        logger.error(
            "Some error occured within the function validate_doi_web: %s" % e
        )
        return None


def validate_doi(identifier: str) -> str:
    url = f"http://dx.doi.org/{identifier}"
    headers = {"accept": "application/citeproc+json"}
    response = client.get(url, headers=headers)
    response.raise_for_status()
    return response.text


def validate_arxiv(identifier: str) -> str:
    url = f"http://export.arxiv.org/api/query?search_query=id:{identifier}"
    result: FeedParserDict = feedparse(url)
    item = str(result["entries"][0])
    return item


def find_identifier_by_googling_first_n_characters_in_pdf(
    text: str,
    num_results: int = 3,
    num_characters: int = 50,
) -> DOIFromPDFResult | None:
    """
    Perform a Google search using the first N characters of the text and
    find an identifier in the search results.

    :param str text: The text in which to search for an identifier.
    :param int num_results: The number of search results to consider, defaults to 3.
    :param int num_characters: The maximum number of characters to consider, defaults to 50.
    """
    logger.info(
        f"Method #4: Trying to do a google search with the first {num_characters} characters of this pdf file..."
    )

    if not text.strip():
        logger.error("No meaningful text could be extracted from this file.")
        return None

    trimmed_text = text[:num_characters].lower()

    logger.info(
        f"Performing google search with first {num_results} characters of the text..."
    )
    return find_identifier_in_google_search(trimmed_text, num_results)


def find_identifier_in_google_search(
    query: str, num_results: int = 3, max_length_display: int = 100
) -> DOIFromPDFResult | None:
    """Perform a Google search using the query and find an identifier in the search results.

    :param str query: The search query.
    :param int num_results:  The number of search results to consider.
    :param int max_length_display: The maximum number of characters to consider. Defaults to 100.

    :rtype: DOIFromPDFResult | None
    :returns: The result object containing the identifier information, if found; otherwise, None.
    """
    query_to_display: str = (
        query[0:max_length_display]
        if len(query) > max_length_display
        else query
    )
    logger.info(
        f"Performing google search with key {query_to_display}, considering first {num_results} results..."
    )
    for url in search(query, stop=num_results):
        result = find_identifier_in_text(url)

        if not (result and result.identifier):
            continue
        logger.info(
            f"A valid {result.identifier_type} was found in the search URL."
        )
        return result

    logger.info("No valid identifier found in the search results.")
    return None
