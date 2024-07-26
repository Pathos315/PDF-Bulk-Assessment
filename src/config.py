"""config contains the dataclass describing the overall configurations and a method to read the config.

Returns
-------
    a function that reads the dataclass as a JSON object.

"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np

FilePath = str | Path
UTF = "utf-8"


@dataclass
class ScrapeConfig:
    """
    A dataclass containing the overall configurations.

    Variables
    --------
    prog : str
        For future argparse support. The name of the program, "sciscraper".
    description : str
        For future argparse support. A description of the program.
    export_dir : str
        A filepath into which exports will be stored.
    source_dir : str
        A filepath from which pdfs are read in the case of pdf scraping.
    log_dir : str
        A filepath into which logs are stored, if enabled.
    dimensions_ai_dataset_url : str
        A link to dimensions.ai, to which the `DimensionsScraper` makes
        requests queries.
    semantic_scholar_url_stub : str
        The protocol and domain name of Semantic Scholar, to which some aspects
        of the program will make requests queries.
    \n
    downloader_url : str
        \n
        "The Dwarves tell no tale; \n
        but even as mithril was the foundation of their wealth, \n
        so also it was their destruction: \n
        they delved too greedily and too deep, \n
        and disturbed that from which they fled..."
        \n
    \n
    source_file : str
        The .csv file to be assessed, almost always via `DimensionsScraper`.
    target_words : str
        The .txt file of target words for `DocScraper`.
    bycatch_words : str
        The .txt file containing the words that `DocScraper` will consider
        bycatch, i.e. words that suggest the Doc is not a match.
    sleep_interval : float
        The default time between web requests.

    """

    prog: str
    description: str = field(repr=False)
    source_file: str
    source_dir: str
    dimensions_ai_dataset_url: str
    google_scholar_url: str
    semantic_scholar_url: str
    citation_crosscite_url: str
    orcid_url: str
    abstract_getting_url: str
    downloader_url: str = field(repr=False)
    export_dir: str
    target_words: str
    bycatch_words: str
    sleep_interval: float
    profiling_path: str
    today: str = date.today().strftime("%y%m%d")


def read_config(config_file: str) -> ScrapeConfig:
    """
    Takes a .json file and returns a ScrapeConfig object.

    Parameters
    ----------
    config_file : str
        The path to the .json file, which contains the configs.

    Returns
    -------
    ScrapeConfig
        A dataclass containing the overall configurations
    """
    with open(config_file, encoding=UTF) as file:
        data = json.load(file)
    return ScrapeConfig(**data)


config: ScrapeConfig = read_config("config_setup.json")

DIMENSIONS_AI_MAPPING: dict[str, str] = {
    "title": "title",
    "pub_date": "pub_date",
    "doi": "doi",
    "internal_id": "id",
    "journal_title": "journal_title",
    "times_cited": "times_cited",
    "author_list": "author_list",
    "citations": "cited_dimensions_ids",
    "keywords": "mesh_terms",
}

SEMANTIC_SCHOLAR_MAPPING: dict[str, str] = {
    "title": "title",
    "pub_date": "publicationDate",
    "doi": "externalIds",
    "internal_id": "paperId",
    "journal_title": "journal",
    "times_cited": "citationCount",
    "keywords": "fieldsOfStudy",
    "author_list": "authors",
    "abstract": "abstract",
}

KEY_TYPE_PAIRINGS: dict[str, Any] = {
    "doi_from_pdf": "string",
    "title": "string",
    "doi": "string",
    "internal_id": "string",
    "times_cited": np.int16,
    "matching_terms": np.int16,
    "bycatch_terms": np.int16,
    "total_word_count": np.int16,
    "wordscore": np.float16,
    "abstract": "string",
    "biblio": "string",
    "journal_title": "string",
    "downloader": "string",
    "filepath": "string",
    "paper_parentheticals": "string",
}
