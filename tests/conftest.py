from __future__ import annotations

import pandas as pd
import pytest
import requests

from src.config import config
from src.docscraper import DocScraper
from src.downloaders import BulkPDFScraper
from src.downloaders import ImagesDownloader
from src.factories import SCISCRAPERS
from src.webscrapers import SemanticWebScraper
from src.webscrapers import WebScrapeResult


# File Path Fixtures
@pytest.fixture()
def mock_dirs():
    return "tests/test_dirs"


@pytest.fixture()
def mock_file_blank():
    with open("tests/test_dirs/test_file") as file_blank:
        return str(file_blank.readlines())


@pytest.fixture()
def mock_file_multiline():
    with open("tests/test_dirs/test_example_2.txt") as file_multiline:
        return str(file_multiline.readlines())


@pytest.fixture()
def test_pdf():
    return "tests/test_dirs/test_pdf_1.pdf"


@pytest.fixture
def mock_dataframe():
    return pd.DataFrame({"col1": [1, 2, 3], "col2": [4, 5, 6]})


@pytest.fixture
def staged_terms():
    return [1, 2, 3]


@pytest.fixture
def staged_tuple():
    return ([1, 2, 3], [4, 5, 6])


@pytest.fixture()
def result_data():
    return {
        "title": "Lorem",
        "pub_date": "01/02/2033",
        "doi": "10.0000",
        "internal_id": "10001",
        "journal_title": "Ipsum Dolor",
        "times_cited": 5,
        "author_list": ["John", "Mark", "Lee"],
        "citations": ["0.0000", "1.0000", "2.0000", "3.0000"],
        "keywords": [""],
        "figures": [""],
        "biblio": "",
        "abstract": "",
    }


@pytest.fixture()
def result_data_as_class():
    return WebScrapeResult(
        title="Lorem",
        pub_date="01/02/2033",
        doi="10.0000",
        internal_id="10001",
        journal_title="Ipsum Dolor",
        times_cited=5,
        author_list=["John", "Mark", "Lee"],
        citations=["0.0000", "1.0000", "2.0000", "3.0000"],
        keywords=[""],
        figures=[""],
        biblio="",
        abstract="",
    )


@pytest.fixture()
def docscraper_summary():
    return DocScraper(config.target_words, config.bycatch_words, False)


@pytest.fixture()
def docscraper_pdf():
    return DocScraper(config.target_words, config.bycatch_words, True)


@pytest.fixture()
def scraper():
    return SemanticWebScraper("https://httpstat.us/200", sleep_val=1.0)


@pytest.fixture()
def fetch_scraper():
    return SCISCRAPERS["wordscore"]


@pytest.fixture()
def faulty_scraper():
    return SemanticWebScraper(url="https://httpstat.us/404", sleep_val=1.0)


@pytest.fixture()
def mock_bulkpdfscraper(mock_dirs):
    return BulkPDFScraper(
        config.downloader_url,
        sleep_val=0.5,
        export_dir=mock_dirs,
    )


@pytest.fixture()
def img_downloader():
    return ImagesDownloader(config.downloader_url, sleep_val=0.5)


# will override the requests.Response returned from requests.get
class MockFetchResponse:
    # mock json() method always returns a specific testing dictionary
    @staticmethod
    def obtain():
        return {"mock_key": "mock_response"}


@pytest.fixture
def mock_response(monkeypatch: pytest.MonkeyPatch):
    """Requests.get() mocked to return {'mock_key':'mock_response'}."""

    def mock_get(*args, **kwargs):
        return MockFetchResponse()

    monkeypatch.setattr(requests, "get", mock_get)


@pytest.fixture
def mock_metadata():
    return {
        "Title": "A test title",
        "doi": "10.1038/s41586-020-0315-z",
        "pdf2doi_identifier": "10.1038/s41586-020-0315-z",
        "arxiv": "arXiv:1905.12345",
    }


@pytest.fixture
def mock_csv():
    return [
        {
            "title": "Fake News and Misinformation",
            "doi": "0.1000/12345",
            "times_cited": 1,
            "authors": "Darius Lettsgetham",
            "listed_data": "['pub.10001', 'pub.10002', 'pub.10003']",
        },
        {
            "title": "Prosocial Eurythmics",
            "doi": "10.1000/23456",
            "times_cited": 0,
            "authors": "Anne Elon-Ux",
            "listed_data": "['pub.10004', 'pub.10005']",
        },
        {
            "title": "Gamification on Social Media",
            "doi": "10.1000/34567",
            "times_cited": None,
            "authors": "I. Ron Butterfly",
            "authors_id": 28252,
            "uni_id": "0600055000200019000",
        },
    ]
