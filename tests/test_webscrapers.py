import pytest
from unittest.mock import Mock, patch
from dataclasses import asdict
import json
import xml.etree.ElementTree as ET
from urllib.parse import urlencode, quote_plus

# Import the classes to be tested
from src.webscrapers import (
    WebScrapeResult,
    WebScraper,
    SemanticWebScraper,
    ORCHIDScraper,
)

# Fixtures


@pytest.fixture
def mock_client():
    with patch("src.webscrapers.client") as mock:
        yield mock


@pytest.fixture
def mock_sleep():
    with patch("time.sleep") as mock:
        yield mock


# WebScrapeResult tests


def test_webscraperesult_default_values():
    result = WebScrapeResult()
    assert result.title == "N/A"
    assert result.pub_date == "N/A"
    assert result.doi == "N/A"
    assert result.internal_id == "N/A"
    assert result.journal_title == "N/A"
    assert result.times_cited == 0
    assert result.author_list == []
    assert result.citations == []
    assert result.references == []
    assert result.keywords == []
    assert result.figures == []
    assert result.biblio == ""
    assert result.abstract == ""


def test_webscraperesult_custom_values():
    custom_result = WebScrapeResult(
        title="Test Title",
        pub_date="2023-01-01",
        doi="10.1234/test",
        internal_id="TEST123",
        journal_title="Test Journal",
        times_cited=10,
        author_list=["Author 1", "Author 2"],
        citations=["Citation 1"],
        references=["Reference 1"],
        keywords=["keyword1", "keyword2"],
        figures=["figure1.jpg"],
        biblio="Test bibliography",
        abstract="Test abstract",
    )
    assert custom_result.title == "Test Title"
    assert custom_result.pub_date == "2023-01-01"
    assert custom_result.doi == "10.1234/test"
    assert custom_result.internal_id == "TEST123"
    assert custom_result.journal_title == "Test Journal"
    assert custom_result.times_cited == 10
    assert custom_result.author_list == ["Author 1", "Author 2"]
    assert custom_result.citations == ["Citation 1"]
    assert custom_result.references == ["Reference 1"]
    assert custom_result.keywords == ["keyword1", "keyword2"]
    assert custom_result.figures == ["figure1.jpg"]
    assert custom_result.biblio == "Test bibliography"
    assert custom_result.abstract == "Test abstract"


# WebScraper tests


class DummyWebScraper(WebScraper):
    def obtain(self, search_text):
        return None

    def process_response(self, search_text, response):
        pass

    def format_request(self, search_text):
        pass


@pytest.mark.skip
def test_webscraper_init():
    scraper = DummyWebScraper(url="https://example.com", sleep_val=1.0)
    assert scraper.url == "https://example.com"
    assert scraper.sleep_val == 1.0


@pytest.mark.skip
def test_webscraper_get_item():
    scraper = DummyWebScraper(url="https://example.com", sleep_val=1.0)
    data = {"key1": {"subkey": "value"}, "key2": "value2"}
    assert scraper.get_item(data, "key1", "subkey") == "value"
    assert scraper.get_item(data, "key2") == "value2"
    assert scraper.get_item(data, "non_existent_key") is None


# SemanticWebScraper tests


@pytest.fixture
def semantic_scraper():
    return SemanticWebScraper(
        url="https://api.semanticscholar.org/graph/v1/paper/", sleep_val=1.0
    )


def test_semantic_scraper_init(semantic_scraper):
    assert (
        semantic_scraper.url
        == "https://api.semanticscholar.org/graph/v1/paper/"
    )
    assert semantic_scraper.sleep_val == 1.0


@pytest.mark.skip
@pytest.mark.parametrize(
    "search_text, status_code, json_data, expected_result",
    [
        (
            "test query",
            200,
            {
                "data": [
                    {
                        "title": "Test Paper",
                        "publicationDate": "2023-01-01",
                        "externalIds": {"DOI": "10.1234/test"},
                        "paperId": "TEST123",
                        "abstract": "Test abstract",
                        "citationCount": 5,
                        "citations": [{"title": "Citation 1"}],
                        "references": [{"title": "Reference 1"}],
                        "journal": {"name": "Test Journal"},
                        "fieldsOfStudy": ["Computer Science"],
                        "authors": [{"name": "John Doe"}],
                    }
                ]
            },
            True,
        ),
        ("no results", 200, {"data": []}, False),
        ("error", 404, {}, False),
    ],
)
def test_semantic_scraper_obtain(
    semantic_scraper,
    mock_client,
    mock_sleep,
    search_text,
    status_code,
    json_data,
    expected_result,
):
    mock_response = Mock()
    mock_response.ok = status_code == 200
    mock_response.status_code = status_code
    mock_response.text = json.dumps(json_data)
    mock_client.get.return_value = mock_response

    result = list(semantic_scraper.obtain(search_text) or [])

    assert bool(result) == expected_result
    if expected_result:
        assert isinstance(result[0], WebScrapeResult)
        assert result[0].title == "Test Paper"
        assert result[0].pub_date == "2023-01-01"
        assert result[0].doi == "10.1234/test"
        assert result[0].internal_id == "TEST123"
        assert result[0].abstract == "Test abstract"
        assert result[0].times_cited == 5
        assert result[0].citations == ["Citation 1"]
        assert result[0].references == ["Reference 1"]
        assert result[0].journal_title == "Test Journal"
        assert result[0].keywords == ["Computer Science"]
        assert result[0].author_list == ["John Doe"]


def test_semantic_scraper_get_authors(semantic_scraper):
    paper_data = {"authors": [{"name": "John Doe"}, {"name": "Jane Smith"}]}
    authors = semantic_scraper.get_authors(paper_data)
    assert authors == ["John Doe", "Jane Smith"]


def test_semantic_scraper_get_authors_empty(semantic_scraper):
    paper_data = {"authors": []}
    authors = semantic_scraper.get_authors(paper_data)
    assert authors == []


def test_semantic_scraper_get_authors_missing_name(semantic_scraper):
    paper_data = {"authors": [{"name": "John Doe"}, {}]}
    authors = semantic_scraper.get_authors(paper_data)
    assert authors == ["John Doe", "N/A"]


# ORCHIDScraper tests


@pytest.fixture
def orchid_scraper():
    return ORCHIDScraper(
        url="https://pub.orcid.org/v3.0/expanded-search/", sleep_val=1.0
    )


def test_orchid_scraper_init(orchid_scraper):
    assert orchid_scraper.url == "https://pub.orcid.org/v3.0/expanded-search/"
    assert orchid_scraper.sleep_val == 1.0
    assert orchid_scraper.namespace == {
        "es": "http://www.orcid.org/ns/expanded-search"
    }


@pytest.mark.parametrize(
    "xml_response, expected_orcid_id",
    [
        (
            '<response><es:expanded-result xmlns:es="http://www.orcid.org/ns/expanded-search"><es:orcid-id>0000-0001-2345-6789</es:orcid-id></es:expanded-result></response>',
            "0000-0001-2345-6789",
        ),
        (
            '<response><es:expanded-result xmlns:es="http://www.orcid.org/ns/expanded-search"><es:orcid-id></es:orcid-id></es:expanded-result></response>',
            None,
        ),
    ],
)
def test_orchid_scraper_parse_xml_response(
    orchid_scraper, xml_response, expected_orcid_id
):
    orcid_id = orchid_scraper.parse_xml_response(xml_response)
    assert orcid_id == expected_orcid_id


@pytest.mark.skip
def test_orchid_scraper_get_extended_response(orchid_scraper, mock_client):
    mock_response = Mock()
    mock_response.ok = True
    mock_response.text = '{"test": "data"}'
    mock_client.get.return_value = mock_response

    result = orchid_scraper.get_extended_response("0000-0001-2345-6789")
    assert result == '{"test": "data"}'


@pytest.mark.skip
def test_orchid_scraper_get_extended_response_error(
    orchid_scraper, mock_client
):
    mock_response = Mock()
    mock_response.ok = False
    mock_client.get.return_value = mock_response

    result = orchid_scraper.get_extended_response("0000-0001-2345-6789")
    assert result is None


def test_orchid_scraper_parse_orcid_json(orchid_scraper):
    json_data = """
    {
        "groups": [
            {
                "works": [
                    {
                        "title": {"value": "Test Paper"},
                        "publicationDate": {"year": "2023"},
                        "workExternalIdentifiers": [{"externalIdentifierType": {"value": "doi"}, "externalIdentifierId": {"value": "10.1234/test"}}],
                        "putCode": {"value": "TEST123"},
                        "journalTitle": {"value": "Test Journal"},
                        "contributorsGroupedByOrcid": [{"creditName": {"content": "John Doe"}}]
                    }
                ]
            }
        ]
    }
    """
    results = list(orchid_scraper.parse_orcid_json(json_data))
    assert len(results) == 1
    assert isinstance(results[0], WebScrapeResult)
    assert results[0].title == "Test Paper"
    assert results[0].pub_date == "2023"
    assert results[0].doi == "10.1234/test"
    assert results[0].internal_id == "TEST123"
    assert results[0].journal_title == "Test Journal"
    assert results[0].author_list == ["John Doe"]


@pytest.mark.skip
def test_orchid_scraper_parse_single_orcid_entry(orchid_scraper):
    group = {
        "works": [
            {
                "title": {"value": "Test Paper"},
                "publicationDate": {"year": "2023"},
                "workExternalIdentifiers": [
                    {
                        "externalIdentifierType": {"value": "doi"},
                        "externalIdentifierId": {"value": "10.1234/test"},
                    }
                ],
                "putCode": {"value": "TEST123"},
                "journalTitle": {"value": "Test Journal"},
                "contributorsGroupedByOrcid": [
                    {"creditName": {"content": "John Doe"}}
                ],
            }
        ]
    }
    results = list(orchid_scraper.parse_single_orcid_entry(group))
    assert len(results) == 1
    assert isinstance(results[0], WebScrapeResult)
    assert results[0].title == "Test Paper"
    assert results[0].pub_date == "2023"
    assert results[0].doi == "10.1234/test"
    assert results[0].internal_id == "TEST123"
    assert results[0].journal_title == "Test Journal"
    assert results[0].author_list == ["John Doe"]


@pytest.mark.skip
def test_orchid_scraper_parse_single_orcid_entry_missing_data(orchid_scraper):
    group = {
        "works": [
            {
                "title": {"value": "Test Paper"},
                "publicationDate": None,
                "workExternalIdentifiers": [],
                "putCode": {"value": "TEST123"},
                "journalTitle": None,
                "contributorsGroupedByOrcid": [],
            }
        ]
    }
    results = list(orchid_scraper.parse_single_orcid_entry(group))
    assert len(results) == 1
    assert isinstance(results[0], WebScrapeResult)
    assert results[0].title == "Test Paper"
    assert results[0].pub_date == "N/A"
    assert results[0].doi == "N/A"
    assert results[0].internal_id == "TEST123"
    assert results[0].journal_title == "N/A"
    assert results[0].author_list == []


@pytest.mark.skip
@pytest.mark.parametrize(
    "search_terms, mock_responses, expected_results",
    [
        (
            "John Doe",
            [
                Mock(
                    status_code=200,
                    text='<response><es:expanded-result xmlns:es="http://www.orcid.org/ns/expanded-search"><es:orcid-id>0000-0001-2345-6789</es:orcid-id></es:expanded-result></response>',
                ),
                Mock(
                    ok=True,
                    text='{"groups": [{"works": [{"title": {"value": "Test Paper"}, "publicationDate": {"year": "2023"}, "workExternalIdentifiers": [{"externalIdentifierType": {"value": "doi"}, "externalIdentifierId": {"value": "10.1234/test"}}], "putCode": {"value": "TEST123"}, "journalTitle": {"value": "Test Journal"}, "contributorsGroupedByOrcid": [{"creditName": {"content": "John Doe"}}]}]}]}',
                ),
            ],
            1,
        ),
        (
            "Jane Smith",
            [
                Mock(status_code=404),
            ],
            0,
        ),
        (
            "Error Case",
            [
                Mock(
                    status_code=200,
                    text='<response><es:expanded-result xmlns:es="http://www.orcid.org/ns/expanded-search"><es:orcid-id>0000-0001-2345-6789</es:orcid-id></es:expanded-result></response>',
                ),
                Mock(ok=False),
            ],
            0,
        ),
    ],
)
def test_orchid_scraper_obtain(
    orchid_scraper,
    mock_client,
    mock_sleep,
    search_terms,
    mock_responses,
    expected_results,
):
    mock_client.get.side_effect = mock_responses

    results = list(orchid_scraper.obtain(search_terms) or [])
    assert len(results) == expected_results

    if expected_results > 0:
        assert isinstance(results[0], WebScrapeResult)
        assert results[0].title == "Test Paper"
        assert results[0].pub_date == "2023"
        assert results[0].doi == "10.1234/test"
        assert results[0].internal_id == "TEST123"
        assert results[0].journal_title == "Test Journal"
        assert results[0].author_list == ["John Doe"]


# Error handling tests
