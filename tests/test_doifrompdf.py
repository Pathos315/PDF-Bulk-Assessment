from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from src.doifrompdf import (
    DOIFromPDFResult,
    doi_from_pdf,
    extract_metadata,
    find_identifier_by_googling_first_n_characters_in_pdf,
    find_identifier_in_google_search,
    find_identifier_in_metadata,
    find_identifier_in_pdf_info,
    find_identifier_in_text,
    validate_identifier,
)


@pytest.fixture
def mock_pdf_file():
    return Path("/path/to/mock.pdf")


@pytest.fixture
def mock_metadata():
    return {
        "Title": "Test Paper",
        "doi": "10.1234/test.doi",
        "pdf2doi_identifier": "10.5678/pdf2doi.test",
    }


def test_doi_from_pdf_success(mock_pdf_file):
    with patch("src.doifrompdf.extract_metadata") as mock_extract:
        mock_extract.return_value = {"doi": "10.1234/test.doi"}
        result = doi_from_pdf(mock_pdf_file, "Test preprint")
        assert isinstance(result, DOIFromPDFResult)
        assert result.identifier == "10.1234/test.doi"
        assert result.identifier_type == "doi"


@pytest.mark.xfail
def test_doi_from_pdf_not_found(mock_pdf_file):
    with patch("src.doifrompdf.extract_metadata") as mock_extract:
        mock_extract.return_value = {}
        result = doi_from_pdf(mock_pdf_file, "Test preprint")
        assert result is None


@pytest.mark.xfail
def test_find_identifier_in_metadata(mock_metadata):
    result = find_identifier_in_metadata(mock_metadata)
    assert isinstance(result, DOIFromPDFResult)
    assert result.identifier == "10.1234/test.doi"
    assert result.identifier_type == "doi"


def test_find_identifier_in_metadata_not_found():
    result = find_identifier_in_metadata({})
    assert result is None


@patch("src.doifrompdf.find_identifier_in_text")
def test_find_identifier_in_pdf_info(mock_find_in_text, mock_metadata):
    mock_find_in_text.return_value = DOIFromPDFResult(
        "10.1234/test.doi", "doi", True
    )
    result = find_identifier_in_pdf_info(mock_metadata)
    assert isinstance(result, DOIFromPDFResult)
    assert result.identifier == "10.1234/test.doi"
    assert result.identifier_type == "doi"


@patch("pdfplumber.open")
def test_extract_metadata(mock_pdf_open, mock_pdf_file):
    mock_pdf = Mock()
    mock_pdf.metadata = {"Title": "Test Paper"}
    mock_pdf_open.return_value.__enter__.return_value = mock_pdf
    result = extract_metadata(mock_pdf_file)
    assert result == {"Title": "Test Paper"}


@pytest.mark.xfail
def test_find_identifier_in_text():
    text = "This paper has DOI: 10.1234/test.doi and arXiv:2101.12345"
    result = find_identifier_in_text(text)
    assert isinstance(result, DOIFromPDFResult)
    assert result.identifier == "10.1234/test.doi"
    assert result.identifier_type == "doi"


@patch("src.doifrompdf.client.get")
def test_validate_identifier_doi(mock_get):
    mock_response = Mock()
    mock_response.text = '{"valid": true}'
    mock_response.raise_for_status.return_value = None
    mock_get.return_value = mock_response
    result = validate_identifier("10.1234/test.doi", "doi")
    assert result == '{"valid": true}'


@patch("src.doifrompdf.feedparse")
def test_validate_identifier_arxiv(mock_feedparse):
    mock_feedparse.return_value = {"entries": [{"title": "Test Paper"}]}
    result = validate_identifier("2101.12345", "arxiv")
    assert result == "{'title': 'Test Paper'}"


@patch("src.doifrompdf.find_identifier_in_google_search")
def test_find_identifier_by_googling(mock_google_search):
    mock_google_search.return_value = DOIFromPDFResult(
        "10.1234/test.doi", "doi", True
    )
    result = find_identifier_by_googling_first_n_characters_in_pdf(
        "Test paper content", num_results=1, num_characters=20
    )
    assert isinstance(result, DOIFromPDFResult)
    assert result.identifier == "10.1234/test.doi"
    assert result.identifier_type == "doi"


@patch("src.doifrompdf.search")
@patch("src.doifrompdf.find_identifier_in_text")
def test_find_identifier_in_google_search(mock_find_in_text, mock_search):
    mock_search.return_value = ["http://example.com/paper"]
    mock_find_in_text.return_value = DOIFromPDFResult(
        "10.1234/test.doi", "doi", True
    )
    result = find_identifier_in_google_search("Test paper title")
    assert isinstance(result, DOIFromPDFResult)
    assert result.identifier == "10.1234/test.doi"
    assert result.identifier_type == "doi"


# Add more tests as needed to cover edge cases and error handling
from src.doifrompdf import find_identifier_in_metadata
from src.scraperesults import DOIFromPDFResult


def test_find_identifier_in_metadata_special_characters():
    # Arrange
    metadata = {
        "Title": "A Paper with a DOI: 10.1000/12345678",
        "Author": "John Doe",
        "Keywords": "Science, Research",
    }

    # Act
    result = find_identifier_in_metadata(metadata)

    # Assert
    assert isinstance(result, DOIFromPDFResult)
    assert result.identifier == "10.1000/12345678"
    assert result.identifier_type == "Title"


def test_find_identifier_in_metadata_with_leading_trailing_whitespace():
    # Arrange
    metadata = {
        "Title": "   A Paper with a DOI: 10.1000/12345678   ",
        "Author": "John Doe",
        "Keywords": "Science, Research",
    }

    # Act
    result = find_identifier_in_metadata(metadata)

    # Assert
    assert isinstance(result, DOIFromPDFResult)
    assert result.identifier.strip() == "10.1000/12345678"
    assert result.identifier_type == "Title"


from src.doifrompdf import find_identifier_in_metadata
from src.scraperesults import DOIFromPDFResult


def test_find_identifier_in_metadata_non_string_values():
    # Arrange
    metadata = {
        "Title": ["A Paper with a DOI: 10.1000/12345678"],
        "Author": 12345,
        "Keywords": ["Science", "Research"],
    }

    # Act
    result = find_identifier_in_metadata(metadata)

    # Assert
    assert isinstance(result, DOIFromPDFResult)
    assert result.identifier == "10.1000/12345678"
    assert result.identifier_type == "Title"


def test_find_identifier_in_metadata_nested_structures():
    # Arrange
    metadata = {
        "Document": {
            "Title": "A Paper with a DOI: 10.1000/12345678",
            "Author": "John Doe",
            "Keywords": "Science, Research",
        }
    }

    # Act
    result = find_identifier_in_metadata(metadata)

    # Assert
    assert isinstance(result, DOIFromPDFResult)
    assert result.identifier == "10.1000/12345678"
    assert result.identifier_type == "Document.Title"


def test_find_identifier_in_metadata_empty_metadata():
    # Arrange
    metadata = {}

    # Act
    result = find_identifier_in_metadata(metadata)

    # Assert
    assert result is None


def test_find_identifier_in_metadata_valid_doi():
    # Arrange
    metadata = {
        "Title": "A Paper with a DOI: 10.1000/12345678",
        "Author": "John Doe",
        "Keywords": "Science, Research",
    }

    # Act
    result = find_identifier_in_metadata(metadata)

    # Assert
    assert isinstance(result, DOIFromPDFResult)
    assert result.identifier == "10.1000/12345678"
    assert result.identifier_type == "Title"


def test_find_identifier_in_metadata_valid_arxiv_id():
    # Arrange
    metadata = {
        "arXiv": "2201.12345",
        "Title": "A Paper with an arXiv ID: 2201.12345",
        "Author": "John Doe",
        "Keywords": "Science, Research",
    }

    # Act
    result = find_identifier_in_metadata(metadata)

    # Assert
    assert isinstance(result, DOIFromPDFResult)
    assert result.identifier == "2201.12345"
    assert result.identifier_type == "arXiv"
