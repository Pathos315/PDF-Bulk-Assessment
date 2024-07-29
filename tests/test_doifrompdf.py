import pytest
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path


CONTENT = "Hello world!"
TEST_DOI = "10.1234/foo.bar"
TEST_ARXIV = "12345678"


@pytest.fixture(scope="session")
def invalid_file(tmp_path):
    d: Path = tmp_path / "path"
    d.mkdir()
    p = d / "hello.pdf"
    p.write_text(CONTENT)
    assert p.read_text() == CONTENT
    return p


@pytest.fixture(scope="session")
def doi_file(tmp_path):
    d: Path = tmp_path / "path"
    d.mkdir()
    p = d / "doi.pdf"
    p.write_text(TEST_DOI)
    assert p.read_text() == TEST_DOI
    return p


@pytest.fixture(scope="session")
def arxiv_file(tmp_path):
    d: Path = tmp_path / "path"
    d.mkdir()
    p = d / "arxiv.pdf"
    p.write_text(TEST_ARXIV)
    assert p.read_text() == TEST_ARXIV
    return p
