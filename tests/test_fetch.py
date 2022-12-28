import logging
from typing import Callable
import numpy as np
import pandas as pd
import pytest
from sciscrape.docscraper import DocScraper
from sciscrape.downloaders import Downloader
from sciscrape.fetch import SciScraper, StagingFetcher, ScrapeFetcher
from sciscrape.factories import SCISCRAPERS, read_factory
from sciscrape.stagers import stage_from_series, stage_with_reference
from sciscrape.change_dir import change_dir
from sciscrape.webscrapers import WebScraper
from sciscrape.config import config
from unittest import mock
from sciscrape.log import logger


@pytest.mark.parametrize(
    ("key"), (("wordscore", "citations", "reference", "download", "images"))
)
def test_read_factory_input(monkeypatch: pytest.MonkeyPatch, key):
    monkeypatch.setattr("builtins.input", lambda _: key)
    output = read_factory()
    assert isinstance(output, SciScraper)
    assert isinstance(output.stager, StagingFetcher)
    assert isinstance(output.scraper, ScrapeFetcher)
    assert isinstance(output.scraper.scraper, (DocScraper, WebScraper, Downloader))


def test_faulty_factory_input(monkeypatch: pytest.MonkeyPatch):
    first = True
    input_calls: int = 0

    def myinp(_):
        nonlocal input_calls, first
        input_calls += 1
        if first:
            first = not first
            return "lisjdfklsdjlkfjdsklfjsd"
        return "citations"

    monkeypatch.setattr("builtins.input", myinp)
    output = read_factory()
    assert input_calls == 2
    assert isinstance(output, SciScraper)


def test_downcast_available_datetimes_validity():
    df = pd.DataFrame({"pub_date": ["2020-01-01", "2020-01-02", "2020-01-03"]})
    assert SciScraper.downcast_available_datetimes(df).dtype == "datetime64[ns]"


def test_null_datetimes_downcasting():
    df = pd.DataFrame({"pub_date": ["", "", ""]})
    assert SciScraper.downcast_available_datetimes(df).isnull().all()


def test_no_datetimes_downcasting():
    with pytest.raises(ValueError):
        df = pd.Series({"pub_date": ["a", "b", "c"]})
        SciScraper.downcast_available_datetimes(df).equals(df)


def test_empty_datetimes_downcasting():
    with pytest.raises(KeyError):
        df = pd.DataFrame()
        SciScraper.downcast_available_datetimes(df)


def test_NaN_datetimes_downcasting():
    # Test NaN values
    df = pd.DataFrame({"pub_date": [np.nan]})
    # TODO: Review values in assert statement
    assert SciScraper.downcast_available_datetimes(df).isnull().all()


def test_none_downcast_available_datetimes():
    df = pd.DataFrame({"pub_date": [None, None, None]})
    assert SciScraper.downcast_available_datetimes(df).isnull().all()


def test_fetch_serializerstrategy(mock_csv, mock_dataframe):
    with mock.patch(
        "sciscrape.fetch.ScrapeFetcher.fetch",
        return_value=mock_dataframe,
        autospec=True,
    ):
        test_sciscraper = SCISCRAPERS["wordscore"]
        assert isinstance(test_sciscraper.scraper.serializer, Callable)
        search_terms = test_sciscraper.scraper.serializer(mock_csv)
        assert isinstance(search_terms, list)
        output_a = test_sciscraper.scraper.fetch(search_terms)
        output_b = test_sciscraper.scraper(mock_csv)
        assert isinstance(output_a, pd.DataFrame)
        assert isinstance(output_b, pd.DataFrame)


def test_fetch(mock_csv):
    with mock.patch(
        "sciscrape.fetch.ScrapeFetcher.__call__",
        return_value=None,
        autospec=True,
    ):
        test_sciscraper = SCISCRAPERS["wordscore"]
        search_terms = test_sciscraper.scraper.serializer(mock_csv)
        assert isinstance(search_terms, list)
        output = test_sciscraper.scraper.fetch(search_terms)
        assert isinstance(output, pd.DataFrame)
        assert output.empty is False
        assert output.dtypes["title"] == "object"
        new_data = test_sciscraper.df_casting(output)
        assert new_data.dtypes["title"] == "string"


def fetch_with_staged_reference(self, staged_terms):
    citations, src_titles = staged_terms
    ref_df: pd.DataFrame = self.fetch(citations, "references")
    dataframe = ref_df.join(pd.Series(src_titles, dtype="string", name="src_titles"))
    return dataframe


def fetch_from_staged_series(self, prior_df, staged_terms):
    df_ext: pd.DataFrame = self.fetch(staged_terms)
    dataframe: pd.DataFrame = prior_df.join(df_ext)
    return dataframe


def test_fetch_with_staged_reference():
    test_sciscraper = SCISCRAPERS["reference"]
    with mock.patch(
        "sciscrape.fetch.StagingFetcher.fetch_with_staged_reference",
        return_value=pd.DataFrame,
    ):
        terms = (
            ["a", "b", "c", "d", "e", "f"],
            ["1", "2", "3", "1", "2", "3"],
        )
        output = test_sciscraper.stager.fetch_with_staged_reference(terms)  # type: ignore
        assert output is not None


def test_fetch_with_null_staged_reference():
    # Test null case
    with mock.patch(
        "sciscrape.fetch.ScrapeFetcher.fetch",
        return_value=None,
        autospec=True,
    ), pytest.raises(AttributeError):
        test_sciscraper = SCISCRAPERS["reference"]
        staged_terms = [None, None]
        fetch_with_staged_reference(test_sciscraper.scraper, staged_terms)


def test_invalid_dataframe_logging(caplog, mock_dataframe):
    with caplog.at_level(logging.INFO, logger="sciscraper"):
        SciScraper.dataframe_logging(mock_dataframe)
        for record in caplog.records:
            assert record.levelname != "CRITICAL"


def test_create_export_name():
    output = SciScraper.create_export_name()
    assert f"{config.today}_sciscraper_" in output


@pytest.mark.xfail
def test_staging_fetcher_raises_value_error_on_invalid_staged_terms(prior_df):
    fetcher = StagingFetcher("invalid", stager=stage_from_series)  # type: ignore
    with pytest.raises(ValueError, match="Staged terms must be lists or tuples."):
        fetcher(prior_df)


@pytest.mark.xfail
def test_staging_fetcher_calls_fetch_from_staged_series_on_list(prior_df, staged_terms):
    fetcher = StagingFetcher(staged_terms, stager=stage_from_series)
    fetcher.fetch_from_staged_series = mock.Mock(return_value=prior_df)
    assert fetcher(prior_df) == prior_df
    fetcher.fetch_from_staged_series.assert_called_once_with(prior_df, staged_terms)


@pytest.mark.xfail
def test_staging_fetcher_calls_fetch_with_staged_reference_on_tuple(
    prior_df, staged_tuple
):
    fetcher = StagingFetcher(staged_tuple, stager=stage_with_reference)
    fetcher.fetch_with_staged_reference = mock.Mock(return_value=prior_df)
    assert fetcher(prior_df) == prior_df
    fetcher.fetch_with_staged_reference.assert_called_once_with(staged_tuple)


import pandas as pd
import random


class TestClass:
    def set_logging(self, debug: bool) -> None:
        logger.setLevel(10) if debug else logger.setLevel(20)

    @classmethod
    def dataframe_logging(cls, dataframe: pd.DataFrame) -> None:
        pass

    @classmethod
    def create_export_name(cls) -> str:
        return "test_50.csv"

    @classmethod
    def export(cls, dataframe: pd.DataFrame, export_dir: str = "export") -> None:
        """Export data to the specified export directory."""
        cls.dataframe_logging(dataframe)
        export_name = cls.create_export_name()
        with change_dir(export_dir):
            logger.info(f"A spreadsheet was exported as {export_name}.")
            dataframe.to_csv(export_name)


def test_export(mock_dataframe):
    with (
        mock.patch("sciscrape.log.logger.info") as mock_logger,
        mock.patch("pandas.DataFrame.to_csv") as mock_to_csv,
        mock.patch(
            "sciscrape.change_dir.change_dir", return_value=None
        ) as mock_change_dir,
    ):
        TestClass.export(mock_dataframe)
        mock_logger.assert_called_once_with(
            f"A spreadsheet was exported as {TestClass.create_export_name()}."
        )
        mock_to_csv.assert_called_once_with(TestClass.create_export_name())


def test_set_logging():
    with mock.patch.object(logger, "setLevel") as mock_set_level:
        instance = TestClass()
        instance.set_logging(True)
        mock_set_level.assert_called_once_with(10)
        instance.set_logging(False)
        mock_set_level.assert_called_with(20)


@pytest.mark.xfail
def test_fetch_with_staged_reference_valid_input():
    staged_terms = (["a", "b", "c"], ["d", "e", "f"])
    scraper = mock.Mock()
    stager = StagingFetcher(scraper, stager=stage_with_reference)
    # TODO: Review values in assert statement
    assert stager.fetch_with_staged_reference(staged_terms) == (
        ["a", "b", "c"],
        ["d", "e", "f"],
    )


@pytest.mark.xfail
def test_fetch_with_staged_reference_no_stager():
    scraper = mock.Mock()
    staged_terms = ([], ["src_title"])
    df = StagingFetcher(stager=None).fetch_with_staged_reference(staged_terms)  # type: ignore
    assert df.empty


def test_fetch_with_staged_reference_tuple_of_lists():
    staged_terms = (["citation"], [])
    scraper = mock.Mock()
    df = StagingFetcher(scraper, stager=None).fetch_with_staged_reference(staged_terms)  # type: ignore
    assert df.empty


def test_fetch_with_staged_reference_empty_tuple():
    staged_terms = ([], [])
    scraper = mock.Mock()
    df = StagingFetcher(scraper, stager=None).fetch_with_staged_reference(staged_terms)  # type: ignore
    assert df.empty


@pytest.mark.xfail
def test_fetch_with_staged_reference_no_scraper():
    staged_terms = (["10.1038/s41586-020-2003-7"], ["Nature"])
    df = StagingFetcher(None).fetch_with_staged_reference(staged_terms)  # type: ignore
    # TODO: Review values in assert statement
    assert not df.empty


@pytest.mark.xfail
def test_fetch_with_staged_reference_NaN_input():
    staged_terms = ([np.nan], [np.nan])
    fetcher = StagingFetcher(stager=None)  # type: ignore
    df = fetcher.fetch_with_staged_reference(staged_terms)
    # TODO: Review values in assert statement
    assert df.empty


@pytest.mark.xfail
def test_stager(mock_dataframe):
    with (
        mock.patch.object(
            TestClass, "fetch_from_staged_series"
        ) as mock_fetch_from_staged_series,
        mock.patch.object(
            TestClass, "fetch_with_staged_reference"
        ) as mock_fetch_with_staged_reference,
    ):
        scraper = mock.Mock()
        stager = StagingFetcher(scraper, stager=stage_with_reference)
        result = stager(mock_dataframe)
        mock_fetch_from_staged_series.assert_called_once_with(mock_dataframe)
        assert result == mock_fetch_from_staged_series.return_value

        with pytest.raises(ValueError, match="Staged terms must be lists or tuples."):
            scraper.stager(1)


@pytest.mark.parametrize(("debug", "expected"), ((True, 10), (False, 20)))
def test_set_logging_valid_input(debug, expected):
    # Test valid input
    test_sciscraper = SCISCRAPERS["reference"]
    test_sciscraper.set_logging(debug)
    assert test_sciscraper.logger.level == expected
    assert SciScraper.logger.level == expected
    assert SciScraper.logger.getEffectiveLevel() == expected


def test_export_invalid_input():
    test_sciscraper = SCISCRAPERS["reference"]
    with (
        pytest.raises(AttributeError),
        change_dir(config.export_dir),
    ):
        test_sciscraper.export(None)  # type: ignore