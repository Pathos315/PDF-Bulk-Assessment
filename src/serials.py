from __future__ import annotations

import pandas as pd

from src.config import UTF, FilePath
from src.log import log_debug

from pathlib import Path


def serialize_from_txt(target: FilePath) -> list[str]:
    """
    Reads a text file, `target`, and returns a list of its contents, after stripping and lowercasing each word.

    :param FilePath target: The target text file, always as a pathname
    :return: A list of words from the provided text file, `target`
    :rtype: list[str]
    """
    with open(target, encoding=UTF) as iowrapper:
        return [word.strip().lower() for word in iowrapper]


@log_debug
def serialize_from_csv(target: FilePath, column: str = "title") -> list[str]:
    """
    Reads a .csv file of papers, `target`, identifies a specific column of interest `column`, and then returns a list of entries.

    :param FilePath target: The target .csv file, always as a pathname
    :param str column: The specific column of interest. Defaults to "doi".

    :rtype list:
    :returns: A list of entries from the provided column, `column`.

    """
    data: pd.DataFrame = pd.read_csv(
        target, skip_blank_lines=True, usecols=[column]
    )
    data_list = list_with_na_replacement(data, column)
    return clean_any_nested_columns(data_list, column)


@log_debug
def serialize_from_directory(
    target: FilePath, suffix: str = "pdf"
) -> list[Path]:
    """
    serialize_directory takes a directory `target` and returns a list[str]
    of its contents to be scraped.

    :param FilePath target:  The target directory, always as a pathname.
    :param str suffix: The file extensions of interest. Defaults to "pdf".

    :rtype list:
    :returns: A list of files from the provided directory, `target` that adhere to the requested format `suffix`.
    """
    return list(Path(target).rglob(f"*.{suffix}"))


def clean_any_nested_columns(data_list: list[str], column: str) -> list[str]:
    """
    This function takes a list of strings and a column name and returns a list of cleaned strings.
    The function first filters out any strings that do not start with a curly brace,
    then it extracts any strings that start with a curly brace and attempts to evaluate them as Python code.
    If the evaluation is successful, the function retrieves the value of the specified column from the resulting object,
    and adds it to the list of cleaned strings.
    Finally, the function returns the combined list of cleaned strings.

    :param list data_list: A list of strings to be cleaned.
    :param str column: The name of the column to be extracted from the nested objects.
    :rtype list:
    :return: A list of cleaned strings.
    """
    initial_terms = [term for term in data_list if not term.startswith("{")]
    nested_terms = [
        eval(term).get(column, "")
        for term in data_list
        if term.startswith("{")
    ]
    return initial_terms + nested_terms


def list_with_na_replacement(
    dataframe: pd.DataFrame,
    column_name: str,
    _replacement_fill: str = "N/A",
) -> list[str]:
    """
    This function takes a pandas dataframe and a column name, and returns a list of values from the specified column,
    with missing values replaced with a specified string.

    :param pd.DataFrame dataframe: The pandas dataframe containing the data.
    :param str column_name: The name of the column to extract from the dataframe.
    :param str _replacement_fill: The string to use to replace missing values. Defaults to "N/A".

    :rtype list:
    :returns: A list of values from the specified column, with missing values replaced.

    """
    return dataframe[column_name].fillna(_replacement_fill).to_list()
