from __future__ import annotations

from typing import TYPE_CHECKING

from src.log import log_debug
from src.serials import clean_any_nested_columns, list_with_na_replacement

if TYPE_CHECKING:
    import pandas as pd


@log_debug
def stage_from_series(
    target: pd.DataFrame, column: str = "abstract"
) -> list[str]:
    """
    stage_from_series reads a `column` of interest from a
    dataframe `target`, which gets copied internally, and then
    returns a list of entries.

    Unavailable entries are filled in with 'N/A'.

    Parameters:
        target(pd.DataFrame): The dataframe to have a `column` isolated.
        column(str): The column of interest to be isolated. Defaults to "abstract".

    Returns:
        list[str]: A list of entries from the provided column.

    See Also:
        `stage_with_reference`: Takes two columns from a dataframe and returns two lists of entries as a tuple.

    -------
    Example
    -------

        >>> df: pd.DataFrame
                A     B                   C
          'apple'    10     ['a', 'b', 'c']
         'orange'    10          ['d', 'e']
              NaN    10          ['f', 'g']
    ...
        >>> stage_from_series(df, 'A') =
        ['apple','orange','N/A']
    """
    raw_terms: list[str] = list_with_na_replacement(target.copy(), column)
    return clean_any_nested_columns(raw_terms, column)


@log_debug
def stage_with_reference(
    target: pd.DataFrame,
    column_x: str = "citations",
    column_y: str = "title",
) -> tuple[list[str], list[str]]:
    """
    stage_with_reference takes a dataframe `target`, transforms each
    element of list-like data in `column_x` into a row
    of its own, and replicates its index values into a new,
    intermediary dataframe.

    From that intermediary dataframe, it isolates the new row-wise
    `column_x`, and also `column_y`, which serves as the initial
    source titles to which the values in `column_x` were
    initially lists of citations.

    It returns them both as a tuple of lists of strings.

    Args:
        target(pd.DataFrame): The initial dataframe to be expanded upon and referenced.
        column_x(str): The column label containing the list-like data to be transformed.
            Defaults to "citations".
        column_y(str): The column label containing the titles of papers
            to be referenced, to show relation. Defaults to "title".


    Returns:
        tuple[list[str], list[str]]: A tuple of two lists of strings, containing rows from the
        aforementioned columns.

    See Also: `stage_from_series`: Take a dataframe's column and return a list of entries.

    -------
    Example
    -------

    >>> target: pd.DataFrame
                A  B      C
        'apple'  1     ['a', 'b', 'c']
        'orange'  1     ['d', 'e']
        'banana'  1     ['f', 'g']
    ...
    >>> stage_with_reference(df)
        df = target.explode('C')
        df =
                A  B      C
        'apple'  1     'a'
        'apple'  1     'b'
        'apple'  1     'c'
        'orange'  1     'd'
        'orange'  1     'e'
        'banana'  1     'f'
        'banana'  1     'g'
    ...
    >>> df['C'].to_list() = ['a','b','c','d','e','f','g']
    >>> df['A'].to_list() = ['apple','apple','apple','orange','orange','banana','banana',]
    >>> stage_with_reference(df) =
    return (['a','b','c','d','e','f','g'],['apple','apple','apple','orange','orange','banana','banana'])
    """
    data = target.copy().explode(column_x)
    data_col_x = clean_any_nested_columns(
        list_with_na_replacement(data, column_x),
        column_x,
    )
    data_col_y = clean_any_nested_columns(
        list_with_na_replacement(data, column_y),
        column_y,
    )
    return data_col_x, data_col_y
