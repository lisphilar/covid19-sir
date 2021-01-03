#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pathlib import Path
import pandas as pd


def save_dataframe(df, filename, index=True):
    """
    Save dataframe as a CSV file.

    Args:
        df (pd.DataFrame): the dataframe
        filename (str or None): CSV filename
        index (bool): if True, include index column.

    Note:
        If @filename is None or OSError was raised, the dataframe will not be saved.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(
            f"@df should be a pandas.DataFrame, but {type(df)} was applied.")
    try:
        filepath = Path(filename)
        df.to_csv(filepath, index=index)
    except (TypeError, OSError):
        pass
