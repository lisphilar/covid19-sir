#!/usr/bin/env python
# -*- coding: utf-8 -*-

import contextlib
from pathlib import Path
import pandas as pd
from covsirphy.util.validator import Validator
from covsirphy.util.term import Term


class _RemoteDatabase(Term):
    """
    Base class to access remote databases.

    Args:
        filename (str): CSV filename to save records
        iso3 (str or None): ISO3 code of the country which must be included in the dataset or None (all available countries)
    """
    # Citation
    CITATION = ""
    # Column names and data types
    # {"name in database": "name defined in Term class"}
    COL_DICT = {}

    def __init__(self, filename, iso3):
        # Filepath to save files
        try:
            self.filepath = Path(filename)
        except TypeError as e:
            raise TypeError(f"@filename should be a path-like object, but {filename} was applied.") from e

        self.filepath.parent.mkdir(exist_ok=True, parents=True)
        # Country
        self._iso3 = None if iso3 is None else str(iso3)
        # List of column names (defined in Term class)
        self.saved_cols = list(self.COL_DICT.values())

    def to_dataframe(self, force, verbose):
        """
        Load the dataset and return it as a dataframe.

        Args:
            force (bool): whether download the dataset from server or not when we have saved dataset
            verbose (int): level of verbosity

        Returns:
            pandas.DataFrame
                Index
                    reset index
                Columns
                    defined by the first values of self.COL_DICT.values()
        """
        # Read local file if available and usable
        if not force and self.filepath.exists():
            with contextlib.suppress(ValueError):
                df = self._ensure_dataframe(self.read(), columns=self.saved_cols)
                if self._iso3 is None and df[self.ISO3].nunique() > 1:
                    return df
                if self._iso3 is not None:
                    Validator(df, "raw data").dataframe()
                    if self._value_included(df, value_dict={self.ISO3: self._iso3}):
                        return df
        # Download dataset from server
        df = self.download(verbose=verbose)
        df = df.rename(columns=self.COL_DICT)
        df = self._ensure_dataframe(df, columns=self.saved_cols)
        df.to_csv(self.filepath, index=False)
        return df

    def _value_included(self, target, value_dict):
        """
        Return whether all expected values are included in the columns or not.

        Args:
            target (pandas.DataFrame): the dataframe to ensure
            value_dict (dict[str, object]): dictionary of values which must be included in the column

        Returns:
            bool: whether all values (specified in @value_dict) are included or not
        """
        return all(value in target[col].unique() for (col, value) in value_dict.items()) if set(value_dict.keys()).issubset(target.columns) else False

    @staticmethod
    def _ensure_dataframe(target, name="df", time_index=False, columns=None, empty_ok=True):
        """
        Ensure the dataframe has the columns.

        Args:
            target (pandas.DataFrame): the dataframe to ensure
            name (str): argument name of the dataframe
            time_index (bool): if True, the dataframe must has DatetimeIndex
            columns (list[str] or None): the columns the dataframe must have
            empty_ok (bool): whether give permission to empty dataframe or not

        Returns:
            pandas.DataFrame:
                Index
                    as-is
                Columns:
                    columns specified with @columns or all columns of @target (when @columns is None)
        """
        if not isinstance(target, pd.DataFrame):
            raise TypeError(f"@{name} must be a instance of (pandas.DataFrame).")
        df = target.copy()
        if time_index and (not isinstance(df.index, pd.DatetimeIndex)):
            raise TypeError(f"Index of @{name} must be <pd.DatetimeIndex>.")
        if not empty_ok and target.empty:
            raise ValueError(f"@{name} must not be a empty dataframe.")
        if columns is None:
            return df
        if not set(columns).issubset(df.columns):
            cols_str = ", ".join(col for col in columns if col not in df.columns)
            included = ", ".join(df.columns.tolist())
            s1 = f"Expected columns were not included in {name} with {included}."
            raise KeyError(f"{s1} {cols_str} must be included.")
        return df.loc[:, columns]

    def read(self):
        """
        Load the dataset with a local file.

        Returns:
            pandas.DataFrame
                Index
                    reset index
                Columns
                    defined by .COL_DICT.values()
        """
        kwargs = {"low_memory": False, "header": 0, "usecols": self.saved_cols}
        return pd.read_csv(self.filepath, **kwargs)

    def download(self, verbose):
        """
        Download the dataset from the server and set the list of primary sources.

        Args:
            verbose (int): level of verbosity

        Returns:
            pandas.DataFrame
                Index
                    reset index
                Columns
                    defined by .COL_DICT.values()
        """
        raise NotImplementedError
