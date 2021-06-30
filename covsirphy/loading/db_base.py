#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pathlib import Path
import pandas as pd
from covsirphy.util.term import Term


class _RemoteDatabase(Term):
    """
    Base class to access remote databases.

    Args:
        filename (str): CSV filename to save records
    """
    # Citation
    CITATION = ""
    # Column names and data types
    # {"name in database": ("name defined in Term class", "data type")}
    COL_DICT = {}

    def __init__(self, filename):
        # Filepath to save files
        try:
            self.filepath = Path(filename)
        except TypeError:
            raise TypeError(f"@filename should be a path-like object, but {filename} was applied.")
        self.filepath.parent.mkdir(exist_ok=True, parents=True)
        # List of primary sources
        self.primary_list = []
        # {"name in database": "name defined in Term class"}
        self.col_convert_dict = {k: v[0] for (k, v) in self.COL_DICT.items()}
        # List of column names (defined in Term class)
        self.saved_cols = [line[0] for line in self.COL_DICT.values()]
        # Dictionary of data types (defined in Term class)
        self.dtype_dict = {line[0]: line[1] for line in self.COL_DICT.values()}

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
            return self._ensure_dataframe(self.read(), columns=self.saved_cols)
        # Download dataset from server
        df = self._ensure_dataframe(self.download(verbose=verbose), columns=self.saved_cols)
        df.to_csv(self.filepath, index=False)
        return df

    def read(self):
        """
        Load the dataset with a local file.

        Returns:
            pandas.DataFrame
                Index
                    reset index
                Columns
                    defined by .COL_DICT
        """
        kwargs = {"low_memory": False, "dtype": self.dtype_dict, "header": 0, "usecols": self.saved_cols}
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
                    defined by .COL_DICT
        """
        raise NotImplementedError

    @property
    def primary(self):
        """
        str: the list of primary sources.
        """
        return self.primary_list
