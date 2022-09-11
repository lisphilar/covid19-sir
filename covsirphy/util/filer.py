#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pathlib import Path
from covsirphy.util.error import deprecate


class Filer(object):
    """
    Produce filenames and manage files.

    Args:
        directory (list[str] or tuple(str) or str): top directory name(s)
        prefix (str or None): prefix of the filenames or None (no prefix)
        suffix (str or None): suffix of the filenames or None (no suffix)
        numbering (str or None): "001", "01", "1" or None (no numbering)

    Examples:
        >>> import covsirphy as cs
        >>> filer = cs.Filer(directory="output", prefix="jpn", suffix=None, numbering="01")
        >>> filer.png("records")
        {"filename": "<absolute path>/output/jpn_01_records.png"}
        >>> filer.jpg("records")
        {"filename": "<absolute path>/output/jpn_01_records.jpg"}
        >>> filer.json("backup")
        {"filename": "<absolute path>/output/jpn_01_backup.json"}
        >>> filer.geojson("geometry", "driver": "GeoJson")
        {"filename": "<absolute path>/output/jpn_01_geometry.geojson", "driver": "GeoJson"}
        >>> filer.csv("records", index=True)
        {"path_or_buf": "<absolute path>/output/jpn_01_records.csv", index: True}
    """

    def __init__(self, directory, prefix=None, suffix=None, numbering=None):
        directories = [directory] if isinstance(directory, (str, Path)) else directory
        self._dir_path = Path(*directories).resolve()
        self._pre = "" if prefix is None else f"{prefix}_"
        self._suf = "" if suffix is None else f"_{suffix}"
        # Create the directory
        self._dir_path.mkdir(parents=True, exist_ok=True)
        # Numbering
        num_dict = {"001": "{num:0>3}_", "01": "{num:0>2}_", "1": "{num:0>1}_", None: ""}
        if numbering not in num_dict:
            num_str = ", ".join(str(sel) for sel in num_dict)
            raise ValueError(
                f"@numbering should be selected from {num_str}, but {numbering} was applied.")
        self._num_format = num_dict[numbering]
        # Filenames
        self._file_dict = {}

    def _register(self, title, ext):
        """
        Create filename with file title and register it.

        Args:
            title (str): title of the filename, like 'records'
            ext (str): extension of the file, like 'jpg'

        Returns:
            str: absolute filename
        """
        # Create filename
        basename_format = f"{self._pre}{self._num_format}{title}{self._suf}.{ext}"
        basename = basename_format.format(num=len(self._file_dict.get(ext, [])) + 1)
        filename = str(self._dir_path.joinpath(basename))
        # Register the filename
        self._file_dict[ext] = self._file_dict.get(ext, []) + [filename]
        return filename

    def files(self, ext=None):
        """
        List-up filenames.

        Args:
            ext (str or None): file extension or None (all)

        Returns:
            list[str]: list of files
        """
        if ext is None:
            return [file for filenames in self._file_dict.values() for file in filenames]
        return self._file_dict.get(ext, [])

    def png(self, title, **kwargs):
        """
        Create PNG filename and register it.

        Args:
            title (str): title of the filename, like 'records'
            kwargs: keyword arguments to be included in the output

        Returns:
            dict[str, str]: absolute filename (key: 'filename') and kwargs
        """
        filename = self._register(title=title, ext="png")
        return {"filename": filename, **kwargs}

    def jpg(self, title, **kwargs):
        """
        Create JPG filename and register it.

        Args:
            title (str): title of the filename, like 'records'
            kwargs: keyword arguments to be included in the output

        Returns:
            dict[str, str]: absolute filename (key: 'filename') and kwargs
        """
        filename = self._register(title=title, ext="jpg")
        return {"filename": filename, **kwargs}

    def json(self, title, **kwargs):
        """
        Create JSON filename and register it.

        Args:
            title (str): title of the filename, like 'records'
            kwargs: keyword arguments to be included in the output

        Returns:
            dict[str, str]: absolute filename (key: 'filename') and kwargs
        """
        filename = self._register(title=title, ext="json")
        return {"filename": filename, **kwargs}

    def geojson(self, title, **kwargs):
        """
        Create GeoJSON filename and register it.

        Args:
            title (str): title of the filename, like 'records'
            kwargs: keyword arguments to be included in the output

        Returns:
            dict[str, str]: absolute filename (key: 'filename') and kwargs
        """
        filename = self._register(title=title, ext="geojson")
        return {"filename": filename, **kwargs}

    def csv(self, title, **kwargs):
        """
        Create CSV filename and register it.

        Args:
            title (str): title of the filename, like 'records'
            kwargs: keyword arguments to be included in the output

        Returns:
            dict[str, str]: absolute filename (key: 'path_or_buf') and kwargs
        """
        filename = self._register(title=title, ext="csv")
        return {"path_or_buf": filename, **kwargs}


@deprecate("covsirphy.save_dataframe", version="2.17.0-eta")
def save_dataframe(df, filename, index=True):
    """
    Save dataframe as a CSV file.

    Args:
        df (pd.DataFrame): the dataframe
        filename (str or None): CSV filename
        index (bool): if True, include index column.
    """
    df.to_csv(filename, index=index)
