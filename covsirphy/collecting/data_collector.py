#!/usr/bin/env python
# -*- coding: utf-8 -*-

from covsirphy.util.term import Term
from covsirphy.util.filer import Filer


class DataCollector(Term):
    """Class for collecting data for the specified location.

    Args:
        directory (str or pathlib.Path): directory to save downloaded datasets
        update_interval (int or None): update interval of downloading dataset or None (avoid downloading)
        basename_dict (dict[str, str]): basename of downloaded CSV files,
            "covid19dh": COVID-19 Data Hub (default: covid19dh.csv),
            "owid": Our World In Data (default: ourworldindata.csv),
            "google: COVID-19 Open Data by Google Cloud Platform (default: google_cloud_platform.csv),
            "japan": COVID-19 Dataset in Japan (default: covid_japan.csv).
        verbose (int): level of verbosity when downloading

    Note:
        If @update_interval (not None) hours have passed since the last update of downloaded datasets,
        the dawnloaded datasets will be updated automatically.
        When we do not use datasets of remote servers, set @update_interval as None.

    Note:
        If @verbose is 0, no description will be shown.
        If @verbose is 1 or larger, URL and database name will be shown.
        If @verbose is 2, detailed citation list will be show, if available.
    """
    # Default file titles of the downloaded datasets
    TITLE_DICT = {
        "covid19dh": "covid19dh",
        "owid": "ourworldindata",
        "google": "google_cloud_platform",
        "japan": "covid_japan",
    }

    def __init__(self, directory="input", update_interval=12, basename_dict=None, verbose=1):
        # Filenames to save remote datasets
        filer = Filer(directory=directory, prefix=None, suffix=None, numbering=None)
        self._file_dict = {
            k: filer.csv(title=(basename_dict or {}).get(k, v)) for (k, v) in self.TITLE_DICT.items()}
        # Update interval of local files to save remote datasets
        self._update_interval = self._ensure_natural_int(
            update_interval, name="update_interval", include_zero=True, none_ok=True)
        # Verbosity
        self._verbose = self._ensure_natural_int(verbose, name="verbose", include_zero=True)

    def collect(self, geo=None, variables=None, data=None):
        """Collect necessary data from remote server and local data.

        Args:
            geo (tuple(list[str] or tuple(str) or str)): location names defined in covsirphy.Geography class
            data (_type_, optional): _description_. Defaults to None.
        """
        pass
