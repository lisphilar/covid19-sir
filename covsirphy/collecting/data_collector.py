#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
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
    _ID = "Location_ID"
    # Default file titles of the downloaded datasets
    TITLE_DICT = {
        "covid19dh": "covid19dh",
        "owid": "ourworldindata",
        "google": "google_cloud_platform",
        "japan": "covid_japan",
    }
    # Default values of layers
    LAYERS = [Term.COUNTRY, Term.PROVINCE, Term.CITY]
    # Default variables of the collected data
    VARIABLES = [
        # The number of cases
        Term.C, Term.CI, Term.F, Term.R, Term.N,
        # PCR tests
        Term.TESTS, Term.T_DIFF,
        # Vaccinations
        Term.VAC, Term.VAC_BOOSTERS, Term.V_ONCE, Term.V_FULL,
    ]

    def __init__(self, directory="input", update_interval=12, basename_dict=None, verbose=1):
        # Filenames to save remote datasets
        filer = Filer(directory=directory, prefix=None, suffix=None, numbering=None)
        self._file_dict = {
            k: filer.csv(title=(basename_dict or {}).get(k, v))["path_or_buf"] for (k, v) in self.TITLE_DICT.items()}
        # Update interval of local files to save remote datasets
        self._update_interval = self._ensure_natural_int(
            update_interval, name="update_interval", include_zero=True, none_ok=True)
        # Verbosity
        self._verbose = self._ensure_natural_int(verbose, name="verbose", include_zero=True)
        # Location data
        self._loc_df = pd.DataFrame(index=[self._ID])
        # All available data
        self._all_df = pd.DataFrame(index=[self._ID, self.DATE])
        # Citations
        self._citations = []

    @property
    def citations(self):
        """
        str: citation list of the secondary data sources
        """
        return self._citations

    def collect(self, layers=None, geo=None, variables=None, data=None):
        """Collect necessary data from remote server and local data.

        Args:
            layers (list): list of layers of geographic information
            geo (tuple(list[str] or tuple(str) or str)): location names defined in covsirphy.Geography class
            variables (list[str] or None): list of variables to collect or None (default values)
            data (pandas.DataFrame): local dataset or None (un-available)

        Note:
            Default layers are defined by covsirphy.DataCollector.LAYERS (class variable).

        Note:
            Please refer to covsirphy.Geography.filter() regarding @geo argument.

        Note:
            Default variables are defined by covsirphy.DataCollector.VARIABLES (class variable).
        """
        pass
