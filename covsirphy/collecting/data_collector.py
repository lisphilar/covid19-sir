#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
from covsirphy.util.term import Term
from covsirphy.util.filer import Filer
from covsirphy.util.geography import Geography


class DataCollector(Term):
    """Class for collecting data for the specified location.

    Args:
        layers (list[str] or None): list of layers of geographic information or None (["Country", "Province", "City"])
    """
    _ID = "Location_ID"

    def __init__(self, layers=None):
        # Location data
        self._layers = self._ensure_list(target=layers or [Term.COUNTRY, Term.PROVINCE, Term.CITY], name="layers")
        self._loc_df = pd.DataFrame(columns=[self._ID, *self._layers])
        # All available data
        self._rec_df = pd.DataFrame(columns=[self._ID, self.DATE])
        # Citations
        self._citation_dict = {}

    def all(self, variables=None):
        """Return all available data.

        Args:
            variables (list[str] or None): list of variables to collect or None (all available variables)

        Returns:
            pandas.DataFrame:
                Index
                    reset index
                Columns
                    - columns defined by covsirphy.DataCollector(layers)
                    - Date (pandas.Timestamp): observation dates
                    - columns defined by @variables
        """
        df = self._loc_df.merge(self._rec_df, how="right", on=self._ID, ignore_index=True).reset_index(drop=True)
        if variables is None:
            return df
        all_variables = df.columns.tolist()
        sel_variables = self._ensure_list(target=variables, candidates=all_variables, name="variables")
        return df.drop(list(set(all_variables) - set(sel_variables)), axis=1)

    def citations(self, variables=None):
        """
        Return citation list of the secondary data sources.

        Args:
            variables (list[str] or None): list of variables to collect or None (all available variables)

        Returns:
            list[str]: citation list
        """
        columns = self._ensure_list(target=variables or None, name="variables")
        return self.flatten([v for (k, v) in self._citation_dict.items() if k in columns], unique=True)

    def manual(self, data, date="Date", variables=None, citations=None, **kwargs):
        """Add data manually.

        Args:
            data (pandas.DataFrame): local dataset or None (un-available)
                Index
                    reset index
                Columns
                    - columns defined by covsirphy.DataCollector(layers)
                    - columns defined by @date
                    - columns defined by @variables
            date (str): column name of date
            variables (list[str] or None): list of variables to add or None (all available columns)
            citations (list[str] or None): citations of the dataset or None (["my own dataset"])
            **kwargs: keyword arguments of pandas.to_datetime() including "dayfirst (bool): whether date format is DD/MM or not"

        Returns:
            covsirphy.DataCollector: self
        """
        df = self._ensure_dataframe(target=data, name="data", columns=[*self._layers, date])
        df.rename(columns={date: self.DATE}, inplace=True)
        df[self.DATE] = pd.to_datetime(df[self.DATE], **kwargs)
        # Locations
        loc_df = self._loc_df.merge(df, how="left", on=self._layers, ignore_index=True)
        loc_df.loc[loc_df[self._ID].isna(), self._ID] = "id" + loc_df[loc_df[self._ID].isna()].index.astype("str")
        self._loc_df = loc_df.reset_index()[[self._ID, *self._layers]]
        # Records
        columns = [self._ID, self.DATE, *self._ensure_list(target=variables or None, name=variables)]
        df = df.merge(self._loc_df, how="left", on=self._layers)
        df = df.loc[:, columns].reset_index(drop=True)
        self._rec_df = pd.concat([self._rec_df, df], axis=0, ignore_index=True)
        # Citations
        citation_dict = {col: self._citation_dict.get(col, []) + (citations or "my own dataset") for col in variables}
        self._citation_dict.update(citation_dict)
        return self

    def collect(self, geo=None, variables=None):
        """Collect necessary data from remote server and local data.

        Args:
            geo (tuple(list[str] or tuple(str) or str)): location names defined in covsirphy.Geography class
            variables (list[str] or None): list of variables to collect or None (all available variables)

        Returns:
            pandas.DataFrame:
                Index
                    reset index
                Columns
                    - Date (pandas.Timestamp): observation dates
                    - columns defined by @variables

        Note:
            Please refer to covsirphy.Geography.filter() regarding @geo argument.

        """
        # Collect data of the area
        all_df = self.all(variables=variables)
        geography = Geography(layers=self._layers)
        df = geography.filter(data=all_df, geo=geo)
        return df.drop(self._layers, axis=1).groupby(self.DATE).sum().reset_index()

    def auto(self, iso3=None, directory="input", update_interval=12, basename_dict=None, verbose=1):
        """Download datasets from remote servers automatically.

        Args:
            iso3 (str or None): ISO3 code of country that must be included or None (all available countries)
            directory (str or pathlib.Path): directory to save downloaded datasets
            update_interval (int or None): update interval of downloading dataset or None (avoid downloading)
            basename_dict (dict[str, str]): basename of downloaded CSV files,
                "covid19dh": COVID-19 Data Hub (default: covid19dh.csv),
                "owid": Our World In Data (default: ourworldindata.csv),
                "google: COVID-19 Open Data by Google Cloud Platform (default: google_cloud_platform.csv),
                "japan": COVID-19 Dataset in Japan (default: covid_japan.csv).
            verbose (int): level of verbosity when downloading

        Returns:
            covsirphy.DataCollector: self

        Note:
            If @update_interval (not None) hours have passed since the last update of downloaded datasets,
            the dawnloaded datasets will be updated automatically.

        Note:
            If @verbose is 0, no description will be shown.
            If @verbose is 1 or larger, URL and database name will be shown.
            If @verbose is 2, detailed citation list will be show, if available.
        """
        self._ensure_natural_int(update_interval, name="update_interval", include_zero=True, none_ok=True)
        self._ensure_natural_int(verbose, name="verbose", include_zero=True)
        # Filenames to save remote datasets
        TITLE_DICT = {
            "covid19dh": "covid19dh",
            "owid": "ourworldindata",
            "google": "google_cloud_platform",
            "japan": "covid_japan",
        }
        filer = Filer(directory=directory, prefix=None, suffix=None, numbering=None)
        file_dict = {
            k: filer.csv(title=(basename_dict or {}).get(k, v))["path_or_buf"] for (k, v) in TITLE_DICT.items()}
        # COVID-19 Data Hub
        self.manual(data=self._auto_covid19dh(iso3=iso3, local_file=file_dict["covid19dh"]), date="date")
        # Our World In Data
        # Google Cloud Plat Form
        # Japan dataset via CovsirPhy project
        return self

    def _auto_covid19dh(**kwargs):
        pass
