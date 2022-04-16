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
    # Default file titles of the downloaded datasets
    TITLE_DICT = {
        "covid19dh": "covid19dh",
        "owid": "ourworldindata",
        "google": "google_cloud_platform",
        "japan": "covid_japan",
    }

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
