#!/usr/bin/env python
# -*- coding: utf-8 -*-

from itertools import zip_longest
import country_converter as coco
import pandas as pd
from covsirphy.util.term import Term
from covsirphy.util.filer import Filer
from covsirphy.collecting.geography import Geography


class DataCollector(Term):
    """Class for collecting data for the specified location.

    Args:
        layers (list[str] or None): list of layers of geographic information or None (["ISO3", "Province", "City"])
        country (str or None): layer name of countries or None (countries are not included in the layers)
        update_interval (int): update interval of downloading dataset
        verbose (int): level of verbosity when downloading

    Note:
        Country level data specified with @country will be stored with ISO3 codes.

    Note:
        If @update_interval hours have passed since the last update of downloaded datasets,
        the dawnloaded datasets will be updated automatically.

    Note:
        If @verbose is 0, no description will be shown.
        If @verbose is 1 or larger, URL and database name will be shown.
        If @verbose is 2, detailed citation list will be show, if available.
    """
    _ID = "Location_ID"

    def __init__(self, layers=None, country="ISO3", update_interval=12, verbose=1):
        self._update_interval = self._ensure_natural_int(update_interval, name="update_interval", include_zero=True)
        self._verbose = self._ensure_natural_int(verbose, name="verbose", include_zero=True)
        # Countries will be specified with ISO3 codes and this requires conversion
        self._country = None if country is None else str(country)
        # Location data
        self._layers = self._ensure_list(target=layers or [self._country, self.PROVINCE, self.CITY], name="layers")
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
        # Convert date type
        df.rename(columns={date: self.DATE}, inplace=True)
        df[self.DATE] = pd.to_datetime(df[self.DATE], **kwargs)
        # Convert country names to ISO3 codes
        if self._country is not None:
            df.loc[:, self._country] = df[self._country].apply(self._to_iso3)
        # Locations
        loc_df = self._loc_df.merge(df, how="left", on=self._layers, ignore_index=True)
        loc_df.loc[loc_df[self._ID].isna(), self._ID] = f"id{len(loc_df)}-" + \
            loc_df[loc_df[self._ID].isna()].index.astype("str")
        self._loc_df = loc_df.reset_index()[[self._ID, *self._layers]].fillna(self.NA)
        # Records
        columns = [self._ID, self.DATE, *self._ensure_list(target=variables or None, name=variables)]
        df = df.merge(self._loc_df, how="left", on=self._layers)
        df = df.loc[:, columns].reset_index(drop=True)
        self._rec_df = pd.concat([self._rec_df, df], axis=0, ignore_index=True)
        # Citations
        citation_dict = {col: self._citation_dict.get(col, []) + (citations or "my own dataset") for col in variables}
        self._citation_dict.update(citation_dict)
        return self

    @staticmethod
    def _to_iso3(name):
        """Convert country name to ISO3 codes.

        Args:
            name (str or list[str]): country name(s)

        Returns:
            str or list[str]: ISO3 code(s) or as-is when not found

        Note:
            "UK" will be converted to "GBR".
        """
        names = ["GBR" if elem == "UK" else elem for elem in ([name] if isinstance(name, str) else name)]
        return coco.convert(names, to="ISO3", not_found=None)

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
        geo_converted = [geo] if isinstance(geo, str) else (geo or [None]).copy()
        geo_converted += [None] * (len(self._layers) - len(geo_converted))
        if self._country is not None:
            geo_converted = [
                self._to_iso3(info) if layer == self._country else info for (layer, info) in zip(self._layers, geo_converted)]
        # Collect data of the area
        all_df = self.all(variables=variables)
        geography = Geography(layers=self._layers)
        df = geography.filter(data=all_df, geo=geo_converted)
        return df.drop(self._layers, axis=1).groupby(self.DATE).sum().reset_index()

    def auto(self, iso3=None, directory="input", basename_dict=None):
        """Download datasets from remote servers automatically.

        Args:
            iso3 (str or None): ISO3 code of country that must be included or None (all available countries)
            directory (str or pathlib.Path): directory to save downloaded datasets
            basename_dict (dict[str, str]): basename of downloaded CSV files,
                "covid19dh": COVID-19 Data Hub (default: covid19dh.csv),
                "owid": Our World In Data (default: ourworldindata.csv),
                "google: COVID-19 Open Data by Google Cloud Platform (default: google_cloud_platform.csv),
                "japan": COVID-19 Dataset in Japan (default: covid_japan.csv).

        Returns:
            covsirphy.DataCollector: self
        """
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
        self.manual(**self._auto_covid19dh(iso3=iso3, path=file_dict["covid19dh"]))
        # Our World In Data
        # Google Cloud Plat Form
        # Japan dataset via CovsirPhy project
        return self

    def _auto_covid19dh(self, iso3, path):
        """Prepare records of the number of confirmed/fatal/recovered cases, the number of PCR tests and OXCGRT indicators.

        Args:
            iso3 (str or None): ISO3 code of country that must be included or None (all available countries)
            path (str): path of the CSV file to save dataset in local environment

        Returns:
            dict[str, pandas.DataFrame or str or list[str]]]
                - data (pandas.DataFrame):
                    Index
                        reset index
                    Columns
                        - (str): layers defined by DataCollector(layers) argument
                        - date (pandas.Timestamp): observation dates
                        - Confirmed (numpy.int64): the number of confirmed cases
                        - Fatal (numpy.int64): the number of fatal cases
                        - Recovered (numpy.int64): the number of recovered cases
                        - Tests (numpy.int64): the number of PCR tests
                        - School_closing (numpy.int64): one of the OxCGRT indicator
                        - Workplace_closing (numpy.int64): one of the OxCGRT indicator
                        - Cancel_events (numpy.int64): one of the OxCGRT indicator
                        - Gatherings_restrictions (numpy.int64): one of the OxCGRT indicator
                        - Transport_closing (numpy.int64): one of the OxCGRT indicator
                        - Stay_home_restrictions (numpy.int64): one of the OxCGRT indicator
                        - Internal_movement_restrictions (numpy.int64): one of the OxCGRT indicator
                        - International_movement_restrictions (numpy.int64): one of the OxCGRT indicator
                        - Information_campaigns (numpy.int64): one of the OxCGRT indicator
                        - Testing_policy (numpy.int64): one of the OxCGRT indicator
                        - Contact_tracing (numpy.int64): one of the OxCGRT indicator
                        - Stringency_index (numpy.int64): one of the OxCGRT indicator
                - citations (list[str]): citation of COVID-19 Data Hub
        """
        col_dict = {
            "date": self.DATE, "iso_alpha_3": self.ISO3,
            "confirmed": self.C, "deaths": self.F, "recovered": self.R, "tests": self.TESTS, "population": self.N,
        }
        # Get raw data from server
        if iso3 is None:
            url = "https://storage.covid19datahub.io/level/1.csv.zip"
        else:
            url = f"https://storage.covid19datahub.io/country/{iso3}.csv.zip"
        df = pd.read_csv(
            url, header=0, use_cols=list(col_dict.values()), parse_dates="date", date_parser=lambda x: pd.datetime.strptime(x, "%Y-%m-%d"))
        df.rename(columns=col_dict, inplace=True)
        # Perform sequence alignment with layers
        dh_layers = [self.ISO3, "administrative_area_level_2", "administrative_area_level_3"]
        if self._country == self.ISO3:
            defined_lower_layers = self._layers[self._layers.index(self.ISO3) + 1:]
            for (admin_level, layer) in zip_longest([self.PROVINCE, self.CITY], defined_lower_layers):
                if admin_level is None:
                    break
                if layer is None:
                    df = df.loc[df[admin_level].isna()]
                    break
                df.rename(columns={admin_level: layer}, inplace=True)
        else:
            df.rename(columns={admin_level: layer for (admin_level, layer) in zip_longest()}, inplace=True)
        for layer in self._layers:
            df.loc[:, layer] = df[layer].fillna(self.NA)
