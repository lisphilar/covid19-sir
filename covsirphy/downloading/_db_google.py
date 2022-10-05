#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
from unidecode import unidecode
from covsirphy.util.term import Term
from covsirphy.downloading._db import _DataBase


class _GoogleOpenData(_DataBase):
    """
    Access "COVID-19 Open Data by Google Cloud Platform" server.
    https://github.com/GoogleCloudPlatform/covid-19-open-data
    https://goo.gle/covid-19-open-data
    """
    # File title without extensions and suffix
    TITLE = "google_cloud_platform"
    # Dictionary of column names
    _MOBILITY_COLS_RAW = [
        "mobility_grocery_and_pharmacy",
        "mobility_parks",
        "mobility_transit_stations",
        "mobility_retail_and_recreation",
        "mobility_residential",
        "mobility_workplaces",
    ]
    MOBILITY_VARS = [v.capitalize() for v in _MOBILITY_COLS_RAW]
    COL_DICT = {
        "date": Term.DATE,
        **dict(zip(_MOBILITY_COLS_RAW, MOBILITY_VARS)),
    }
    # Stdout when downloading (shown at most one time)
    STDOUT = "Retrieving datasets from COVID-19 Open Data by Google Cloud Platform https://github.com/GoogleCloudPlatform/covid-19-open-data"
    # Citation
    CITATION = "O. Wahltinez and others (2020)," \
        " COVID-19 Open-Data: curating a fine-grained, global-scale data repository for SARS-CoV-2, " \
        " Work in progress, https://goo.gle/covid-19-open-data"

    def __init__(self, directory, update_interval):
        super().__init__(directory=directory, update_interval=None)

    def _country(self):
        """Returns country-level data.

        Returns:
            pandas.DataFrame:
                Index
                    reset index
                Columns
                    - Date (pandas.Timestamp): observation date
                    - ISO3 (str): country names
                    - Province (pandas.NA): NAs
                    - City (pandas.NA): NAs
                    - Mobility_grocery_and_pharmacy: % to baseline in visits (grocery markets, pharmacies etc.)
                    - Mobility_parks: % to baseline in visits (parks etc.)
                    - Mobility_transit_stations: % to baseline in visits (public transport hubs etc.)
                    - Mobility_retail_and_recreation: % to baseline in visits (restaurant, museums etc.)
                    - Mobility_residential: % to baseline in visits (places of residence)
                    - Mobility_workplaces: % to baseline in visits (places of work)
        """
        level_file = self._filer.csv(title=f"{self.TITLE}_{self.COUNTRY}".lower())["path_or_buf"]
        if self._provider.download_necessity(level_file):
            df = self._mobility()
            df = df.loc[df["ISO2"].str.len() == 2]
            df = df.merge(self._country_information()[["ISO2", "ISO3"]], how="left", on="ISO2")
            df = df.drop("ISO2", axis=1).dropna(subset=["ISO3"]).rename(columns={"ISO3": self.ISO3})
            df.to_csv(level_file, index=False)
        else:
            df = self._provider.read_csv(level_file, columns=None, date=self.DATE, date_format="%Y-%m-%d")
        df[self.PROVINCE] = pd.NA
        df[self.CITY] = pd.NA
        return df

    def _province(self, country):  # sourcery skip: class-extract-method
        """Returns province-level data.

        Args:
            country (str): country name

        Returns:
            pandas.DataFrame:
                Index
                    reset index
                Columns
                    - Date (pandas.Timestamp): observation date
                    - ISO3 (str): country names
                    - Province (str): province/state/prefecture names
                    - City (pandas.NA): NAs
                    - Mobility_grocery_and_pharmacy: % to baseline in visits (grocery markets, pharmacies etc.)
                    - Mobility_parks: % to baseline in visits (parks etc.)
                    - Mobility_transit_stations: % to baseline in visits (public transport hubs etc.)
                    - Mobility_retail_and_recreation: % to baseline in visits (restaurant, museums etc.)
                    - Mobility_residential: % to baseline in visits (places of residence)
                    - Mobility_workplaces: % to baseline in visits (places of work)
        """
        iso3 = self._to_iso3(country)[0]
        index_df = self._index_data()
        # Mobility data
        level_file = self._filer.csv(title=f"{self.TITLE}_{self.PROVINCE}".lower())["path_or_buf"]
        if self._provider.download_necessity(level_file):
            df = self._mobility()
            df = df.loc[df["ISO2"].str.len() != 2]
            df = df.merge(index_df, how="left", on="ISO2")
            df = df.drop("ISO2", axis=1).dropna(subset=["ISO3"]).rename(columns={"ISO3": self.ISO3})
            df = df.loc[(df[self.PROVINCE] != self.NA) & (df[self.CITY] == self.NA)]
            df.to_csv(level_file, index=False)
        else:
            df = self._provider.read_csv(level_file, columns=None, date=self.DATE, date_format="%Y-%m-%d")
        return df.loc[df[self.ISO3] == iso3]

    def _city(self, country, province):
        """Returns city-level data.

        Args:
            country (str): country name
            province (str): province/state/prefecture name

        Returns:
            pandas.DataFrame:
                Index
                    reset index
                Columns
                    - Date (pandas.Timestamp): observation date
                    - ISO3 (str): country names
                    - Province (str): province/state/prefecture names
                    - City (str): city names
                    - Mobility_grocery_and_pharmacy: % to baseline in visits (grocery markets, pharmacies etc.)
                    - Mobility_parks: % to baseline in visits (parks etc.)
                    - Mobility_transit_stations: % to baseline in visits (public transport hubs etc.)
                    - Mobility_retail_and_recreation: % to baseline in visits (restaurant, museums etc.)
                    - Mobility_residential: % to baseline in visits (places of residence)
                    - Mobility_workplaces: % to baseline in visits (places of work)
        """
        iso3 = self._to_iso3(country)[0]
        index_df = self._index_data()
        # Mobility data
        level_file = self._filer.csv(title=f"{self.TITLE}_{self.CITY}".lower())["path_or_buf"]
        if self._provider.download_necessity(level_file):
            df = self._mobility()
            df = df.loc[df["ISO2"].str.len() != 2]
            df = df.merge(index_df, how="left", on="ISO2")
            df = df.drop("ISO2", axis=1).dropna(subset=["ISO3"]).rename(columns={"ISO3": self.ISO3})
            df = df.loc[df[self.CITY] != self.NA]
            df.to_csv(level_file, index=False)
        else:
            df = self._provider.read_csv(level_file, columns=None, date=self.DATE, date_format="%Y-%m-%d")
        return df.loc[(df[self.ISO3] == iso3) & (df[self.PROVINCE] == province)]

    def _index_data(self):
        """Returns index data.

        Args:
            country (str): country name

        Returns:
            pandas.DataFrame:
                Index
                    reset index
                Columns
                    - location key (str): Google location keys
                    - ISO2 (str): ISO2 codes
                    - Province (str): province/state/prefecture names
                    - City (str): city names
        """
        URL_I = "https://storage.googleapis.com/covid19-open-data/v3/index.csv"
        index_cols = ["location_key", "subregion1_name", "subregion2_name", "locality_name", "iso_3166_1_alpha_3"]
        df = self._provide(url=URL_I, suffix="_index", columns=index_cols, date=None, date_format=None)
        df = df.rename(columns={"subregion1_name": self.PROVINCE, "iso_3166_1_alpha_3": self.ISO3})
        df[self.PROVINCE] = df[self.PROVINCE].fillna(self.NA).apply(unidecode)
        df[self.CITY] = df["subregion2_name"].fillna(df["locality_name"]).fillna(self.NA).apply(unidecode)
        return df.fillna(self.NA).loc[:, ["location_key", self.ISO3, self.PROVINCE, self.CITY]].rename(columns={"location_key": "ISO2"})

    def _mobility(self):
        """Returns mobility data.

        Returns:
            pandas.DataFrame:
                Index
                    reset index
                Columns
                    - Date (pandas.Timestamp): observation date
                    - ISO2 (str): ISO2 codes
                    - Mobility_grocery_and_pharmacy: % to baseline in visits (grocery markets, pharmacies etc.)
                    - Mobility_parks: % to baseline in visits (parks etc.)
                    - Mobility_transit_stations: % to baseline in visits (public transport hubs etc.)
                    - Mobility_retail_and_recreation: % to baseline in visits (restaurant, museums etc.)
                    - Mobility_residential: % to baseline in visits (places of residence)
                    - Mobility_workplaces: % to baseline in visits (places of work)
        """
        URL_M = "https://storage.googleapis.com/covid19-open-data/v3/mobility.csv"
        df = self._provide(
            url=URL_M, suffix="", columns=["date", "location_key", *self._MOBILITY_COLS_RAW], date="date", date_format="%Y-%m-%d")
        df.loc[:, self.MOBILITY_VARS] = df.loc[:, self.MOBILITY_VARS] + 100
        return df.rename(columns={"location_key": "ISO2"})
