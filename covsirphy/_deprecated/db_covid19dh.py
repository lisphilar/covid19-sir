#!/usr/bin/env python
# -*- coding: utf-8 -*-

from io import BytesIO, StringIO
import urllib
from zipfile import ZipFile
import pandas as pd
import requests
from covsirphy.util.term import Term
from covsirphy._deprecated.db_base import _RemoteDatabase


class _COVID19dh(_RemoteDatabase):
    """
    Access COVID-19 Data Hub.

    Args:
        filename (str): CSV filename to save records
        iso3 (str or None): ISO3 code of the country which must be included in the dataset or None (all available countries)
    """
    # Citation
    CITATION = '(Secondary source)' \
        ' Guidotti, E., Ardia, D., (2020), "COVID-19 Data Hub",' \
        ' Journal of Open Source Software 5(51):2376, doi: 10.21105/joss.02376.'
    # Column names and data types
    # {"name in database": "name defined in Term class"}
    _OXCGRT_COLS_RAW = [
        "school_closing",
        "workplace_closing",
        "cancel_events",
        "gatherings_restrictions",
        "transport_closing",
        "stay_home_restrictions",
        "internal_movement_restrictions",
        "international_movement_restrictions",
        "information_campaigns",
        "testing_policy",
        "contact_tracing",
        "stringency_index",
    ]
    OXCGRT_VARS = [v.capitalize() for v in _OXCGRT_COLS_RAW]
    COL_DICT = {
        "date": Term.DATE,
        "administrative_area_level_1": Term.COUNTRY,
        "administrative_area_level_2": Term.PROVINCE,
        "tests": Term.TESTS,
        "confirmed": Term.C,
        "deaths": Term.F,
        "recovered": Term.R,
        "population": Term.N,
        "iso_alpha_3": Term.ISO3,
        **{v: v.capitalize() for v in _OXCGRT_COLS_RAW},
    }

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
                    defined by the first values of self.COL_DICT.values()

        Note:
            If @verbose is 2, detailed citation list will be shown when downloading.
            If @verbose is 1, how to show the list will be explained.
        """

        # Download datasets
        if verbose:
            print("Retrieving datasets from COVID-19 Data Hub https://covid19datahub.io/")
        df = self._download()
        # Perform pre-processing
        df = self._preprocessing(df)
        # Show citation list
        if verbose:
            if isinstance(verbose, int) and verbose >= 2:
                print("\nDetailed citaition list:")
                print(self.primary_list)
            else:
                print("\tPlease set verbose=2 to see the detailed citation list.")
        return df

    def _download(self):
        """
        Retrieve dataset and citation list from COVID-19 Data Hub.

        Returns:
            pandas.DataFrame: raw dataset
        """
        if self._iso3 is None:
            url = "https://storage.covid19datahub.io/level/1.csv.zip"
        else:
            url = f"https://storage.covid19datahub.io/country/{self._iso3}.csv.zip"
        raw = self._read_csv(url, col_dict=None, date="date", date_format="%Y-%m-%d")
        raw = raw.groupby("id").ffill()
        # Remove city-level data
        return raw.loc[raw["administrative_area_level_3"].isna()]

    def primary(self):
        """
        Retrieve citation list from COVID-19 Data Hub.

        Returns:
            str: the list of primary sources
        """
        c_url = "https://storage.covid19datahub.io/src.csv"
        try:
            cite = pd.read_csv(c_url, storage_options={"User-Agent": "Mozilla/5.0"})
        except TypeError:
            # When pandas < 1.2: note that pandas v1.2.0 requires Python >= 3.7.1
            r = requests.get(url=c_url, headers={"User-Agent": "Mozilla/5.0"})
            cite = pd.read_csv(StringIO(r.text))
        if self._iso3 is not None:
            cite = cite.loc[cite["iso_alpha_3"] == self._iso3]
        cite = cite.loc[cite["data_type"].isin(self.COL_DICT.keys())]
        cite = cite.loc[:, ["title", "year", "url"]].drop_duplicates(subset="title")
        cite = cite.sort_values(["year", "url"], ascending=[False, True])
        series = cite.apply(lambda x: f"{x[0]} ({x[1]}), {x[2]}", axis=1)
        return "\n".join(series.tolist())

    def _preprocessing(self, raw):
        """
        Perform pre-processing with the raw dataset.

        Args:
            raw (pandas.DataFrame):
                Index
                    reset index
                Columns
                    Refer to COVID-19 Data Hub homepage.
                    https://covid19datahub.io/articles/doc/data.html

        Returns:
            pandas.DataFrame:
                Index
                    reset index
                Columns
                    - Date: observation date
                    - Country: country/region name
                    - Province: province/prefecture/state name
                    - Confirmed: the number of confirmed cases
                    - Infected: the number of currently infected cases
                    - Fatal: the number of fatal cases
                    - Recovered: the number of recovered cases
                    - School_closing
                    - Workplace_closing
                    - Cancel_events
                    - Gatherings_restrictions
                    - Transport_closing
                    - Stay_home_restrictions
                    - Internal_movement_restrictions
                    - International_movement_restrictions
                    - Information_campaigns
                    - Testing_policy
                    - Contact_tracing
                    - Stringency_index

        Note:
            Data types are not confirmed.
        """
        df = raw.copy()
        c, p, iso = "administrative_area_level_1", "administrative_area_level_2", "iso_alpha_3"
        # Country
        df[c] = df[c].replace(
            {
                # COD
                "Congo, the Democratic Republic of the": "Democratic Republic of the Congo",
                # COG
                "Congo": "Republic of the Congo",
                # KOR
                "Korea, South": "South Korea",
            }
        )
        # Set 'Others' as the country name of cruise ships
        ships = ["Diamond Princess", "Costa Atlantica", "Grand Princess", "MS Zaandam"]
        for ship in ships:
            df.loc[df[c] == ship, [c, p, iso]] = [self.OTHERS, ship, self.OTHERS]
        return df

    def _read_csv(self, filepath_or_buffer, col_dict, date="date", date_format="%Y-%m-%d"):
        """Read CSV data.

        Args:
            filepath_or_buffer (str, path object or file-like object): file path or URL
            col_dict (dict[str, str] or None): dictionary to convert column names or None (not perform conversion)
            date (str): column name of date
            date_format (str): format of date column, like %Y-%m-%d
        """
        read_dict = {
            "header": 0, "usecols": None if col_dict is None else list(col_dict.keys()),
            "parse_dates": None if date is None else [date], "date_parser": lambda x: pd.datetime.strptime(x, date_format)
        }
        try:
            df = pd.read_csv(filepath_or_buffer, **read_dict)
        except urllib.error.HTTPError:
            try:
                df = pd.read_csv(filepath_or_buffer, storage_options={"User-Agent": "Mozilla/5.0"}, **read_dict)
            except TypeError:
                # When pandas < 1.2: note that pandas v1.2.0 requires Python >= 3.7.1
                r = requests.get(url=filepath_or_buffer, headers={"User-Agent": "Mozilla/5.0"})
                z = ZipFile(BytesIO(r.content))
                with z.open(z.namelist()[0], "r") as fh:
                    df = pd.read_csv(BytesIO(fh.read()), **read_dict)
        return df if col_dict is None else df.rename(columns=col_dict)
