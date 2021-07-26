#!/usr/bin/env python
# -*- coding: utf-8 -*-

import warnings
import covid19dh
import pandas as pd
from covsirphy.util.term import Term
from covsirphy.loading.db_base import _RemoteDatabase


class _COVID19dh(_RemoteDatabase):
    """
    Access COVID-19 Data Hub.

    Args:
        filename (str): CSV filename to save records
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
        c_df, p_df, self.primary_list = self._download()
        # Merge the datasets
        df = pd.concat([c_df, p_df], axis=0, ignore_index=True)
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
            tuple:
                pandas.DataFrame: dataset at country level
                pandas.DataFrame: dataset at province level
                str: the list of primary sources

        Note:
            For some countries, province-level data is included.
        """
        warnings.simplefilter("ignore", ResourceWarning)
        levels = [f"administrative_area_level_{i}" for i in range(1, 4)]
        # Level 1 (country/region)
        c_raw, c_cite = covid19dh.covid19(country=None, level=1, verbose=False, raw=True)
        c_df = c_raw.groupby(levels[0]).ffill()
        for num in range(3):
            c_df.loc[:, levels[num]] = c_raw[levels[num]]
        # Level 2 (province/state)
        p_raw, p_cite = covid19dh.covid19(country=None, level=2, verbose=False, raw=True)
        p_df = p_raw.groupby(levels[:2]).ffill()
        for num in range(3):
            p_df.loc[:, levels[num]] = p_raw[levels[num]]
        # Citation
        cite = pd.concat([c_cite, p_cite], axis=0, ignore_index=True)
        cite = cite.loc[:, ["title", "year", "url"]]
        cite = cite.sort_values(["year", "url"], ascending=[False, True])
        cite.drop_duplicates(subset="title", inplace=True)
        series = cite.apply(lambda x: f"{x[0]} ({x[1]}), {x[2]}", axis=1)
        return (c_df, p_df, "\n".join(series.tolist()))

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
        c, p = "administrative_area_level_1", "administrative_area_level_2"
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
            df.loc[df[c] == ship, [c, p]] = [self.OTHERS, ship]
        return df
