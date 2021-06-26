#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pathlib import Path
import warnings
import covid19dh
import pandas as pd
from covsirphy.util.term import Term
from covsirphy.cleaning.cbase import CleaningBase
from covsirphy.cleaning.jhu_data import JHUData
from covsirphy.cleaning.oxcgrt import OxCGRTData
from covsirphy.cleaning.population import PopulationData
from covsirphy.cleaning.pcr_data import PCRData


class COVID19DataHub(Term):
    """
    Load datasets retrieved from COVID-19 Data Hub.
    https://covid19datahub.io/

    Args:
        filename (str): CSV filename to save records
    """
    # Citation
    CITATION = '(Secondary source)' \
        ' Guidotti, E., Ardia, D., (2020), "COVID-19 Data Hub",' \
        ' Journal of Open Source Software 5(51):2376, doi: 10.21105/joss.02376.'
    # Name conversion list of columns
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
        "stringency_index"
    ]
    _COL_DICT = {
        "date": Term.DATE,
        "administrative_area_level_1": Term.COUNTRY,
        "administrative_area_level_2": Term.PROVINCE,
        "tests": Term.TESTS,
        "confirmed": Term.C,
        "deaths": Term.F,
        "recovered": Term.R,
        "population": Term.N,
        "iso_alpha_3": Term.ISO3,
        **{v: v.capitalize() for v in _OXCGRT_COLS_RAW}
    }
    # Class objects of datasets
    OBJ_DICT = {
        "jhu": JHUData,
        "population": PopulationData,
        "oxcgrt": OxCGRTData,
        "pcr": PCRData,
    }

    def __init__(self, filename):
        try:
            self.filepath = Path(filename)
        except TypeError:
            raise TypeError(f"@filename should be a path-like object, but {filename} was applied.")
        self.filepath.parent.mkdir(exist_ok=True, parents=True)
        self.primary_list = None
        self._loaded_df = None

    def load(self, name="jhu", force=True, verbose=1):
        """
        Load the datasets of COVID-19 Data Hub and create dataset object.

        Args:
            name (str): name of dataset, "jhu", "population", "oxcgrt" or "pcr"
            force (bool): if True, always download the dataset from the server
            verbose (int): level of verbosity

        Returns:
            covsirphy.CleaningBase: the dataset

        Note:
            If @verbose is 2, detailed citation list will be shown when downloading.
            If @verbose is 1, how to show the list will be explained.
            Citation of COVID-19 Data Hub will be set as JHUData.citation etc.
        """
        if name not in self.OBJ_DICT:
            raise KeyError(
                f"@name must be {', '.join(list(self.OBJ_DICT.keys()))}, but {name} was applied.")
        # Get all data
        if self._loaded_df is None:
            self._loaded_df = self._load(force=force, verbose=verbose)
        return self.OBJ_DICT[name](data=self._loaded_df, citation=self.CITATION)

    def _load(self, force, verbose):
        """
        Load the datasets of COVID-19 Data Hub.

        Args:
            force (bool): if True, always download the dataset from the server
            verbose (int): level of verbosity

        Returns:
            pandas.DataFrame: as the same as COVID19DataHub._preprocessing()

        Note:
            If @verbose is 2, detailed citation list will be shown when downloading.
            If @verbose is 1, how to show the list will be explained.
        """
        # Use local CSV file
        if not force and self.filepath.exists():
            df = CleaningBase.load(
                self.filepath,
                dtype={
                    self.PROVINCE: "object", "Province/State": "object",
                    "key": "object", "key_alpha_2": "object",
                })
            if set(self._COL_DICT.values()).issubset(df.columns):
                return df
        # Download dataset from server
        raw_df = self._retrieve(verbose=verbose)
        raw_df.to_csv(self.filepath, index=False)
        return raw_df

    def _retrieve(self, verbose=1):
        """
        Retrieve datasets from COVID-19 Data Hub.
        Level 1 (country) and level2 (province) will be used and combined to a dataframe.

        Args:
            verbose (int): level of verbosity

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
        c_df = c_raw.groupby(levels[0]).ffill().fillna(0)
        for num in range(3):
            c_df.loc[:, levels[num]] = c_raw[levels[num]]
        # Level 2 (province/state)
        p_raw, p_cite = covid19dh.covid19(country=None, level=2, verbose=False, raw=True)
        p_df = p_raw.groupby(levels[:2]).ffill().fillna(0)
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
        # Replace column names
        df = df.rename(columns=self._COL_DICT)
        self._ensure_dataframe(df, columns=list(self._COL_DICT.values()))
        # Country
        df[self.COUNTRY] = df[self.COUNTRY].replace(
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
            df.loc[df[self.COUNTRY] == ship, [self.COUNTRY, self.PROVINCE]] = [self.OTHERS, ship]
        return df

    @property
    def primary(self):
        """
        str: the list of primary sources.
        """
        return self.primary_list or self._download()[-1]
