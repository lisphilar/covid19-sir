#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pathlib import Path
import warnings
import covid19dh
import pandas as pd
from covsirphy.util.file import save_dataframe
from covsirphy.util.term import Term
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
            raise TypeError(
                f"@filename should be a path-like object, but {filename} was applied.")
        self.filepath.parent.mkdir(exist_ok=True, parents=True)
        self.primary_list = None

    def load(self, name="jhu", force=True, verbose=1):
        """
        Load the datasets of COVID-19 Data Hub.

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
        if force and self.filepath.exists():
            self.filepath.unlink()
        if not self.filepath.exists():
            raw_df = self._retrieve(verbose=verbose)
            save_dataframe(raw_df, self.filepath, index=False)
        if name not in self.OBJ_DICT:
            raise KeyError(
                f"@name must be {', '.join(list(self.OBJ_DICT.keys()))}, but {name} was applied.")
        return self.OBJ_DICT[name](self.filepath, citation=self.CITATION)

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
        # Country level
        if verbose:
            print(
                "Retrieving datasets from COVID-19 Data Hub https://covid19datahub.io/")
        c_df, p_df, self.primary_list = self._download()
        # Change column names and select columns to use
        # All columns: https://covid19datahub.io/articles/doc/data.html
        col_dict = {
            "date": "ObservationDate",
            "tests": self.TESTS,
            "confirmed": self.C,
            "recovered": self.R,
            "deaths": "Deaths",
            "population": self.N,
            "iso_alpha_3": self.ISO3,
            "administrative_area_level_2": "Province/State",
            "administrative_area_level_1": "Country/Region",
        }
        columns = list(col_dict.values()) + OxCGRTData.OXCGRT_VARIABLES_RAW
        # Merge the datasets
        c_df = c_df.rename(col_dict, axis=1).loc[:, columns]
        p_df = p_df.rename(col_dict, axis=1).loc[:, columns]
        df = pd.concat([c_df, p_df], axis=0, ignore_index=True)
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
        Citation list will be saved to self.

        Returns:
            tuple:
                pandas.DataFrame: dataset at country level
                pandas.DataFrame: dataset at province level
                str: the list of primary sources

        Note:
            For some countries, province-level data is included.
        """
        warnings.simplefilter("ignore", ResourceWarning)
        c_res = covid19dh.covid19(
            country=None, level=1, verbose=False, raw=False)
        p_res = covid19dh.covid19(
            country=None, level=2, verbose=False, raw=False)
        try:
            c_df, c_cite = c_res
            p_df, p_cite = p_res
        except ValueError:
            # covid19dh <= 1.14
            c_df, c_cite = c_res.copy(), covid19dh.cite(c_res)
            p_df, p_cite = p_res.copy(), covid19dh.cite(p_res)
            citations = list(dict.fromkeys(c_cite + p_cite))
            return (c_df, p_df, "\n".join(citations))
        # Citation
        cite = pd.concat([c_cite, p_cite], axis=0, ignore_index=True)
        cite = cite.loc[:, ["title", "year", "url"]]
        cite = cite.sort_values(["year", "url"], ascending=[False, True])
        cite.drop_duplicates(subset="title", inplace=True)
        series = cite.apply(lambda x: f"{x[0]} ({x[1]}), {x[2]}", axis=1)
        return (c_df, p_df, "\n".join(series.tolist()))

    @property
    def primary(self):
        """
        str: the list of primary sources.
        """
        return self.primary_list or self._download()[-1]
