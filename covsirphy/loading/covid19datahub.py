#!/usr/bin/env python
# -*- coding: utf-8 -*-

from covsirphy.util.error import deprecate
from covsirphy.cleaning.cbase import CleaningBase
from covsirphy.cleaning.jhu_data import JHUData
from covsirphy.cleaning.oxcgrt import OxCGRTData
from covsirphy.cleaning.population import PopulationData
from covsirphy.cleaning.pcr_data import PCRData
from covsirphy.loading.db_covid19dh import _COVID19dh


class COVID19DataHub(_COVID19dh):
    """
    Deprecated. Please use DataLoader() class.
    Load datasets retrieved from COVID-19 Data Hub.
    https://covid19datahub.io/

    Args:
        filename (str): CSV filename to save records
    """
    # Class objects of datasets
    OBJ_DICT = {
        "jhu": JHUData,
        "population": PopulationData,
        "oxcgrt": OxCGRTData,
        "pcr": PCRData,
    }

    @deprecate("covsirphy.COVID19DataHub()", new="covsirphy.DataLoader()", version="2.21.0-eta")
    def __init__(self, filename):
        super().__init__(filename=filename)
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
            if set(self.COL_DICT.values()).issubset(df.columns):
                return df
        # Download dataset from server
        raw_df = self.download(verbose)
        raw_df.to_csv(self.filepath, index=False)
        return raw_df

    @property
    def primary(self):
        """
        str: the list of primary sources.
        """
        return self.primary_list or self._download()[-1]
