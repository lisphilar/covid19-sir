#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pathlib import Path
from covsirphy.util.error import deprecate
from covsirphy.util.term import Term
from covsirphy._deprecated.dataloader import DataLoader


class COVID19DataHub(Term):
    """
    Deprecated. Please use DataLoader() class.
    Load datasets retrieved from COVID-19 Data Hub.
    https://covid19datahub.io/

    Args:
        filename (str): CSV filename to save records
    """

    @deprecate("covsirphy.COVID19DataHub()", new="covsirphy.DataLoader()", version="2.21.0-eta")
    def __init__(self, filename):
        self._filepath = Path(filename)

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
        loader = DataLoader(
            directory=self._filepath.parent,
            update_interval=0 if force else 12,
            basename_dict={"covid19dh": self._filepath.name},
            verbose=verbose
        )
        if force:
            self._filepath.unlink()
        return {
            "jhu": loader.jhu,
            "population": loader.population,
            "oxcgrt": loader.oxcgrt,
            "pcr": loader.pcr
        }[name]()

    @property
    def primary(self):
        """
        str: the list of primary sources.
        """
        loader = DataLoader(directory=self._filepath.parent)
        return loader.covid19dh_citation
