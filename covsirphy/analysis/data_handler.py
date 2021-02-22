#!/usr/bin/env python
# -*- coding: utf-8 -*-

import itertools
from varname import nameof
from covsirphy.util.error import UnExpectedValueError
from covsirphy.util.term import Term
from covsirphy.cleaning.cbase import CleaningBase
from covsirphy.cleaning.country_data import CountryData
from covsirphy.cleaning.jhu_data import JHUData
from covsirphy.cleaning.oxcgrt import OxCGRTData
from covsirphy.cleaning.pcr_data import PCRData
from covsirphy.cleaning.population import PopulationData
from covsirphy.cleaning.vaccine_data import VaccineData


class DataHandler(Term):
    """
    Data handler for analysis.

    Args:
        country (str): country name
        province (str or None): province name
    """
    # Main datasets {str: class}
    MAIN_DICT = {
        nameof(JHUData): JHUData,
        nameof(PopulationData): PopulationData
    }
    # Extra datasets {str: class}
    EXTRA_DICT = {
        nameof(CountryData): CountryData,
        nameof(OxCGRTData): OxCGRTData,
        nameof(PCRData): PCRData,
        nameof(VaccineData): VaccineData,
    }

    def __init__(self, country, province=None):
        # Details of the area name
        self._area_dict = {"country": str(country), "province": str(province or self.UNKNOWN)}
        # Data {str: instance}
        self._data_dict = {}

    def register(self, jhu_data=None, population_data=None, extras=None):
        """
        Register datasets.

        Args:
            jhu_data (covsirphy.JHUData or None): object of records
            population_data (covsirphy.PopulationData or None): PopulationData object
            extras (list[covsirphy.CleaningBase] or None): extra datasets

        Raises:
            TypeError: non-data cleaning instance was included
            UnExpectedValueError: instance of un-expected data cleaning class was included as an extra dataset
        """
        # Main: JHUData
        if jhu_data is not None:
            self._ensure_instance(jhu_data, JHUData, name=nameof(jhu_data))
            self._data_dict[nameof(JHUData)] = jhu_data
        # Main: PopulationData
        if population_data is not None:
            self._ensure_instance(population_data, PopulationData, name=nameof(population_data))
            self._data_dict[nameof(PopulationData)] = population_data
        # Extra datasets
        if extras is not None:
            self._register_extras(extras)

    def _register_extras(self, extras):
        """
        Verify the extra datasets.

        Args:
            extras (list[covsirphy.CleaningBase]): extra datasets

        Raises:
            TypeError: non-data cleaning instance was included as an extra dataset
            UnExpectedValueError: instance of un-expected data cleaning class was included as an extra dataset
        """
        self._ensure_list(extras, name=nameof(extras))
        # Verify the datasets
        for (i, extra_data) in enumerate(extras, start=1):
            statement = f"{self.num2str(i)} extra dataset"
            # Check the data is a data cleaning class
            self._ensure_instance(extra_data, CleaningBase, name=statement)
            # Check the data can be accepted as an extra dataset
            if type(extra_data) in self.EXTRA_DICT.values():
                continue
            raise UnExpectedValueError(
                name=statement, value=type(extra_data), candidates=list(self.EXTRA_DICT.keys()))
        # Register the datasets
        for (extra_data, name) in itertools.product(extras, self.EXTRA_DICT.keys()):
            self._data_dict[name] = extra_data
