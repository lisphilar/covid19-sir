#!/usr/bin/env python
# -*- coding: utf-8 -*-

import itertools
from varname import nameof
from covsirphy.util.error import SubsetNotFoundError, UnExpectedValueError, UnExecutedError
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
        kwargs: arguments of DataHandler.register()
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

    def __init__(self, country, province=None, **kwargs):
        # Details of the area name
        self._area_dict = {"country": str(country), "province": str(province or self.UNKNOWN)}
        # Data {str: instance}
        self._data_dict = dict.fromkeys(self.MAIN_DICT.keys(), None)
        # Auto complement
        self._complement_dict = {"auto_complement": True}
        self._complemented = None
        # Date
        self._first_date = None
        self._last_date = None
        self._today = None
        # Register datasets
        self.register(**kwargs)

    @property
    def complemented(self):
        """
        bool or str or None: whether complemented or not and the details, None when not confirmed
        """
        if self._complemented is None:
            raise UnExecutedError("DataHandler.records_main()")
        return self._complemented

    @property
    def first_date(self):
        """
        str: the first date of the records
        """
        return self._first_date

    @property
    def last_date(self):
        """
        str: the last date of the records
        """
        return self._last_date

    @property
    def today(self):
        """
        str: reference date to determine whether a phase is a past phase or a future phase
        """
        return self._today

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
        # Update date range
        try:
            self.set_date(
                first_date=self._first_date, last_date=self._last_date, today=self._today)
        except UnExecutedError:
            # Some of main datasets were not registered
            pass
        except SubsetNotFoundError as e:
            raise e from None
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

    def switch_complement(self, whether, **kwargs):
        """
        Switch whether perform auto complement or not. (Default: True)

        Args:
            whether (bool): if True and necessary, the number of cases will be complemented
            kwargs: the other arguments of JHUData.subset_complement()
        """
        self._complement_dict = {"auto_complement": bool(whether), **kwargs}

    def records_main(self):
        """
        Return main dataset of all dates as a dataframe.

        Raises:
            UnExecutedError: either JHUData or PopulationData was not registered
            SubsetNotFoundError: failed in subsetting because of lack of data

        Returns:
            pandas.DataFrame:
                Index
                    reset index
                Columns:
                    - Date(pd.TimeStamp): Observation date
                    - Confirmed(int): the number of confirmed cases
                    - Infected(int): the number of currently infected cases
                    - Fatal(int): the number of fatal cases
                    - Recovered (int): the number of recovered cases ( > 0)
                    - Susceptible(int): the number of susceptible cases
        """
        jhu_data = self._data_dict[nameof(JHUData)]
        population_data = self._data_dict[nameof(PopulationData)]
        if None in [jhu_data, population_data]:
            raise UnExecutedError("DataHandler.register(jhu_data, population_data)")
        df, self._complemented = jhu_data.records(
            **self._area_dict,
            start_date=self._first_date, end_date=self._last_date,
            population=population_data.value(**self._area_dict),
            **self._complement_dict,
        )
        return df

    def set_date(self, first_date=None, last_date=None, today=None):
        """
        Set the range of data and reference date to determine past/future of phases.

        Args:
            first_date (str or None): the first date of the records or None (min date of main dataset)
            last_date (str or None): the first date of the records or None (max date of main dataset)
            today (str or None): reference date to determine whether a phase is a past phase or a future phase

        Raises:
            UnExecutedError: either JHUData or PopulationData was not registered
            SubsetNotFoundError: failed in subsetting because of lack of data

        Note:
            When @today is None, the reference date will be the same as @last_date (or max date).
        """
        main_df = self.records_main()
        # The first date
        if first_date is None:
            self._first_date = main_df[self.DATE].min().strftime(self.DATE_FORMAT)
        else:
            self._ensure_date_order(self._first_date, first_date, name=nameof(first_date))
            self._first_date = first_date
        # The last date
        if last_date is None:
            self._last_date = main_df[self.DATE].max().strftime(self.DATE_FORMAT)
        else:
            self._ensure_date_order(last_date, self._last_date, name=nameof(last_date))
            self._last_date = last_date
        # Today
        if today is None:
            self._today = self._last_date
        else:
            self._ensure_date_order(self._first_date, today, name=nameof(today))
            self._ensure_date_order(today, self._last_date, name=nameof(today))
            self._today = today
