#!/usr/bin/env python
# -*- coding: utf-8 -*-

from datetime import timedelta
import itertools
import warnings
import numpy as np
import pandas as pd
import ruptures as rpt
from covsirphy.util.error import SubsetNotFoundError, UnExpectedValueError
from covsirphy.util.error import NotRegisteredMainError, NotRegisteredExtraError
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
    # {nameof(JHUData): JHUData} does not work with AST magics, including pytest and ipython
    # Main datasets {str: class}
    __NAME_JHU = "JHUData"
    __NAME_POPULATION = "PopulationData"
    MAIN_DICT = {
        __NAME_JHU: JHUData,
        __NAME_POPULATION: PopulationData
    }
    # Extra datasets {str: class}
    __NAME_COUNTRY = "CountryData"
    __NAME_OXCGRT = "OxCGRTData"
    __NAME_PCR = "PCRData"
    __NAME_VACCINE = "VaccineData"
    EXTRA_DICT = {
        __NAME_COUNTRY: CountryData,
        __NAME_OXCGRT: OxCGRTData,
        __NAME_PCR: PCRData,
        __NAME_VACCINE: VaccineData,
    }

    def __init__(self, country, province=None, **kwargs):
        # Details of the area name
        self._area_dict = {"country": str(country), "province": str(province or self.UNKNOWN)}
        # Data {str: instance}
        self._data_dict = dict.fromkeys(self.MAIN_DICT.keys(), None)
        # Population
        self._population = None
        # Auto complement: manually changed with DataHandler.switch_complement()
        self._complement_dict = {"auto_complement": True}
        self._complemented = None
        # Date
        self._first_date = None
        self._last_date = None
        self._today = None
        # Columns which is included in the main datasets (updated in .records_main()) except for 'Date'
        self._main_cols = None
        # Register datasets: date and main columns will be set internally if main data available
        self.register(**kwargs)

    @ property
    def main_satisfied(self):
        """
        bool: all main datasets were registered or not
        """
        return all(self._data_dict[name] for name in self.MAIN_DICT.keys())

    @ property
    def complemented(self):
        """
        bool or str: whether complemented or not and the details, None when not confirmed

        Raises:
            NotRegisteredMainError: no information because either JHUData or PopulationData was not registered
        """
        if self._complemented is None:
            raise NotRegisteredMainError(".register(jhu_data, population_data)")
        return self._complemented

    @ property
    def population(self):
        """
        int: population value

        Raises:
            NotRegisteredMainError: no information because either JHUData or PopulationData was not registered
        """
        if self._population is None:
            raise NotRegisteredMainError(".register(jhu_data, population_data)")
        return self._population

    @ property
    def first_date(self):
        """
        str: the first date of the records
        """
        return self._first_date

    @ property
    def last_date(self):
        """
        str: the last date of the records
        """
        return self._last_date

    @ property
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
            self._ensure_instance(jhu_data, JHUData, name="jhu_data")
            self._data_dict[self.__NAME_JHU] = jhu_data
        # Main: PopulationData
        if population_data is not None:
            self._ensure_instance(population_data, PopulationData, name="population_data")
            self._data_dict[self.__NAME_POPULATION] = population_data
        # Update date range
        try:
            self.timepoints(
                first_date=self._first_date, last_date=self._last_date, today=self._today)
        except NotRegisteredMainError:
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
        self._ensure_list(extras, name="extras")
        # Verify the datasets
        for (i, extra_data) in enumerate(extras, start=1):
            statement = f"{self.num2str(i)} extra dataset"
            # Check the data is a data cleaning class
            self._ensure_instance(extra_data, CleaningBase, name=statement)
            # Check the data can be accepted as an extra dataset
            if isinstance(extra_data, tuple(self.EXTRA_DICT.values())):
                continue
            raise UnExpectedValueError(
                name=statement, value=type(extra_data), candidates=list(self.EXTRA_DICT.keys()))
        # Register the datasets
        extra_iter = itertools.product(extras, self.EXTRA_DICT.items())
        for (extra_data, (name, data_class)) in extra_iter:
            if isinstance(extra_data, data_class):
                self._data_dict[name] = extra_data

    def recovery_period(self):
        """
        Return representative value of recovery period of all countries.

        Raises:
            NotRegisteredMainError: JHUData was not registered

        Returns:
            int: recovery period [days]
        """
        jhu_data = self._data_dict[self.__NAME_JHU]
        if jhu_data is None:
            raise NotRegisteredMainError(".register(jhu_data)")
        return jhu_data.recovery_period

    def records_main(self):
        """
        Return records of the main datasets as a dataframe.

        Raises:
            NotRegisteredMainError: either JHUData or PopulationData was not registered
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
        jhu_data = self._data_dict[self.__NAME_JHU]
        population_data = self._data_dict[self.__NAME_POPULATION]
        # Main datasets should be registered
        if None in [jhu_data, population_data]:
            raise NotRegisteredMainError(".register(jhu_data, population_data)")
        # Population
        self._population = population_data.value(**self._area_dict)
        # Subsetting
        df, self._complemented = jhu_data.records(
            **self._area_dict,
            start_date=self._first_date, end_date=self._last_date,
            population=self._population,
            **self._complement_dict,
        )
        # Columns which are included in the main dataset except for 'Date'
        self._main_cols = list(set(df.columns) - set([self.DATE]))
        return df

    def switch_complement(self, whether=None, **kwargs):
        """
        Switch whether perform auto complement or not. (Default: True)

        Args:
            whether (bool or None): if True and necessary, the number of cases will be complemented
            kwargs: the other arguments of JHUData.subset_complement()

        Note:
            When @whether is None, @whether will not be changed.
        """
        comp_dict = self._complement_dict.copy()
        if whether is not None:
            comp_dict["auto_complement"] = bool(whether)
        comp_dict.update(kwargs)
        self._complement_dict = comp_dict.copy()

    def show_complement(self):
        """
        Show the details of complement that was (or will be) performed for the records.

        Raises:
            NotRegisteredMainError: JHUData was not registered

        Returns:
            pandas.DataFrame: as the same as JHUData.show_complement()

        Note:
            Keyword arguments of JHUData,subset_complement() can be specified with DataHandler.switch_complement().
        """
        jhu_data = self._data_dict[self.__NAME_JHU]
        if jhu_data is None:
            raise NotRegisteredMainError(".register(jhu_data)")
        comp_dict = self._complement_dict.copy()
        comp_dict.pop("auto_complement")
        return jhu_data.show_complement(
            start_date=self._first_date, end_date=self._last_date, **self._area_dict, ** comp_dict)

    def timepoints(self, first_date=None, last_date=None, today=None):
        """
        Set the range of data and reference date to determine past/future of phases.

        Args:
            first_date (str or None): the first date of the records or None (min date of main dataset)
            last_date (str or None): the first date of the records or None (max date of main dataset)
            today (str or None): reference date to determine whether a phase is a past phase or a future phase

        Raises:
            NotRegisteredMainError: either JHUData or PopulationData was not registered
            SubsetNotFoundError: failed in subsetting because of lack of data

        Note:
            When @today is None, the reference date will be the same as @last_date (or max date).
        """
        main_df = self.records_main()
        # The first date
        if first_date is None:
            self._first_date = main_df[self.DATE].min().strftime(self.DATE_FORMAT)
        else:
            self._ensure_date_order(self._first_date, first_date, name="first_date")
            self._first_date = first_date
        # The last date
        if last_date is None:
            self._last_date = main_df[self.DATE].max().strftime(self.DATE_FORMAT)
        else:
            self._ensure_date_order(last_date, self._last_date, name="last_date")
            self._last_date = last_date
        # Today
        if today is None:
            self._today = self._last_date
        else:
            self._ensure_date_order(self._first_date, today, name="today")
            self._ensure_date_order(today, self._last_date, name="today")
            self._today = today

    def records_extras(self):
        """
        Return records of the extra datasets as a dataframe.

        Raises:
            NotRegisteredMainError: either JHUData or PopulationData was not registered
            NotRegisteredExtraError: no extra datasets were registered

        Returns:
            pandas.DataFrame:
                Index
                    reset index
                Columns:
                    - Date(pd.TimeStamp): Observation date
                    - columns defined in the extra datasets
        """
        if None in self._data_dict.values():
            raise NotRegisteredMainError(".register(jhu_data, population_data)")
        if not set(self._data_dict) - set(self.MAIN_DICT):
            raise NotRegisteredExtraError(
                ".register(jhu_data, population_data, extras=[...])",
                message="with extra datasets")
        # Get all subset
        df = pd.DataFrame(columns=[self.DATE])
        for (name, data) in self._data_dict.items():
            if name in self.MAIN_DICT:
                continue
            try:
                subset_df = data.subset(**self._area_dict)
            except TypeError:
                subset_df = data.subset(country=self._area_dict["country"])
            except SubsetNotFoundError:
                continue
            new_cols = (set(subset_df) - set(df.columns)) | set([self.DATE])
            subset_df = subset_df.loc[:, subset_df.columns.isin(new_cols)]
            df = df.merge(subset_df, how="outer", on=self.DATE)
        # Remove columns which is included in the main datasets
        df = df.loc[:, ~df.columns.isin(self._main_cols)]
        # Data cleaning
        df = df.set_index(self.DATE).resample("D").last()
        df = df.fillna(method="ffill").fillna(0)
        # Subsetting by dates
        df = df.loc[pd.to_datetime(self._first_date): pd.to_datetime(self._last_date)]
        # Convert float values to integer if values will not be changed
        for col in df.columns:
            converted2int = df[col].astype(np.int64)
            if np.array_equal(converted2int, df[col]):
                df[col] = converted2int
        return df.reset_index()

    def _records(self, main=True, extras=True):
        """
        Return records of the datasets as a dataframe.

        Args:
            main (bool): whether include main datasets or not
            extras (bool): whether include extra datasets or not

        Raises:
            NotRegisteredMainError: either JHUData or PopulationData was not registered
            SubsetNotFoundError: failed in subsetting because of lack of data
            NotRegisteredExtraError: @extras is True and no extra datasets were registered
            ValueError: both of @main and @extras were False

        Returns:
            pandas.DataFrame:
                Index
                    reset index
                Columns:
                    - Date(pd.TimeStamp): Observation date
                    - if @main is True,
                        - Confirmed(int): the number of confirmed cases
                        - Infected(int): the number of currently infected cases
                        - Fatal(int): the number of fatal cases
                        - Recovered (int): the number of recovered cases ( > 0)
                        - Susceptible(int): the number of susceptible cases
                    - if @extra is True,
                        - columns defined in the extra datasets
        """
        if main and extras:
            main_df = self.records_main()
            extra_df = self.records_extras()
            return main_df.merge(extra_df, on=self.DATE)
        if main:
            return self.records_main()
        if extras:
            return self.records_extras()
        raise ValueError("Either @main or @extras must be True.")

    def records(self, main=True, extras=True, past=True, future=True):
        """
        Return records of the datasets as a dataframe.

        Args:
            main (bool): whether include main datasets or not
            extras (bool): whether include extra datasets or not
            past (bool): whether include past records or not
            future (bool): whether include future records or not

        Raises:
            NotRegisteredMainError: either JHUData or PopulationData was not registered
            SubsetNotFoundError: failed in subsetting because of lack of data
            NotRegisteredExtraError: @extras is True and no extra datasets were registered
            ValueError: both of @main and @extras were False, or both of @past and @future were False

        Returns:
            pandas.DataFrame:
                Index
                    reset index
                Columns:
                    - Date(pd.TimeStamp): Observation date
                    - if @main is True,
                        - Confirmed(int): the number of confirmed cases
                        - Infected(int): the number of currently infected cases
                        - Fatal(int): the number of fatal cases
                        - Recovered (int): the number of recovered cases ( > 0)
                        - Susceptible(int): the number of susceptible cases
                    - if @extra is True,
                        - columns defined in the extra datasets
        """
        if past and future:
            return self._records(main=main, extras=extras)
        if not past and not future:
            raise ValueError("Either @past or @future must be True.")
        df = self._records(main=main, extras=extras).set_index(self.DATE)
        if past:
            return df.loc[:pd.to_datetime(self._today)].reset_index()
        if future:
            return df.loc[pd.to_datetime(self._today) + timedelta(days=1):].reset_index()

    def records_all(self):
        """
        Return registered all records of the datasets as a dataframe.

        Raises:
            NotRegisteredMainError: either JHUData or PopulationData was not registered
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
                    - columns defined in the extra datasets
        """
        try:
            return self.records(main=True, extras=True, past=True, future=True)
        except NotRegisteredExtraError:
            return self.records(main=True, extras=False, past=True, future=True)

    def estimate_delay(self, indicator, target, delay_name="Period Length"):
        """
        Estimate the average day [days] between the indicator and the target.
        We assume that the indicator impact on the target value with delay.
        All results will be returned with a dataframe.

        Args:
            indicator (str): indicator name, a column of any registered datasets
            target (str): target name, a column of any registered datasets
            delay_name (str): column name of delay in the output dataframe

        Raises:
            NotRegisteredMainError: either JHUData or PopulationData was not registered
            SubsetNotFoundError: failed in subsetting because of lack of data
            UserWarning: failed in calculating and returned the default value (recovery period)

        Returns:
            pandas.DataFrame:
                Index
                    reset index
                Columns
                    - (int or float): column defined by @indicator
                    - (int or float): column defined by @target
                    - (int): column defined by @delay_name [days]

        Note:
            - We use change point analysis of ruptures package. Refer to the documentation.
              https://centre-borelli.github.io/ruptures-docs/
            - When failed in calculation, recovery period will be returned after raising UserWarning.
        """
        output_cols = [target, indicator, delay_name]
        # Create dataframe with indicator and target
        record_df = self.records_all()
        self._ensure_list(
            [indicator, target], candidates=record_df.columns.tolist(), name="indicator and target")
        pivot_df = record_df.pivot_table(values=indicator, index=target)
        run_df = pivot_df.copy()
        # Convert index (target) to serial numbers
        serial_df = pd.DataFrame(np.arange(1, run_df.index.max() + 1, 1))
        serial_df.index += 1
        run_df = run_df.join(serial_df, how="outer")
        series = run_df.reset_index(drop=True).iloc[:, 0].dropna()
        # Detection with Ruptures using indicator values
        warnings.simplefilter("ignore", category=RuntimeWarning)
        algorithm = rpt.Pelt(model="rbf", jump=1, min_size=0)
        try:
            results = algorithm.fit_predict(series.values, pen=0.5)
        except ValueError:
            default_delay = self.recovery_period()
            warnings.warn(
                f"Delay days could not be estimated and delay set to default: {default_delay} [days]",
                UserWarning, stacklevel=2)
            return pd.DataFrame(columns=output_cols)
        # Convert the output of Ruptures to indicator values
        reset_series = series.reset_index(drop=True)
        reset_series.index += 1
        results_df = reset_series[results].reset_index()
        results_df = results_df.interpolate(method="linear").dropna().astype(np.float64)
        # Convert the indicator values to dates
        df = pd.merge_asof(
            results_df.sort_values(indicator),
            pivot_df.astype(np.float64).reset_index().sort_values(indicator),
            on=indicator, direction="nearest")
        # Calculate number of days between the periods
        df[delay_name] = df["index"].sort_values(ignore_index=True).diff()
        return df.loc[:, output_cols]
