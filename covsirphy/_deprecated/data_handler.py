#!/usr/bin/env python
# -*- coding: utf-8 -*-

from datetime import timedelta
import itertools
import warnings
import numpy as np
import pandas as pd
import ruptures as rpt
from covsirphy.util.error import SubsetNotFoundError, UnExpectedValueError, deprecate
from covsirphy.util.error import NotRegisteredMainError, NotRegisteredExtraError
from covsirphy.util.term import Term
from covsirphy._deprecated.cbase import CleaningBase
from covsirphy._deprecated.country_data import CountryData
from covsirphy._deprecated.japan_data import JapanData
from covsirphy._deprecated.jhu_data import JHUData
from covsirphy._deprecated.oxcgrt import OxCGRTData
from covsirphy._deprecated.pcr_data import PCRData
from covsirphy._deprecated.population import PopulationData
from covsirphy._deprecated.vaccine_data import VaccineData
from covsirphy._deprecated.mobility_data import MobilityData


class DataHandler(Term):
    """
    Data handler for analysis.

    Args:
        country (str): country name
        province (str or None): province name
        kwargs: arguments of DataHandler.register()
    """
    # Deprecated
    __NAME_COUNTRY = "CountryData"
    __NAME_JAPAN = "JapanData"
    # Extra datasets {str: class}
    __NAME_OXCGRT = "OxCGRTData"
    __NAME_PCR = "PCRData"
    __NAME_VACCINE = "VaccineData"
    __NAME_MOBILE = "MobilityData"
    EXTRA_DICT = {
        __NAME_COUNTRY: CountryData,
        __NAME_JAPAN: JapanData,
        __NAME_OXCGRT: OxCGRTData,
        __NAME_PCR: PCRData,
        __NAME_VACCINE: VaccineData,
        __NAME_MOBILE: MobilityData,
    }

    @deprecate(old="DataHandler", version="2.24.0-xi")
    def __init__(self, country, province=None, **kwargs):
        # Details of the area name
        self._area_dict = {"country": str(country), "province": str(province or self.NA)}
        # Main dataset before complement
        main_cols = [self.DATE, self.C, self.CI, self.F, self.R, self.S]
        self._main_raw = pd.DataFrame(columns=main_cols)
        # Main dataset After complement
        self._main_df = pd.DataFrame(columns=main_cols)
        # Extra dataset
        self._extra_df = pd.DataFrame(columns=[self.DATE])
        # Population
        self._population = None
        # Complement
        self._jhu_data = None
        self._complemented = None
        self._comp_dict = {}
        # Date
        self._first_date = None
        self._last_date = None
        self._today = None
        # Register datasets: date and main columns will be set internally if main data available
        self.register(**kwargs)

    @classmethod
    def _ensure_date_order(cls, previous_date, following_date, name="following_date"):
        """
        Ensure that the order of dates.

        Args:
            previous_date (str or pandas.Timestamp): previous date
            following_date (str or pandas.Timestamp): following date
            name (str): name of @following_date

        Raises:
            ValueError: @previous_date > @following_date
        """
        previous_date = cls._ensure_date(previous_date)
        following_date = cls._ensure_date(following_date)
        p_str = previous_date.strftime(cls.DATE_FORMAT)
        f_str = following_date.strftime(cls.DATE_FORMAT)
        if previous_date <= following_date:
            return None
        raise ValueError(f"@{name} must be the same as/over {p_str}, but {f_str} was applied.")

    @property
    def main_satisfied(self):
        """
        bool: all main datasets were registered or not
        """
        return not self._main_raw.empty

    @property
    def complemented(self):
        """
        bool or str: whether complemented or not and the details

        Raises:
            NotRegisteredMainError: no information because JHUData was not registered
        """
        if not self.main_satisfied:
            raise NotRegisteredMainError(".register(jhu_data)")
        return self._complemented

    @property
    @deprecate("DataHandler.population property", version="2.19.1-lambda")
    def population(self):
        """
        int: population value

        Raises:
            NotRegisteredMainError: no information because JHUData was not registered
        """
        if self._population is None:
            raise NotRegisteredMainError(".register(jhu_data)")
        return self._population

    @property
    def first_date(self):
        """
        str or None: the first date of the records
        """
        return self._first_date.strftime(self.DATE_FORMAT)

    @property
    def last_date(self):
        """
        str or None: the last date of the records
        """
        return self._last_date.strftime(self.DATE_FORMAT)

    @property
    def today(self):
        """
        str or None: reference date to determine whether a phase is a past phase or a future phase
        """
        return self._today.strftime(self.DATE_FORMAT)

    @staticmethod
    def _ensure_instance(target, class_obj, name="target"):
        """
        Ensure the target is an instance of the class object.

        Args:
            target (instance): target to ensure
            parent (class): class object
            name (str): argument name of the target

        Returns:
            object: as-is target
        """
        s = f"@{name} must be an instance of {class_obj}, but {type(target)} was applied."
        if not isinstance(target, class_obj):
            raise TypeError(s)
        return target

    def register(self, jhu_data=None, population_data=None, extras=None):
        """
        Register datasets.

        Args:
            jhu_data (covsirphy.JHUData or None): object of records
            population_data (covsirphy.PopulationData or None): PopulationData object (deprecated)
            extras (list[covsirphy.CleaningBase] or None): extra datasets

        Raises:
            TypeError: non-data cleaning instance was included
            UnExpectedValueError: instance of un-expected data cleaning class was included as an extra dataset
        """
        # Main: JHUData
        if jhu_data is not None:
            self._ensure_instance(jhu_data, JHUData, name="jhu_data")
            try:
                self._main_raw = jhu_data.subset(**self._area_dict, recovered_min=0)
            except SubsetNotFoundError as e:
                raise e from None
            self._jhu_data = jhu_data
            self.switch_complement(whether=True)
        # Main: PopulationData
        if population_data is not None:
            warnings.warn(
                ".register(population_data) was deprecated because population values are included in JHUData.",
                DeprecationWarning,
                stacklevel=2
            )
            self._ensure_instance(population_data, PopulationData, name="population_data")
            self._population = population_data.value(**self._area_dict)
        # Extra datasets
        if extras is not None:
            self._register_extras(extras)

    @staticmethod
    def _ensure_list(target, candidates=None, name="target"):
        """
        Ensure the target is a sub-list of the candidates.

        Args:
            target (list[object]): target to ensure
            candidates (list[object] or None): list of candidates, if we have
            name (str): argument name of the target

        Returns:
            object: as-is target
        """
        if not isinstance(target, (list, tuple)):
            raise TypeError(f"@{name} must be a list or tuple, but {type(target)} was applied.")
        if candidates is None:
            return target
        # Check the target is a sub-list of candidates
        try:
            strings = [str(candidate) for candidate in candidates]
        except TypeError:
            raise TypeError(f"@candidates must be a list, but {candidates} was applied.") from None
        ok_list = [element in candidates for element in target]
        if all(ok_list):
            return target
        candidate_str = ", ".join(strings)
        raise KeyError(f"@{name} must be a sub-list of [{candidate_str}], but {target} was applied.") from None

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
            if isinstance(extra_data, (CountryData, JapanData)):
                warnings.warn(
                    ".register(extras=[CountryData, JapanData]) was deprecated because its role is played by the other classes.",
                    DeprecationWarning,
                    stacklevel=2
                )
            if isinstance(extra_data, tuple(self.EXTRA_DICT.values())):
                continue
            raise UnExpectedValueError(
                name=statement, value=type(extra_data), candidates=list(self.EXTRA_DICT.keys()))
        # Register the datasets
        extra_df = self._extra_df.set_index(self.DATE)
        for (extra_data, data_class) in itertools.product(extras, self.EXTRA_DICT.values()):
            if isinstance(extra_data, data_class):
                try:
                    subset_df = extra_data.subset(**self._area_dict)
                except TypeError:
                    subset_df = extra_data.subset(country=self._area_dict["country"])
                except SubsetNotFoundError:
                    continue
                extra_df = extra_df.combine_first(subset_df.set_index(self.DATE))
        self._extra_df = extra_df.reset_index()

    def recovery_period(self):
        """
        Return representative value of recovery period of all countries.

        Raises:
            NotRegisteredMainError: JHUData was not registered

        Returns:
            int or None: recovery period [days]
        """
        if self._jhu_data is None:
            raise NotRegisteredMainError(".register(jhu_data)")
        return self._jhu_data.recovery_period

    def records_main(self):
        """
        Return records of the main datasets as a dataframe from the first date to the last date.

        Raises:
            NotRegisteredMainError: JHUData was not registered
            SubsetNotFoundError: failed in subsetting because of lack of data

        Returns:
            pandas.DataFrame:
                Index
                    reset index
                Columns:
                    - Date (pd.Timestamp): Observation date
                    - Confirmed (int): the number of confirmed cases
                    - Infected (int): the number of currently infected cases
                    - Fatal (int): the number of fatal cases
                    - Recovered (int): the number of recovered cases ( > 0)
                    - Susceptible (int): the number of susceptible cases
        """
        if self._main_df.empty:
            raise NotRegisteredMainError(".register(jhu_data)")
        df = self._main_df.copy()
        df = df.loc[(df[self.DATE] >= self._first_date) & (df[self.DATE] <= self._last_date)]
        return df.reset_index(drop=True)

    def switch_complement(self, whether=None, **kwargs):
        """
        Switch whether perform auto complement or not.

        Args:
            whether (bool): if True and necessary, the number of cases will be complemented
            kwargs: the other arguments of JHUData.subset_complement()
        """
        if not whether:
            df = self._main_raw.copy()
            self._main_df = df.loc[df[self.R] > 0].reset_index(drop=True)
            self._complemented = False
            return
        self._comp_dict.update(kwargs)
        if self._jhu_data is None:
            return
        self._main_df, self._complemented = self._jhu_data.records(**self._area_dict, **self._comp_dict)
        self.timepoints()

    def show_complement(self, **kwargs):
        """
        Show the details of complement that was (or will be) performed for the records.

        Args:
            kwargs: keyword arguments of JHUDataComplementHandler() i.e. control factors of complement

        Raises:
            NotRegisteredMainError: JHUData was not registered

        Returns:
            pandas.DataFrame: as the same as JHUData.show_complement()
        """
        if self._jhu_data is None:
            raise NotRegisteredMainError(".register(jhu_data)")
        comp_dict = self._comp_dict.copy()
        comp_dict.update(kwargs)
        return self._jhu_data.show_complement(
            start_date=self._first_date, end_date=self._last_date, **self._area_dict, **comp_dict)

    def timepoints(self, first_date=None, last_date=None, today=None):
        """
        Set the range of data and reference date to determine past/future of phases.

        Args:
            first_date (str or None): the first date of the records or None (min date of main dataset)
            last_date (str or None): the first date of the records or None (max date of main dataset)
            today (str or None): reference date to determine whether a phase is a past phase or a future phase

        Raises:
            NotRegisteredMainError: JHUData was not registered
            SubsetNotFoundError: failed in subsetting because of lack of data

        Note:
            When @today is None, the reference date will be the same as @last_date (or max date).
        """
        df = self._main_df.copy()
        first_date = self._ensure_date(
            first_date, name="first_date", default=self._first_date or df[self.DATE].min())
        last_date = self._ensure_date(
            last_date, name="last_date", default=self._last_date or df[self.DATE].max())
        today = self._ensure_date(today, name="today", default=min(self._today or last_date, last_date))
        # Check the order of dates
        self._ensure_date_order(df[self.DATE].min(), first_date, name="first_date")
        self._ensure_date_order(last_date, df[self.DATE].max(), name="the last date before changing")
        self._ensure_date_order(first_date, today, name="today")
        self._ensure_date_order(today, last_date, name="last_date")
        # Set timepoints
        self._first_date = first_date
        self._last_date = last_date
        self._today = today

    def records_extras(self):
        """
        Return records of the extra datasets as a dataframe.

        Raises:
            NotRegisteredMainError: JHUData was not registered
            NotRegisteredExtraError: no extra datasets were registered

        Returns:
            pandas.DataFrame:
                Index
                    reset index
                Columns:
                    - Date(pd.Timestamp): Observation date
                    - columns defined in the extra datasets
        """
        if self._main_df.empty:
            raise NotRegisteredMainError(".register(jhu_data)")
        if self._extra_df.empty:
            raise NotRegisteredExtraError(".register(jhu_data, extras=[...]) with extra datasets")
        # Get all subset
        df = self._extra_df.copy()
        # Remove columns which is included in the main datasets
        unused_set = set(self._main_df.columns) - {self.DATE}
        df = df.loc[:, ~df.columns.isin(unused_set)]
        # Data cleaning
        df = df.set_index(self.DATE).resample("D").last()
        df = df.fillna(method="ffill").fillna(0)
        # Subsetting by dates
        df = df.loc[self._first_date: self._last_date]
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
            NotRegisteredMainError: JHUData was not registered
            SubsetNotFoundError: failed in subsetting because of lack of data
            NotRegisteredExtraError: @extras is True and no extra datasets were registered
            ValueError: both of @main and @extras were False

        Returns:
            pandas.DataFrame:
                Index
                    reset index
                Columns:
                    - Date(pd.Timestamp): Observation date
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
            NotRegisteredMainError: JHUData was not registered
            SubsetNotFoundError: failed in subsetting because of lack of data
            NotRegisteredExtraError: @extras is True and no extra datasets were registered
            ValueError: both of @main and @extras were False, or both of @past and @future were False

        Returns:
            pandas.DataFrame:
                Index
                    reset index
                Columns:
                    - Date(pd.Timestamp): Observation date
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
            return df.loc[:self._today].reset_index()
        if future:
            return df.loc[self._today + timedelta(days=1):].reset_index()

    def records_all(self):
        """
        Return registered all records of the datasets as a dataframe.

        Raises:
            NotRegisteredMainError: JHUData was not registered
            SubsetNotFoundError: failed in subsetting because of lack of data

        Returns:
            pandas.DataFrame:
                Index
                    reset index
                Columns:
                    - Date(pd.Timestamp): Observation date
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

    def estimate_delay(self, indicator, target, min_size=7, use_difference=False, delay_name="Period Length"):
        """
        Estimate the average day [days] between the indicator and the target.
        We assume that the indicator impact on the target value with delay.
        All results will be returned with a dataframe.

        Args:
            indicator (str): indicator name, a column of any registered datasets
            target (str): target name, a column of any registered datasets
            min_size (int): minimum size of the delay period
            use_difference (bool): if True, use first discrete difference of target
            delay_name (str): column name of delay in the output dataframe

        Raises:
            NotRegisteredMainError: JHUData was not registered
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
        if use_difference:
            record_df[target] = record_df[target].diff()
        pivot_df = record_df.pivot_table(values=indicator, index=target)
        run_df = pivot_df.copy()
        # Convert index (target) to serial numbers
        serial_df = pd.DataFrame(np.arange(1, run_df.index.max() + 1, 1))
        serial_df.index += 1
        run_df = run_df.join(serial_df, how="outer")
        series = run_df.reset_index(drop=True).iloc[:, 0].dropna()
        # Detection with Ruptures using indicator values
        warnings.simplefilter("ignore", category=RuntimeWarning)
        algorithm = rpt.Pelt(model="rbf", jump=1, min_size=min_size)
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

    @classmethod
    def _ensure_date(cls, target, name="date", default=None):
        """
        Ensure the format of the string.

        Args:
            target (str or pandas.Timestamp): string to ensure
            name (str): argument name of the string
            default (pandas.Timestamp or None): default value to return

        Returns:
            pandas.Timestamp or None: as-is the target or default value
        """
        if target is None:
            return default
        if isinstance(target, pd.Timestamp):
            return target.replace(hour=0, minute=0, second=0, microsecond=0)
        try:
            return pd.to_datetime(target).replace(hour=0, minute=0, second=0, microsecond=0)
        except ValueError as e:
            raise ValueError(f"{name} was not recognized as a date, {target} was applied.") from e
