#!/usr/bin/env python
# -*- coding: utf-8 -*-

import contextlib
import numpy as np
import pandas as pd
from covsirphy.util.error import SubsetNotFoundError, deprecate
from covsirphy.util.validator import Validator
from covsirphy._deprecated.cbase import CleaningBase
from covsirphy._deprecated.country_data import CountryData
from covsirphy._deprecated.jhu_complement import JHUDataComplementHandler


class JHUData(CleaningBase):
    """
    Data cleaning of JHU-style dataset.

    Args:
        filename (str or None): CSV filename of the dataset
        data (pandas.DataFrame or None):
            Index
                reset index
            Columns
                - Date: Observation date
                - ISO3: ISO3 code
                - Country: country/region name
                - Province: province/prefecture/state name
                - Confirmed: the number of confirmed cases
                - Fatal: the number of fatal cases
                - Recovered: the number of recovered cases
                - Population: population values
        citation (str or None): citation or None (empty)

    Note:
        Either @filename (high priority) or @data must be specified.

    Note:
        The number of infected cases will be (re-)calculated when data cleaning automatically.
    """
    # For JHUData.from_dataframe()
    _RAW_COLS_DEFAULT = [
        CleaningBase.DATE, CleaningBase.ISO3, CleaningBase.COUNTRY, CleaningBase.PROVINCE,
        CleaningBase.C, CleaningBase.CI, CleaningBase.F, CleaningBase.R, CleaningBase.N
    ]

    @deprecate(old="JHUData", new="DataEngineer", version="2.24.0-xi")
    def __init__(self, filename=None, data=None, citation=None):
        variables = [self.C, self.CI, self.F, self.R, self.N]
        super().__init__(filename=filename, data=data, citation=citation, variables=variables)
        # Recovery period
        self._recovery_period = None

    @property
    def recovery_period(self):
        """
        int: expected value of recovery period [days]
        """
        self._recovery_period = self._recovery_period or self.calculate_recovery_period()
        return self._recovery_period

    @recovery_period.setter
    def recovery_period(self, value):
        self._recovery_period = self._ensure_natural_int(value)

    def cleaned(self, **kwargs):
        """
        Return the cleaned dataset.

        Args:
            kwargs: keyword arguments will be ignored

        Returns:
            pandas.DataFrame
                Index
                    reset index
                Columns
                    - Date (pandas.Timestamp): Observation date
                    - ISO3: ISO3 code
                    - Country (pandas.Category): country/region name
                    - Province (pandas.Category): province/prefecture/state name
                    - Confirmed (int): the number of confirmed cases
                    - Infected (int): the number of currently infected cases
                    - Fatal (int): the number of fatal cases
                    - Recovered (int): the number of recovered cases
                    - Population: population values
        """
        if "population" in kwargs.keys():
            raise ValueError(
                "@population was removed in JHUData.cleaned(). Please use JHUData.subset()")
        df = self._cleaned_df.copy()
        df[self.CI] = (df[self.C] - df[self.F] - df[self.R]).astype(np.int64)
        return df

    def _cleaning(self):
        """
        Perform data cleaning of the raw data.

        Returns:
            pandas.DataFrame
                Index
                    reset index
                Columns
                    - Date (pd.Timestamp): Observation date
                    - ISO3 (str): ISO3 code
                    - Country (pandas.Category): country/region name
                    - Province (pandas.Category): province/prefecture/state name
                    - Confirmed (int): the number of confirmed cases
                    - Infected (int): the number of currently infected cases
                    - Fatal (int): the number of fatal cases
                    - Recovered (int): the number of recovered cases
                    - Population (int): population values or 0 (when raw data is 0 values)
        """
        df = self._raw.loc[:, self._raw_cols]
        # Datetime columns
        df[self.DATE] = pd.to_datetime(df[self.DATE]).dt.round("D")
        with contextlib.suppress(TypeError):
            df[self.DATE] = df[self.DATE].dt.tz_convert(None)
        # Province
        df[self.PROVINCE] = df[self.PROVINCE].fillna(self.NA)
        # Values
        for col in [self.C, self.F, self.R, self.N]:
            df[col] = df.groupby([self.COUNTRY, self.PROVINCE])[col].ffill().fillna(0).astype(np.int64)
        # Calculate Infected
        df[self.CI] = (df[self.C] - df[self.F] - df[self.R]).astype(np.int64)
        # Update data types to reduce memory
        df[self.AREA_ABBR_COLS] = df[self.AREA_ABBR_COLS].astype("category")
        return df.loc[:, self._raw_cols]

    @deprecate("JHUData.replace()", version="2.21.0-xi-fu1")
    def replace(self, country_data):
        """
        Replace a part of cleaned dataset with a dataframe.

        Args:
            country_data (covsirphy.CountryData): dataset object of the country

        Returns:
            covsirphy.JHUData: self

        Note:
            Citation of the country data will be added to 'JHUData.citation' description.
        """
        self._ensure_instance(country_data, CountryData, name="country_data")
        df = self._cleaned_df.copy()
        # Read new dataset
        country = country_data.country
        new = country_data.cleaned()
        new[self.ISO3] = self.country_to_iso3(country)
        # Add population data
        new[self.N] = new.loc[:, self.N] if self.N in new else None
        new = new.set_index([self.COUNTRY, self.PROVINCE, self.DATE])
        new.update(df.set_index([self.COUNTRY, self.PROVINCE, self.DATE]).loc[:, [self.N]])
        new = new.reset_index().loc[:, self._raw_cols]
        # Calculate Infected
        new[self.CI] = (new[self.C] - new[self.F] - new[self.R]).astype(np.int64)
        # Remove the data in the country from JHU dataset
        df = df.loc[df[self.COUNTRY] != country]
        # Combine JHU data and the new data
        df = pd.concat([df, new], axis=0, sort=False)
        # Update data types to reduce memory
        df[self.AREA_ABBR_COLS] = df[self.AREA_ABBR_COLS].astype("category")
        self._cleaned_df = df.copy()
        # Citation
        self._citation += f"\n{country_data.citation}"
        return self

    def _calculate_susceptible(self, subset_df, population):
        """
        Return the subset of dataset.

        Args:
            subset_df (pandas.DataFrame)
                Index
                    reset index
                Columns
                    - Date (pd.Timestamp): Observation date
                    - Confirmed (int): the number of confirmed cases
                    - Infected (int): the number of currently infected cases
                    - Fatal (int): the number of fatal cases
                    - Recovered (int): the number of recovered cases
                    - Population (int): population values or 0 values (0 will be ignored)
            population (int or None): population value

        Returns:
            pandas.DataFrame
                Index
                    reset index
                Columns
                    - Date (pd.Timestamp): Observation date
                    - Confirmed (int): the number of confirmed cases
                    - Infected (int): the number of currently infected cases
                    - Fatal (int): the number of fatal cases
                    - Recovered (int): the number of recovered cases
                    - Susceptible (int): the number of susceptible cases, if calculated

        Note:
            If @population (high priority) is not None or population values are registered in subset,
            the number of susceptible cases will be calculated.
        """
        df = subset_df.copy()
        df[self.S] = (population or df[self.N]) - df[self.C]
        try:
            df[self.S] = df[self.S].astype(np.int64)
        except ValueError:
            return df.loc[:, [self.DATE, self.C, self.CI, self.F, self.R]]
        return df.loc[:, [self.DATE, self.C, self.CI, self.F, self.R, self.S]]

    def subset(self, country, province=None, start_date=None, end_date=None,
               population=None, recovered_min=1):
        """
        Return the subset of dataset.

        Args:
            country (str): country name or ISO3 code
            province (str or None): province name
            start_date (str or None): start date, like 22Jan2020
            end_date (str or None): end date, like 01Feb2020
            population (int or None): population value
            recovered_min (int): minimum number of recovered cases records must have

        Returns:
            pandas.DataFrame
                Index
                    reset index
                Columns
                    - Date (pd.Timestamp): Observation date
                    - Confirmed (int): the number of confirmed cases
                    - Infected (int): the number of currently infected cases
                    - Fatal (int): the number of fatal cases
                    - Recovered (int): the number of recovered cases (> 0)
                    - Susceptible (int): the number of susceptible cases, if calculated

        Note:
            If @population (high priority) is not None or population values are registered in subset,
            the number of susceptible cases will be calculated.
        """
        country_alias = self.ensure_country_name(country)
        # Subset with area, start/end date
        try:
            subset_df = super().subset(
                country=country, province=province, start_date=start_date, end_date=end_date)
        except SubsetNotFoundError:
            raise SubsetNotFoundError(
                country=country, country_alias=country_alias, province=province,
                start_date=start_date, end_date=end_date) from None
        # Calculate Susceptible
        df = self._calculate_susceptible(subset_df, population)
        # Select records where Recovered >= recovered_min
        recovered_min = self._ensure_natural_int(recovered_min, name="recovered_min", include_zero=True)
        df = df.loc[df[self.R] >= recovered_min, :].reset_index(drop=True)
        if df.empty:
            raise SubsetNotFoundError(
                country=country, country_alias=country_alias, province=province,
                start_date=start_date, end_date=end_date,
                details=f"with 'Recovered >= {recovered_min}'") from None
        return df

    @deprecate("JHUData.to_sr()", version="2.17.0-zeta")
    def to_sr(self, country, province=None,
              start_date=None, end_date=None, population=None):
        """
        Create Susceptible/Recovered dataset without complement.

        Args:
            country (str): country name
            province (str): province name
            start_date (str or None): start date, like 22Jan2020
            end_date (str or None): end date, like 01Feb2020
            population (int): population value

        Returns:
            pandas.DataFrame
                Index
                    Date (pd.Timestamp): Observation date
                Columns
                    - Recovered (int): the number of recovered cases (> 0)
                    - Susceptible (int): the number of susceptible cases

        Note:
            @population must be specified.
            Records with Recovered > 0 will be used.
        """
        population = Validator(population, "population").int(value_range=(1, None))
        subset_df = self.subset(
            country=country, province=province,
            start_date=start_date, end_date=end_date, population=population)
        return subset_df.set_index(self.DATE).loc[:, [self.R, self.S]]

    @classmethod
    @deprecate("JHUData.from_dataframe()", new="DataLoader.read_dataframe()",
               version="2.21.0-xi-fu1", ref="https://lisphilar.github.io/covid19-sir/markdown/LOADING.html")
    def from_dataframe(cls, dataframe, directory="input"):
        """
        Create JHUData instance using a pandas dataframe.

        Args:
            dataframe (pd.DataFrame): cleaned dataset
                Index
                    reset index
                Columns
                    - Date: Observation date
                    - ISO3: ISO3 code (optional)
                    - Country: country/region name
                    - Province: province/prefecture/state name
                    - Confirmed: the number of confirmed cases
                    - Infected: the number of currently infected cases
                    - Fatal: the number of fatal cases
                    - Recovered: the number of recovered cases
                    - Population: population values (optional)
            directory (str): directory to save geography information (for .map() method)

        Returns:
            covsirphy.JHUData: JHU-style dataset
        """
        df = cls._ensure_dataframe(dataframe, name="dataframe")
        df[cls.ISO3] = df[cls.ISO3] if cls.ISO3 in df else cls.NA
        df[cls.N] = df[cls.N] if cls.N in df else 0
        instance = cls()
        instance.directory = str(directory)
        instance._cleaned_df = cls._ensure_dataframe(df, name="dataframe", columns=cls._RAW_COLS_DEFAULT)
        return instance

    def total(self):
        """
        Calculate total number of cases and rates.

        Returns:
            pandas.DataFrame: group-by Date, sum of the values

                Index
                    Date (pandas.Timestamp): Observation date
                Columns
                    - Confirmed (int): the number of confirmed cases
                    - Infected (int): the number of currently infected cases
                    - Fatal (int): the number of fatal cases
                    - Recovered (int): the number of recovered cases
                    - Fatal per Confirmed (int)
                    - Recovered per Confirmed (int)
                    - Fatal per (Fatal or Recovered) (int)
        """
        df = self._cleaned_df.copy()
        df = df.loc[df[self.PROVINCE] == self.NA]
        df = df.groupby(self.DATE).sum()
        total_series = df.loc[:, self.C]
        r_cols = self.RATE_COLUMNS[:]
        df[r_cols[0]] = df[self.F] / total_series
        df[r_cols[1]] = df[self.R] / total_series
        df[r_cols[2]] = df[self.F] / (df[self.F] + df[self.R])
        # Set the final date of the records
        raw_df = self._raw.copy()
        final_date = pd.to_datetime(raw_df[self.DATE]).dt.date.max()
        df = df.loc[df.index.date <= final_date]
        return df.loc[:, [*self.VALUE_COLUMNS, *r_cols]]

    def countries(self, complement=True, **kwargs):
        """
        Return names of countries where records.

        Args:
            complement (bool): whether say OK for complement or not
            interval (int): expected update interval of the number of recovered cases [days]
            kwargs: the other keyword arguments of JHUData.subset_complement()

        Returns:
            list[str]: list of country names
        """
        df = self._cleaned_df.copy()
        df = df.loc[df[self.PROVINCE] == self.NA]
        # All countries
        all_set = set((df[self.COUNTRY].unique()))
        # Selectable countries without complement
        raw_ok_set = set(df.loc[df[self.R] > 0, self.COUNTRY].unique())
        if not complement:
            return sorted(raw_ok_set)
        # Selectable countries
        comp_ok_list = []
        for country in all_set - raw_ok_set:
            with contextlib.suppress(SubsetNotFoundError):
                self.subset_complement(country=country, **kwargs)
                comp_ok_list.append(country)
        return sorted(raw_ok_set | set(comp_ok_list))

    @deprecate("JHUData.calculate_closing_period()")
    def calculate_closing_period(self):
        """
        Calculate mode value of closing period, time from confirmation to get outcome.

        Returns:
            int: closing period [days]

        Note:
            If no records we can use for calculation were registered, 12 [days] will be applied.
        """
        # Get cleaned dataset at country level
        df = self._cleaned_df.copy()
        df = df.loc[df[self.PROVINCE] == self.NA]
        # Select records of countries where recovered values are reported
        df = df.groupby(self.COUNTRY).filter(lambda x: x[self.R].sum() != 0)
        if df.empty:
            return 12
        # Total number of confirmed/closed cases of selected records
        df = df.groupby(self.DATE).sum()
        df[self.FR] = df[[self.F, self.R]].sum(axis=1)
        df = df.loc[:, [self.C, self.FR]]
        # Calculate how many days to confirmed, closed
        df = df.unstack().reset_index()
        df.columns = ["Variable", self.DATE, "Number"]
        df["Days"] = (df[self.DATE] - df[self.DATE].min()).dt.days
        df = df.pivot_table(values="Days", index="Number", columns="Variable")
        df = df.interpolate(limit_direction="both").fillna(method="ffill")
        df["Elapsed"] = df[self.FR] - df[self.C]
        df = df.loc[df["Elapsed"] > 0]
        # Calculate mode value of closing period
        return int(df["Elapsed"].mode().astype(np.int64).values[0])

    def calculate_recovery_period(self):
        """
        Calculate the median value of recovery period of all countries
        where recovered values are reported.

        Returns:
            int: recovery period [days]

        Note:
            If no records we can use for calculation were registered, 17 [days] will be applied.
        """
        default = 17
        # Get valid data for calculation
        df = self._cleaned_df.copy()
        df = df.loc[df[self.PROVINCE] == self.NA]
        df = df.groupby(self.COUNTRY).filter(lambda x: x[self.R].sum() != 0)
        # If no records were found the default value will be returned
        if df.empty:
            return default
        # Calculate median value of recovery period in all countries with valid data
        periods = [
            self._calculate_recovery_period_country(df, country)
            for country in df[self.COUNTRY].unique()
        ]
        valid_periods = list(filter(lambda x: x >= 0, periods))
        if not valid_periods:
            return default
        try:
            return int(pd.Series(valid_periods).median())
        except ValueError:
            return default

    def _calculate_recovery_period_country(self, valid_df, country, upper_limit_days=90,
                                           lower_limit_days=7, upper_percentage=0.5, lower_percentage=0.5):
        """
        Calculate mode value of recovery period in the country.
        If many mode values were found, mean value of mode values will be returned.

        Args:
            valid_df (pandas.DataFrame):
                Index
                    reset_index
                Columns
                    Date, Confirmed, Recovered, Fatal
            country(str): country name or ISO3 code
            upper_limit_days (int): maximum number of valid partial recovery periods [days]
            lower_limit_days (int): minimum number of valid partial recovery periods [days]
            upper_percentage (float): fraction of partial recovery periods with value greater than upper_limit_days
            lower_percentage (float): fraction of partial recovery periods with value less than lower_limit_days

        Returns:
            int: mode value of recovery period [days]
        """
        # Select country data
        df = valid_df.copy()
        df = df.loc[df[self.COUNTRY] == country].groupby(self.DATE).sum()
        # Calculate "Confirmed - Fatal"
        df["diff"] = df[self.C] - df[self.F]
        df = df.loc[:, ["diff", self.R]]
        # Calculate how many days passed to reach the number of cases
        df = df.unstack().reset_index()
        df.columns = ["Variable", "Date", "Number"]
        df["Days"] = (df[self.DATE] - df[self.DATE].min()).dt.days
        # Calculate recovery period (mode value because bimodal)
        df = df.pivot_table(values="Days", index="Number", columns="Variable")
        df = df.interpolate(limit_area="inside").dropna().astype(np.int64)
        df["Elapsed"] = df[self.R] - df["diff"]
        df = df.loc[df["Elapsed"] > 0]
        # Check partial recovery periods
        per_up = (df["Elapsed"] > upper_limit_days).sum()
        per_lw = (df["Elapsed"] < lower_limit_days).sum()
        if df.empty or per_up / len(df) >= upper_percentage or per_lw / len(df) >= lower_percentage:
            return -1
        return df["Elapsed"].mode().mean()

    def subset_complement(self, country, province=None,
                          start_date=None, end_date=None, population=None, **kwargs):
        """
        Return the subset of dataset and complement recovered data, if necessary.
        Records with Recovered > 0 will be selected.

        Args:
            country(str): country name or ISO3 code
            province(str or None): province name
            start_date(str or None): start date, like 22Jan2020
            end_date(str or None): end date, like 01Feb2020
            population(int or None): population value
            kwargs: keyword arguments of JHUDataComplementHandler(), control factors of complement

        Returns:
            tuple(pandas.DataFrame, str or bool):
                pandas.DataFrame:
                    Index
                        reset index
                    Columns
                        - Date(pd.Timestamp): Observation date
                        - Confirmed(int): the number of confirmed cases
                        - Infected(int): the number of currently infected cases
                        - Fatal(int): the number of fatal cases
                        - Recovered (int): the number of recovered cases ( > 0)
                        - Susceptible(int): the number of susceptible cases, if calculated
                str or bool: kind of complement or False

        Note:
            If @population (high priority) is not None or population values are registered in subset,
            the number of susceptible cases will be calculated.
        """
        # Subset with area
        country_alias = self.ensure_country_name(country)
        try:
            subset_df = super().subset(country=country, province=province, start_date=None, end_date=None)
        except SubsetNotFoundError:
            raise SubsetNotFoundError(
                country=country, country_alias=country_alias, province=province) from None
        # Complement, if necessary
        self._recovery_period = self._recovery_period or self.calculate_recovery_period()
        handler = JHUDataComplementHandler(recovery_period=self._recovery_period, **kwargs)
        df, status, _ = handler.run(subset_df)
        # Subsetting with dates
        if start_date is not None:
            df = df.loc[df[self.DATE] >= self._ensure_date(start_date, name="start_date")]
        if end_date is not None:
            df = df.loc[df[self.DATE] <= self._ensure_date(end_date, name="end_date")]
        if df.empty:
            raise SubsetNotFoundError(
                country=country, country_alias=country_alias, province=province,
                start_date=start_date, end_date=end_date) from None
        # Calculate Susceptible
        df.loc[:, self.N] = subset_df.loc[:, self.N]
        df = self._calculate_susceptible(df, population)
        # Kind of complement or False
        is_complemented = status or False
        # Select records where Recovered > 0
        df = df.loc[df[self.R] > 0, :].reset_index(drop=True)
        return (df, is_complemented)

    def records(self, country, province=None, start_date=None, end_date=None, population=None,
                auto_complement=True, **kwargs):
        """
        JHU-style dataset for the area from the start date to the end date.
        Records with Recovered > 0 will be selected.

        Args:
            country(str): country name or ISO3 code
            province(str or None): province name
            start_date(str or None): start date, like 22Jan2020
            end_date(str or None): end date, like 01Feb2020
            population(int or None): population value
            auto_complement (bool): if True and necessary, the number of cases will be complemented
            kwargs: the other arguments of JHUData.subset_complement()

        Raises:
            SubsetNotFoundError: failed in subsetting because of lack of data

        Returns:
            tuple(pandas.DataFrame, bool):
                pandas.DataFrame:

                    Index
                        reset index
                    Columns
                        - Date(pd.Timestamp): Observation date
                        - Confirmed(int): the number of confirmed cases
                        - Infected(int): the number of currently infected cases
                        - Fatal(int): the number of fatal cases
                        - Recovered (int): the number of recovered cases ( > 0)
                        - Susceptible(int): the number of susceptible cases, if calculated
                str or bool: kind of complement or False

        Note:
            If @population (high priority) is not None or population values are registered in subset,
            the number of susceptible cases will be calculated.

        Note:
            If necessary and @auto_complement is True, complement recovered data.
        """
        country_alias = self.ensure_country_name(country)
        subset_arg_dict = {
            "country": country, "province": province,
            "start_date": start_date, "end_date": end_date, "population": population,
        }
        if auto_complement:
            df, is_complemented = self.subset_complement(**subset_arg_dict, **kwargs)
            if not df.empty:
                return (df, is_complemented)
        try:
            return (self.subset(**subset_arg_dict), False)
        except ValueError:
            raise SubsetNotFoundError(
                country=country, country_alias=country_alias, province=province,
                start_date=start_date, end_date=end_date, details="with 'Recovered > 0'") from None

    def show_complement(self, country=None, province=None,
                        start_date=None, end_date=None, **kwargs):
        """
        To monitor effectivity and safety of complement on JHU subset,
        we need to know what kind of complement was done for JHU subset
        for each country (if country/countries specified) or for all countries.

        Args:
            country (str or list[str] or None): country/countries name or None (all countries)
            province(str or None): province name
            start_date(str or None): start date, like 22Jan2020
            end_date(str or None): end date, like 01Feb2020
            kwargs: keyword arguments of JHUDataComplementHandler(), control factors of complement

        Raises:
            ValueError: @province was specified when @country is not a string
            covsirphy.SubsetNotFoundError: No records were registered for the area/dates

        Returns:
            pandas.DataFrame

                Index
                    reset index
                Columns
                    - country (str): country name
                    - province (str): province name
                    - Monotonic_confirmed (bool): True if applied for confirmed cases or False otherwise
                    - Monotonic_fatal (bool): True if applied for fatal cases or False otherwise
                    - Monotonic_recovered (bool): True if applied for recovered or False otherwise
                    - Full_recovered (bool): True if applied for recovered or False otherwise
                    - Partial_recovered (bool): True if applied for recovered or False otherwise
        """
        self._recovery_period = self._recovery_period or self.calculate_recovery_period()
        # Area name
        if country is None:
            country = [c for c in self._cleaned_df[self.COUNTRY].unique() if c != "Others"]
        province = province or self.NA
        if not isinstance(country, str) and province != self.NA:
            raise ValueError("@province cannot be specified when @country is not a string.")
        if not isinstance(country, list):
            country = [country]
        # Create complement handler
        handler = JHUDataComplementHandler(recovery_period=self._recovery_period, **kwargs)
        # Check each country
        complement_df = pd.DataFrame(
            columns=[self.COUNTRY, self.PROVINCE, *JHUDataComplementHandler.SHOW_COMPLEMENT_FULL_COLS])
        complement_df.set_index(self.COUNTRY, inplace=True)
        for cur_country in country:
            try:
                subset_df = super().subset(
                    country=cur_country, province=province, start_date=start_date, end_date=end_date)
            except SubsetNotFoundError:
                raise SubsetNotFoundError(
                    country=cur_country, province=province,
                    start_date=start_date, end_date=end_date) from None
            * _, complement_dict = handler.run(subset_df)
            complement_dict_values = pd.Series(complement_dict.values(), dtype=bool).values
            complement_df.loc[cur_country] = [province, *complement_dict_values]
        return complement_df.reset_index()

    def map(self, country=None, variable="Confirmed", date=None, **kwargs):
        """
        Create global colored map to show the values.

        Args:
            country (str or None): country name or None (global map)
            variable (str): variable name to show
            date (str or None): date of the records or None (the last value)
            kwargs: arguments of ColoredMap() and ColoredMap.plot()

        Note:
            When @country is None, country level data will be shown on global map.
            When @country is a country name, province level data will be shown on country map.
        """
        # Date
        date_str = date or self.cleaned()[self.DATE].max().strftime(self.DATE_FORMAT)
        country_str = country or "Global"
        title = f"{country_str}: the number of {variable.lower()} cases on {date_str}"
        # Global map
        if country is None:
            return self._colored_map_global(
                variable=variable, title=title, date=date, **kwargs)
        # Country-specific map
        return self._colored_map_country(
            country=country, variable=variable, title=title, date=date, **kwargs)
