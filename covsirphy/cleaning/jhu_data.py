#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from covsirphy.util.error import SubsetNotFoundError, deprecate
from covsirphy.cleaning.cbase import CleaningBase
from covsirphy.cleaning.country_data import CountryData
from covsirphy.cleaning.jhu_complement import JHUDataComplementHandler


class JHUData(CleaningBase):
    """
    Data cleaning of JHU-style dataset.

    Args:
        filename (str): CSV filename of the dataset
        citation (str): citation
    """

    def __init__(self, filename, citation=None):
        super().__init__(filename, citation)
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
            kwargs: keword arguments will be ignored.

        Returns:
            pandas.DataFrame
                Index
                    reset index
                Columns
                    - Date (pd.TimeStamp): Observation date
                    - Country (pandas.Category): country/region name
                    - Province (pandas.Category): province/prefecture/state name
                    - Confirmed (int): the number of confirmed cases
                    - Infected (int): the number of currently infected cases
                    - Fatal (int): the number of fatal cases
                    - Recovered (int): the number of recovered cases

        Note:
            Cleaning method is defined by self._cleaning() method.
        """
        if "population" in kwargs.keys():
            raise ValueError(
                "@population was removed in JHUData.cleaned(). Please use JHUData.subset()")
        return self._cleaned_df.loc[:, self.COLUMNS]

    def _cleaning(self):
        """
        Perform data cleaning of the raw data.
        This method overwrite super()._cleaning() method.

        Returns:
            pandas.DataFrame
                Index
                    reset index
                Columns
                    - Date (pd.TimeStamp): Observation date
                    - ISO3 (str): ISO3 code
                    - Country (pandas.Category): country/region name
                    - Province (pandas.Category): province/prefecture/state name
                    - Confirmed (int): the number of confirmed cases
                    - Infected (int): the number of currently infected cases
                    - Fatal (int): the number of fatal cases
                    - Recovered (int): the number of recovered cases
        """
        df = super()._cleaning()
        # Rename the columns
        df = df.rename(
            {
                "ObservationDate": self.DATE,
                "Country/Region": self.COUNTRY,
                "Province/State": self.PROVINCE,
                "Deaths": self.F,
            },
            axis=1
        )
        # Confirm the expected columns are in raw data
        expected_cols = [
            self.DATE, self.ISO3, self.COUNTRY, self.PROVINCE, self.C, self.F, self.R
        ]
        self._ensure_dataframe(df, name="the raw data", columns=expected_cols)
        # Datetime columns
        df[self.DATE] = pd.to_datetime(df[self.DATE])
        # Country
        df[self.COUNTRY] = df[self.COUNTRY].replace(
            {
                # COD
                "Congo, the Democratic Republic of the": "Democratic Republic of the Congo",
                # COG
                "Congo": "Republic of the Congo",
                # South Korea
                "Korea, South": "South Korea",
            }
        )
        # Province
        df[self.PROVINCE] = df[self.PROVINCE].fillna(self.UNKNOWN)
        # Set 'Others' as the country name of cruise ships
        ships = ["Diamond Princess", "Costa Atlantica", "Grand Princess", "MS Zaandam"]
        for ship in ships:
            df.loc[df[self.COUNTRY] == ship, [self.COUNTRY, self.PROVINCE]] = [self.OTHERS, ship]
        # Values
        df = df.fillna(method="ffill").fillna(0)
        df[self.CI] = df[self.C] - df[self.F] - df[self.R]
        df[self.VALUE_COLUMNS] = df[self.VALUE_COLUMNS].astype(np.int64)
        df = df.loc[:, [self.ISO3, *self.COLUMNS]].reset_index(drop=True)
        # As country level data in China, use the total values of provinces
        p_chn_df = df.loc[
            (df[self.COUNTRY] == "China") & (df[self.PROVINCE] != self.UNKNOWN)]
        p_chn_df = p_chn_df.groupby(self.DATE).sum().reset_index()
        p_chn_df.insert(0, self.COUNTRY, "China")
        p_chn_df.insert(0, self.PROVINCE, self.UNKNOWN)
        p_chn_df[self.ISO3] = self.country_to_iso3(
            country="China", check_data=False)
        without_c_chn_df = df.loc[
            (df[self.COUNTRY] != "China") | (df[self.PROVINCE] != self.UNKNOWN)]
        df = pd.concat([without_c_chn_df, p_chn_df], ignore_index=True)
        # Update data types to reduce memory
        df[self.AREA_ABBR_COLS] = df[self.AREA_ABBR_COLS].astype("category")
        return df

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
        # Read new dataset
        country = country_data.country
        new = country_data.cleaned().loc[:, self.COLUMNS]
        new[self.ISO3] = self.country_to_iso3(country)
        # Remove the data in the country from JHU dataset
        df = self._cleaned_df.copy()
        df = df.loc[df[self.COUNTRY] != country]
        # Combine JHU data and the new data
        df = pd.concat([df, new], axis=0, sort=False)
        # Update data types to reduce memory
        df[self.AREA_ABBR_COLS] = df[self.AREA_ABBR_COLS].astype("category")
        self._cleaned_df = df.copy()
        # Citation
        self._citation += f"\n{country_data.citation}"
        return self

    def _subset(self, country, province, start_date, end_date):
        """
        Return the subset of dataset or empty dataframe (when no records were found).

        Args:
            country (str): country name or ISO3 code
            province (str or None): province name
            start_date (str or None): start date, like 22Jan2020
            end_date (str or None): end date, like 01Feb2020

        Returns:
            pandas.DataFrame
                Index
                    reset index
                Columns
                    - Date (pd.TimeStamp): Observation date
                    - Confirmed (int): the number of confirmed cases
                    - Infected (int): the number of currently infected cases
                    - Fatal (int): the number of fatal cases
                    - Recovered (int): the number of recovered cases
        """
        try:
            return super().subset(
                country=country, province=province, start_date=start_date, end_date=end_date)
        except SubsetNotFoundError:
            return pd.DataFrame(columns=self.NLOC_COLUMNS)

    def _calculate_susceptible(self, subset_df, population):
        """
        Return the subset of dataset.

        Args:
            subset_df (pandas.DataFrame)
                Index
                    reset index
                Columns
                    - Date (pd.TimeStamp): Observation date
                    - Confirmed (int): the number of confirmed cases
                    - Infected (int): the number of currently infected cases
                    - Fatal (int): the number of fatal cases
                    - Recovered (int): the number of recovered cases
            population (int or None): population value

        Returns:
            pandas.DataFrame
                Index
                    reset index
                Columns
                    - Date (pd.TimeStamp): Observation date
                    - Confirmed (int): the number of confirmed cases
                    - Infected (int): the number of currently infected cases
                    - Fatal (int): the number of fatal cases
                    - Recovered (int): the number of recovered cases
                    - Susceptible (int): the number of susceptible cases, if calculated

        Note:
            If @population is not None, the number of susceptible cases will be calculated.
        """
        if population is None:
            return subset_df
        subset_df.loc[:, self.S] = population - subset_df.loc[:, self.C]
        return subset_df[self.SUB_COLUMNS]

    def subset(self, country, province=None, start_date=None, end_date=None, population=None):
        """
        Return the subset of dataset with Recovered > 0.

        Args:
            country (str): country name or ISO3 code
            province (str or None): province name
            start_date (str or None): start date, like 22Jan2020
            end_date (str or None): end date, like 01Feb2020
            population (int or None): population value

        Returns:
            pandas.DataFrame
                Index
                    reset index
                Columns
                    - Date (pd.TimeStamp): Observation date
                    - Confirmed (int): the number of confirmed cases
                    - Infected (int): the number of currently infected cases
                    - Fatal (int): the number of fatal cases
                    - Recovered (int): the number of recovered cases (> 0)
                    - Susceptible (int): the number of susceptible cases, if calculated

        Note:
            If @population is not None, the number of susceptible cases will be calculated.
        """
        country_alias = self.ensure_country_name(country)
        # Subset with area, start/end date and calculate Susceptible
        subset_df = self._subset(
            country=country, province=province, start_date=start_date, end_date=end_date)
        if subset_df.empty:
            raise SubsetNotFoundError(
                country=country, country_alias=country_alias, province=province,
                start_date=start_date, end_date=end_date)
        # Calculate Susceptible
        df = self._calculate_susceptible(subset_df, population)
        # Select records where Recovered > 0
        df = df.loc[df[self.R] > 0, :].reset_index(drop=True)
        if df.empty:
            raise SubsetNotFoundError(
                country=country, country_alias=country_alias, province=province,
                start_date=start_date, end_date=end_date, message="with 'Recovered > 0'") from None
        return df

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
                    Date (pd.TimeStamp): Observation date
                Columns
                    - Recovered (int): the number of recovered cases (> 0)
                    - Susceptible (int): the number of susceptible cases

        Note:
            @population must be specified.
            Records with Recovered > 0 will be used.
        """
        population = self._ensure_population(population)
        subset_df = self.subset(
            country=country, province=province,
            start_date=start_date, end_date=end_date, population=population)
        return subset_df.set_index(self.DATE).loc[:, [self.R, self.S]]

    @classmethod
    def from_dataframe(cls, dataframe, directory="input"):
        """
        Create JHUData instance using a pandas dataframe.

        Args:
            dataframe (pd.DataFrame): cleaned dataset
            directory (str): directory to save geometry information (for .map() method)

        Returns:
            covsirphy.JHUData: JHU-style dataset
        """
        instance = cls(filename=None)
        instance.directory = str(directory)
        instance._cleaned_df = cls._ensure_dataframe(
            dataframe, name="dataframe", columns=cls.COLUMNS)
        return instance

    def total(self):
        """
        Calculate total number of cases and rates.

        Returns:
            pandas.DataFrame: group-by Date, sum of the values

                Index
                    Date (pandas.TimeStamp): Observation date
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
        df = df.loc[df[self.PROVINCE] == self.UNKNOWN]
        df = df.groupby(self.DATE).sum()
        total_series = df.loc[:, self.C]
        r_cols = self.RATE_COLUMNS[:]
        df[r_cols[0]] = df[self.F] / total_series
        df[r_cols[1]] = df[self.R] / total_series
        df[r_cols[2]] = df[self.F] / (df[self.F] + df[self.R])
        # Set the final date of the records
        raw_df = self._raw.rename({"ObservationDate": self.DATE}, axis=1)
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
        df = df.loc[df[self.PROVINCE] == self.UNKNOWN]
        # All countries
        all_set = set((df[self.COUNTRY].unique()))
        # Selectable countries without complement
        raw_ok_set = set(df.loc[df[self.R] > 0, self.COUNTRY].unique())
        if not complement:
            return sorted(raw_ok_set)
        # Selectable countries
        comp_ok_list = [
            country for country in all_set - raw_ok_set
            if not self.subset_complement(country=country, **kwargs)[0].empty]
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
        df = df.loc[df[self.PROVINCE] == self.UNKNOWN]
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
        df = df.loc[df[self.PROVINCE] == self.UNKNOWN]
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
                        - Date(pd.TimeStamp): Observation date
                        - Confirmed(int): the number of confirmed cases
                        - Infected(int): the number of currently infected cases
                        - Fatal(int): the number of fatal cases
                        - Recovered (int): the number of recovered cases ( > 0)
                        - Susceptible(int): the number of susceptible cases, if calculated
                str or bool: kind of complement or False

        Note:
            If @population is not None, the number of susceptible cases will be calculated.
        """
        # Subset with area, start/end date and calculate Susceptible
        country_alias = self.ensure_country_name(country)
        subset_df = self._subset(
            country=country, province=province, start_date=start_date, end_date=end_date)
        if subset_df.empty:
            raise SubsetNotFoundError(
                country=country, country_alias=country_alias, province=province,
                start_date=start_date, end_date=end_date) from None
        # Complement, if necessary
        self._recovery_period = self._recovery_period or self.calculate_recovery_period()
        handler = JHUDataComplementHandler(
            recovery_period=self._recovery_period, **kwargs)
        df, status, _ = handler.run(subset_df)
        # Calculate Susceptible
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
                        - Date(pd.TimeStamp): Observation date
                        - Confirmed(int): the number of confirmed cases
                        - Infected(int): the number of currently infected cases
                        - Fatal(int): the number of fatal cases
                        - Recovered (int): the number of recovered cases ( > 0)
                        - Susceptible(int): the number of susceptible cases, if calculated
                str or bool: kind of complement or False

        Note:
            - If @ population is not None, the number of susceptible cases will be calculated.
            - If necessary and @auto_complement is True, complement recovered data.
        """
        country_alias = self.ensure_country_name(country)
        subset_arg_dict = {
            "country": country, "province": province,
            "start_date": start_date, "end_date": end_date, "population": population,
        }
        if auto_complement:
            df, is_complemented = self.subset_complement(
                **subset_arg_dict, **kwargs)
            if not df.empty:
                return (df, is_complemented)
        try:
            return (self.subset(**subset_arg_dict), False)
        except ValueError:
            raise SubsetNotFoundError(
                country=country, country_alias=country_alias, province=province,
                start_date=start_date, end_date=end_date, message="with 'Recovered > 0'") from None

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
            country = [
                c for c in self._cleaned_df[self.COUNTRY].unique() if c != "Others"]
        province = province or self.UNKNOWN
        if not isinstance(country, str) and province != self.UNKNOWN:
            raise ValueError(
                "@province cannot be specified when @country is not a string.")
        if not isinstance(country, list):
            country = [country]
        # Create complement handler
        handler = JHUDataComplementHandler(
            recovery_period=self._recovery_period, **kwargs)
        # Check each country
        complement_df = pd.DataFrame(
            columns=[
                self.COUNTRY, self.PROVINCE, *JHUDataComplementHandler.SHOW_COMPLEMENT_FULL_COLS])
        complement_df.set_index(self.COUNTRY, inplace=True)
        for cur_country in country:
            subset_df = self._subset(
                country=cur_country, province=province, start_date=start_date, end_date=end_date)
            if subset_df.empty:
                raise SubsetNotFoundError(
                    country=cur_country, province=province, start_date=start_date, end_date=end_date)
            *_, complement_dict = handler.run(subset_df)
            complement_dict_values = pd.Series(
                complement_dict.values(), dtype=bool).values
            complement_df.loc[cur_country] = [
                province, *complement_dict_values]
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
