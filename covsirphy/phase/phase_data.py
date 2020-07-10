#!/usr/bin/env python
# -*- coding: utf-8 -*-

from covsirphy.cleaning.word import Word


class PhaseData(Word):
    """
    Basic class to create dataset for Trend/ODE analysis.

    Args:
        clean_df (pandas.DataFrame): cleaned data

            Index:
                reset index
            Columns:
                - Date (pd.TimeStamp): Observation date
                - Country (str): country/region name
                - Province (str): province/prefecture/sstate name
                - Confirmed (int): the number of confirmed cases
                - Infected (int): the number of currently infected cases
                - Fatal (int): the number of fatal cases
                - Recovered (int): the number of recovered cases

        country (str): country name
        province (str): province name
    """

    def __init__(self, clean_df, country=None, province=None):
        df = self.validate_dataframe(
            clean_df, name="clean_df", columns=self.NLOC_COLUMNS
        )
        self.country = country
        self.province = province
        df[self.COUNTRY] = self.country
        df[self.PROVINCE] = self.province
        df = self._set_place(
            df, country=country, province=province
        )
        self.all_df = self._groupby_date(df)

    def _set_place(self, clean_df, country=None, province=None):
        """
        Args:
            clean_df (pandas.DataFrame): cleaned data

                Index:
                    reset index
                Columns:
                    - Date (pd.TimeStamp): Observation date
                    - Country (str): country/region name
                    - Province (str): province/prefecture/sstate name
                    - Confirmed (int): the number of confirmed cases
                    - Infected (int): the number of currently infected cases
                    - Fatal (int): the number of fatal cases
                    - Recovered (int): the number of recovered cases
            country (str): country name
            province (str): province name

        Returns:
            (pandas.DataFrame):
                Index:
                    reset index
                Columns:
                    - Date (pd.TimeStamp): Observation date
                    - Country (str): country/region name
                    - Province (str): province/prefecture/sstate name
                    - Confirmed (int): the number of confirmed cases
                    - Infected (int): the number of currently infected cases
                    - Fatal (int): the number of fatal cases
                    - Recovered (int): the number of recovered cases
        """
        df = clean_df.copy()
        c, p = country, province
        if c is None:
            if p is not None:
                return df
            s = "@country must be defined when @province is not None."
            raise Exception(s)
        if p is None:
            return df.loc[df[self.COUNTRY] == c, :]
        return df.loc[(df[self.COUNTRY] == c) & (df[self.PROVINCE] == p), :]

    def _groupby_date(self, cleaned_df):
        """
        Grouping by date.

        Args:
            clean_df (pandas.DataFrame): cleaned data

                Index:
                    reset index
                Columns:
                    - Date (pd.TimeStamp): Observation date
                    - Country (str): country/region name
                    - Province (str): province/prefecture/sstate name
                    - Confirmed (int): the number of confirmed cases
                    - Infected (int): the number of currently infected cases
                    - Fatal (int): the number of fatal cases
                    - Recovered (int): the number of recovered cases

        Returns:
            (pandas.DataFrame):
                Index:
                    - Date (pd.TimeStamp): Observation date
                Columns:
                    - Confirmed (int): the number of confirmed cases
                    - Infected (int): the number of currently infected cases
                    - Fatal (int): the number of fatal cases
                    - Recovered (int): the number of recovered cases
        """
        df = cleaned_df.copy()
        df = df.drop([self.COUNTRY, self.PROVINCE], axis=1)
        df = df.groupby(self.DATE).sum()
        return df

    def _make(self, grouped_df):
        """
        Make a dataset using the cleaned dataset.
        This method must be overwritten in child class.

        Args:
            grouped_df (pandas.DataFrame): cleaned data grouped by Date

                Index:
                    - Date (pd.TimeStamp): Observation date
                Columns:
                    - Confirmed (int): the number of confirmed cases
                    - Infected (int): the number of currently infected cases
                    - Fatal (int): the number of fatal cases
                    - Recovered (int): the number of recovered cases

        Returns:
            (pandas.DataFrame)
        """
        df = self.validate_dataframe(
            grouped_df,
            name="grouped_df", time_index=True, columns=self.VALUE_COLUMNS
        )
        return df

    def make(self, start_date=None, end_date=None):
        """
        Make a dataset using the cleaned dataset.
        This method must be overwritten in child class.

        Args:
            start_date (str): start date, like 22Jan2020
            end_date (str): end date, like 01Feb2020

        Returns:
            (pandas.DataFrame)
        """
        df = self.subset(start_date=start_date, end_date=end_date)
        return self._make(df)

    def subset(self, start_date, end_date):
        """
        Return the subset of the data with start/end date.

        Args:
            start_date (str): start date, like 22Jan2020
            end_date (str): end date, like 01Feb2020

        Returns:
            (pandas.DataFrame):
                Index:
                    - Date (pd.TimeStamp): Observation date
                Columns:
                    - Confirmed (int): the number of confirmed cases
                    - Infected (int): the number of currently infected cases
                    - Fatal (int): the number of fatal cases
                    - Recovered (int): the number of recovered cases
        """
        df = self.all_df.copy()
        series = df.index.copy()
        # Start/end date
        start_obj = self.to_date_obj(date_str=start_date, default=series.min())
        end_obj = self.to_date_obj(date_str=end_date, default=series.max())
        # Subset
        df = df.loc[(start_obj <= series) & (series <= end_obj), :]
        df = df.reset_index().groupby(self.DATE).last()
        return df
