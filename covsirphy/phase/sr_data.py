#!/usr/bin/env python
# -*- coding: utf-8 -*-

from covsirphy.phase.phase_data import PhaseData


class SRData(PhaseData):
    """
    Create dataset for S-R trend analysis.
    """

    def __init__(self, clean_df, country=None, province=None):
        super().__init__(clean_df, country=country, province=province)

    def _make(self, grouped_df, population):
        """
        Make dataset for S-R trend analysis.

        Args:
            grouped_df (pandas.DataFrame): cleaned data grouped by Date

                Index:
                    Date (pd.TimeStamp): Observation date
                Columns:
                    - Confirmed (int): the number of confirmed cases
                    - Infected (int): the number of currently infected cases
                    - Fatal (int): the number of fatal cases
                    - Recovered (int): the number of recovered cases

            population (int): total population in the place

        Returns:
            (pandas.DataFrame)
                Index:
                    Date (pd.TimeStamp): Observation date
                Columns:
                    - Recovered: The number of recovered cases
                    - Susceptible_actual: Actual data of Susceptible
        """
        df = grouped_df.copy()
        df[self.COUNTRY] = self.country
        df[self.PROVINCE] = self.province
        df = self.validate_dataframe(
            df, name="grouped_df", time_index=True, columns=self.VALUE_COLUMNS
        )
        df[f"{self.S}{self.A}"] = population - df[self.C]
        df = df.loc[:, [self.R, f"{self.S}{self.A}"]]
        return df

    def make(self, population, start_date=None, end_date=None):
        """
        Make dataset for S-R trend analysis.

        Args:
            population (int): total population in the place
            start_date (str or None): start date, like 22Jan2020
            end_date (str or None): end date, like 01Feb2020

        Notes:
            - When @start_date is None, the first date of the records will be used
            - When @end_date is None, the last date of the records will be used

        Returns:
            (pandas.DataFrame)
                Index:
                    Date (pd.TimeStamp): Observation date
                Columns:
                    - Recovered (int): The number of recovered cases
                    - Susceptible_actual (int): Actual values of Susceptible (> 0)
        """
        df = self.all_df.copy()
        series = df.index.copy()
        # Start/end date
        start_obj = self.to_date_obj(date_str=start_date, default=series.min())
        end_obj = self.to_date_obj(date_str=end_date, default=series.max())
        # Subset
        df = df.loc[(start_obj <= series) & (series <= end_obj), :]
        df = df.loc[df[self.R] > 0, :]
        return self._make(df, population)
