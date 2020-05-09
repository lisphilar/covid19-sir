#!/usr/bin/env python
# -*- coding: utf-8 -*-

from datetime import datetime
import pandas as pd


def select_area(ncov_df, group="Date",
                places=None, areas=None, excluded_places=None,
                start_date=None, end_date=None, date_format="%d%b%Y"):
    """
    Wrapper function of SelectArea class.
    """
    sel = SelectArea(ncov_df)
    sel.set_date(
        start_date=start_date,
        end_date=end_date,
        date_format=date_format
    )
    # Area
    if areas is not None:
        sel.set_areas("Area", included=areas)
    # Places
    if places is not None:
        before_df = sel.selected()
        df = pd.DataFrame(columns=before_df.columns)
        for (country, province) in places:
            if country is None:
                raise TypeError("country must be a string.")
            sel2 = SelectArea(before_df)
            sel2.set_areas("Country", included=[country])
            df2 = sel2.selected()
            if province is not None:
                sel2 = SelectArea(df2)
                sel2.set_areas("Province", included=[province])
                df2 = sel.selected()
            df = pd.concat([df, df2], axis=0)
        sel = SelectArea(df)
    # Excluded places
    if excluded_places is not None:
        df = sel.selected()
        for (country, province) in excluded_places:
            if country is None:
                raise TypeError("country must be a string.")
            sel2 = SelectArea(df)
            sel2.set_areas("Country", excluded=[country])
            df2 = sel2.selected()
            if province is not None:
                sel2 = SelectArea(df2)
                sel2.set_areas("Province", excluded=[province])
                df2 = sel.selected()
            df = df.loc[df.index != df2.index, :]
        sel = SelectArea(df)
    # Return
    try:
        sel.set_min(["Recovered", "Deaths"], 0)
    except KeyError:
        pass
    return sel.selected(group=group)


class SelectArea(object):
    """
    Class for select the data in an area.
    """

    def __init__(self, ncov_df):
        """
        @ncov_df <pd.DataFrame>: the clean data
        """
        self.all_df = ncov_df.copy()
        self.df = ncov_df.copy()

    def set_date(self, start_date=None, end_date=None, date_format="%d%b%Y"):
        """
        Set start/end date.
        @start_date <str>: start date or None
        @end_date <str>: end date or None
        @date_format <str>: format of @start_date and @end_date
        """
        df = self.df.copy()
        if start_date is not None:
            start_str = datetime.strptime(start_date, date_format)
            df = df.loc[df["Date"] >= start_str, :]
        if end_date is not None:
            end_str = datetime.strptime(end_date, date_format)
            df = df.loc[df["Date"] >= end_str, :]
        self.df = df.copy()

    def set_areas(self, column, included=None, excluded=None):
        """
        Select areas if the dataframe has the column.
        @column <str>: column name
        @included <list[str]>: list of areas to include
        @excluded <list[str]>: list of areas to exclude
        """
        if column not in self.df.columns:
            raise KeyError(f"The dataset does not have {column} column.")
        df = self.df.copy()
        if included is not None:
            df = df.loc[df[column].isin(included), :]
        if excluded is not None:
            df = df.loc[~df[column].isin(excluded), :]
        self.df = df.copy()

    def set_min(self, columns, total):
        """
        Select rows whose total is larger than @total.
        @columns <list[str]>: list of columns
        @total <float/int>: min value of the total value
        """
        df = self.df.copy()
        if not set(columns).issubset(set(df.columns)):
            diffs = list(set(columns) - set(df.columns))
            diff_str = ", ".join(diffs)
            diff_str += " columns" if len(diffs) > 1 else " column"
            raise KeyError(f"The dataset does not have {diff_str}.")
        df = df.loc[df[columns].sum(axis=1) > total, :]
        self.df = df.copy()

    def clear_selection(self):
        """
        Clear the selection.
        """
        self.df = self.all_df.copy()

    def selected(self, group=None):
        """
        Return the dataframe with selection.
        @group <str>: group to groupby
        """
        df = self.df.copy()
        if group is not None:
            df = df.groupby(group).sum().reset_index()
        if df.empty:
            raise Exception("The output dataframe is empty!")
        return df
