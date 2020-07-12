#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
from covsirphy.cleaning.jhu_data import JHUData


class ExampleData(JHUData):
    """
    Example dataset as a child class of JHUData.

    Args:
        clean_df (pandas.DataFrame): cleaned data

            Index:
                - reset index
            Columns:
                - Date (pd.TimeStamp): Observation date
                - Country (str): country/region name
                - Province (str): province/prefecture/sstate name
                - Confirmed (int): the number of confirmed cases
                - Infected (int): the number of currently infected cases
                - Fatal (int): the number of fatal cases
                - Recovered (int): the number of recovered cases
    """

    def __init__(self, clean_df=None):
        if clean_df is None:
            clean_df = pd.DataFrame(columns=self.COLUMNS)
        self._raw = clean_df.copy()
        self._cleaned_df = clean_df.copy()
        self._citation = str()
