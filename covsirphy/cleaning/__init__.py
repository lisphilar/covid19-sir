#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd


class CleaningBase(object):
    """
    Basic class for data cleaning.
    """

    def __init__(self, filename):
        """
        @filename <str>: CSV filename of the dataset
        """
        self._raw = pd.read_csv(filename)

    @property
    def raw(self):
        """
        Return the raw data.
        @return <pd.DataFrame>
        """
        return self._raw
