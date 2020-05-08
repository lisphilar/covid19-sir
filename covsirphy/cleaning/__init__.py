#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd


class CleaningBase(object):
    """
    Basic class for data cleaning.
    """

    def __inut__(self, filename):
        """
        @filename <str>: CSV filename of the dataset
        """
        self.raw = pd.read_csv(filename)

    def raw(self):
        """
        Return the raw data.
        @return <pd.DataFrame>
        """
        return self.raw
