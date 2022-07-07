#!/usr/bin/env python
# -*- coding: utf-8 -*-

from covsirphy import compare_plot


class TestComparePlot(object):
    def test_plot(self, japan_df, imgfile):
        df = japan_df.set_index("date")
        tokyo_df = df.loc[df["Prefecture"] == "Tokyo"].drop("Prefecture", axis=1)
        osaka_df = df.loc[df["Prefecture"] == "Osaka"].drop("Prefecture", axis=1)
        df = tokyo_df.merge(osaka_df, on="date", suffixes=("_tokyo", "_osaka"))
        compare_plot(df, variables=["Positive", "Fatal", "Discharged"], groups=["tokyo", "osaka"], filename=imgfile)
