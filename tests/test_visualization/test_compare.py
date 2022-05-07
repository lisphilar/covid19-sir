#!/usr/bin/env python
# -*- coding: utf-8 -*-

from covsirphy import compare_plot, Term


class TestComparePlot(object):
    def test_plot(self, jhu_data, imgfile):
        tokyo_df = jhu_data.subset(country="Japan", province="Tokyo")
        osaka_df = jhu_data.subset(country="Japan", province="Osaka")
        df = tokyo_df.merge(osaka_df, on=Term.DATE, suffixes=("_tokyo", "_osaka"))
        compare_plot(df, variables=[Term.CI, Term.F, Term.R], groups=["tokyo", "osaka"], filename=imgfile)
