#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import pytest
from covsirphy import LinePlot, line_plot, Term, UnExecutedError


class TestLinePlot(object):
    def test_plot(self, jhu_data, imgfile):
        df = jhu_data.subset(country="Japan").set_index(Term.DATE)
        with LinePlot(filename=imgfile) as lp:
            lp.plot(data=df)
        with LinePlot(filename=imgfile) as lp:
            lp.plot(data=df[Term.C])
        with LinePlot(filename=imgfile) as lp:
            lp.plot(data=df, colormap="rainbow")
        with LinePlot(filename=imgfile) as lp:
            lp.plot(data=df, color_dict={Term.C: "blue"})
        with pytest.raises(ValueError):
            with LinePlot(filename=imgfile) as lp:
                lp.plot(data=df, colormap="unknown")

    def test_axis(self, jhu_data, imgfile):
        df = jhu_data.subset(country="Japan").set_index(Term.DATE)
        with LinePlot(filename=imgfile) as lp:
            lp.plot(data=df)
            lp.x_axis(x_logscale=True)
            lp.y_axis(y_logscale=True)
            lp.line(v=pd.Timestamp("01Jan2021"))
        with LinePlot(filename=imgfile) as lp:
            lp.plot(data=df)
            lp.y_axis(y_integer=True)
            lp.line(h=100_000)

    def test_legend(self, jhu_data, imgfile):
        df = jhu_data.subset(country="Japan").set_index(Term.DATE)
        with LinePlot(filename=imgfile) as lp:
            with pytest.raises(UnExecutedError):
                lp.legend()
            lp.plot(data=df)
            lp.legend_hide()
            lp.legend()

    def test_function(self, jhu_data, imgfile):
        df = jhu_data.subset(country="Japan").set_index(Term.DATE)
        line_plot(df=df, filename=imgfile, show_legend=True)
        line_plot(df=df, filename=imgfile, show_legend=False)
