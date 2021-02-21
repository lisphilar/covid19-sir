#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pathlib import Path
import warnings
import matplotlib
import pytest
from covsirphy import VisualizeBase, ColoredMap
from covsirphy import jpn_map
from covsirphy import Term


class TestVisualizeBase(object):
    def test_base(self):
        warnings.filterwarnings("ignore", category=UserWarning)
        with VisualizeBase() as vb:
            with pytest.raises(NotImplementedError):
                vb.plot()

    def test_file(self, imgfile):
        with VisualizeBase(filename=imgfile):
            pass
        assert Path(imgfile).exists()

    def test_setting(self, imgfile):
        with VisualizeBase(filename=imgfile) as vb:
            assert not vb.title
            assert isinstance(vb.ax, matplotlib.axes.Axes)
            vb.ax = vb.ax
            vb.title = "title"
            vb.tick_params(
                labelbottom=False, labelleft=False, left=False, bottom=False)


class TestColoredMap(object):
    def test_directory(self, imgfile):
        with ColoredMap(filename=imgfile) as cm:
            cm.directory = "input"
            assert cm.directory == "input"

    @pytest.mark.parametrize("variable", ["Infected"])
    def test_global_country(self, imgfile, jhu_data, variable):
        df = jhu_data.cleaned()
        df = df.loc[df[Term.PROVINCE] == Term.UNKNOWN]
        df = df.groupby(Term.COUNTRY).last().reset_index()
        df.rename(columns={variable: "Value"}, inplace=True)
        with ColoredMap(filename=imgfile) as cm:
            cm.plot(data=df, level=Term.COUNTRY)
        # Not with log10 scale
        with ColoredMap(filename=imgfile) as cm:
            cm.plot(data=df, level=Term.COUNTRY, logscale=False)

    @pytest.mark.parametrize("variable", ["Infected"])
    def test_global_country_ununique(self, imgfile, jhu_data, variable):
        df = jhu_data.cleaned()
        df = df.loc[df[Term.PROVINCE] == Term.UNKNOWN]
        df.rename(columns={variable: "Value"}, inplace=True)
        with pytest.raises(ValueError):
            with ColoredMap(filename=imgfile) as cm:
                cm.plot(data=df, level=Term.COUNTRY)

    @pytest.mark.parametrize("country", ["Japan", "United States", "China"])
    @pytest.mark.parametrize("variable", ["Infected"])
    def test_in_a_country(self, imgfile, jhu_data, country, variable):
        df = jhu_data.cleaned()
        df = df.loc[df[Term.COUNTRY] == country]
        df = df.loc[df[Term.PROVINCE] != Term.UNKNOWN]
        df = df.groupby(Term.PROVINCE).last().dropna().reset_index()
        df.rename(columns={variable: "Value"}, inplace=True)
        with ColoredMap(filename=imgfile) as cm:
            cm.plot(data=df, level=Term.PROVINCE)

    @pytest.mark.parametrize("variable", ["Infected"])
    def test_in_a_country_unselected_country(self, imgfile, jhu_data, variable):
        df = jhu_data.cleaned()
        df = df.loc[df[Term.PROVINCE] != Term.UNKNOWN]
        df = df.groupby(Term.PROVINCE).last().dropna().reset_index()
        df.rename(columns={variable: "Value"}, inplace=True)
        with pytest.raises(ValueError):
            with ColoredMap(filename=imgfile) as cm:
                cm.plot(data=df, level=Term.PROVINCE)

    @pytest.mark.parametrize("country", ["Japan"])
    @pytest.mark.parametrize("variable", ["Infected"])
    def test_in_a_country_ununique(self, imgfile, jhu_data, country, variable):
        df = jhu_data.cleaned()
        df = df.loc[df[Term.COUNTRY] == country]
        df = df.loc[df[Term.PROVINCE] != Term.UNKNOWN]
        df = df.dropna().reset_index()
        df.rename(columns={variable: "Value"}, inplace=True)
        with pytest.raises(ValueError):
            with ColoredMap(filename=imgfile) as cm:
                cm.plot(data=df, level=Term.PROVINCE)


class TestJapanMap(object):

    @pytest.mark.parametrize("variable", ["Infected"])
    def test_japan_map_display(self, jhu_data, variable):
        warnings.filterwarnings("ignore", category=UserWarning)
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        df = jhu_data.cleaned()
        df = df.loc[(df[Term.COUNTRY] == "Japan") & (df[Term.PROVINCE] != "-")]
        df = df.groupby(Term.PROVINCE).last().reset_index().dropna()
        jpn_map(
            prefectures=df[Term.PROVINCE], values=df[variable],
            title="Japan: the number of {variable/lower()} cases"
        )

    @pytest.mark.parametrize("variable", ["Infected"])
    def test_japan_map(self, jhu_data, imgfile, variable):
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        df = jhu_data.cleaned()
        df = df.loc[(df[Term.COUNTRY] == "Japan") & (df[Term.PROVINCE] != "-")]
        df = df.groupby(Term.PROVINCE).last().reset_index().dropna()
        jpn_map(
            prefectures=df[Term.PROVINCE], values=df[variable],
            title="Japan: the number of {variable.lower()} cases",
            filename=imgfile
        )
