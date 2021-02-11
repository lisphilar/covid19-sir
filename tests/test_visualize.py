#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pathlib import Path
import warnings
import matplotlib
import pytest
from covsirphy import VisualizeBase, ColoredMap
from covsirphy import Term


@pytest.fixture(scope="function")
def imgfile():
    dirpath = Path("input")
    dirpath.mkdir(exist_ok=True)
    filepath = dirpath.joinpath("test.jpg")
    yield str(filepath)
    try:
        filepath.unlink(missing_ok=True)
    except TypeError:
        # Python 3.7
        if filepath.exists():
            filepath.unlink()


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
    @pytest.mark.parametrize("variable", ["Infected"])
    def test_global_country(self, imgfile, jhu_data, variable):
        df = jhu_data.cleaned()
        df = df.loc[df[Term.PROVINCE] == Term.UNKNOWN]
        df = df.groupby(Term.COUNTRY).last().reset_index()
        df.rename(columns={variable: "Value"}, inplace=True)
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
