#!/usr/bin/env python
# -*- coding: utf-8 -*-

from covsirphy.util.error import SubsetNotFoundError
from pathlib import Path
import warnings
import matplotlib
import pytest
from covsirphy import VisualizeBase, ColoredMap
from covsirphy import UnExpectedValueError, Term


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
    def test_global_country(self, imgfile, jhu_data):
        df = jhu_data.cleaned()
        df = df.loc[df[Term.PROVINCE] == Term.UNKNOWN]
        df = df.groupby(Term.COUNTRY).last()
        with ColoredMap(filename=imgfile) as cm:
            cm.plot(series=df[Term.C], index_name=Term.COUNTRY)

    @pytest.mark.parametrize("country", ["Japan", "United States", "China"])
    def test_in_a_country(self, imgfile, jhu_data, country):
        df = jhu_data.cleaned()
        df = df.loc[df[Term.COUNTRY] == country]
        df = df.loc[df[Term.PROVINCE] != Term.UNKNOWN]
        df = df.groupby(Term.PROVINCE).last().dropna()
        with ColoredMap(filename=imgfile) as cm:
            cm.plot(series=df[Term.C], index_name=Term.PROVINCE)

    @pytest.mark.parametrize("country", ["Greece"])
    def test_in_a_country_error(self, imgfile, jhu_data, country):
        df = jhu_data.cleaned()
        df = df.loc[df[Term.COUNTRY] == country]
        df = df.loc[df[Term.PROVINCE] != Term.UNKNOWN]
        # No records found at province level
        assert df.empty
        df = df.groupby(Term.PROVINCE).last().dropna()
        with pytest.raises(SubsetNotFoundError):
            with ColoredMap(filename=imgfile) as cm:
                cm.plot(series=df[Term.C], index_name=Term.PROVINCE)
