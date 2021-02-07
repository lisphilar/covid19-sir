#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pathlib import Path
import warnings
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
            vb.title = "title"
            vb.tick_params(
                labelbottom=False, labelleft=False, left=False, bottom=False)


class TestColoredMap(object):
    def test_error_index_name(self, imgfile, jhu_data):
        df = jhu_data.cleaned().set_index(Term.COUNTRY)
        with pytest.raises(UnExpectedValueError):
            with ColoredMap(filename=imgfile) as cm:
                cm.plot(series=df[Term.C], index_name="feeling")

    def test_not_unique(self, imgfile, jhu_data):
        df = jhu_data.cleaned().set_index(Term.COUNTRY)
        with pytest.raises(ValueError):
            with ColoredMap(filename=imgfile) as cm:
                cm.plot(series=df[Term.C], index_name=Term.COUNTRY)

    def test_global_country(self, imgfile, jhu_data):
        df = jhu_data.cleaned().set_index(Term.COUNTRY)
        df = df.loc[df[Term.DATE] == df[Term.DATE].max()]
        df = df.loc[df[Term.PROVINCE] == Term.UNKNOWN]
        with ColoredMap(filename=imgfile) as cm:
            cm.plot(series=df[Term.C], index_name=Term.COUNTRY)

    def test_global_iso3(self, imgfile, jhu_data):
        df = jhu_data._cleaned_df.set_index(Term.ISO3)
        df = df.loc[df[Term.DATE] == df[Term.DATE].max()]
        df = df.loc[df[Term.PROVINCE] == Term.UNKNOWN]
        with ColoredMap(filename=imgfile) as cm:
            cm.plot(series=df[Term.C], index_name=Term.ISO3)
