#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pathlib import Path
import pytest
from covsirphy import VisualizeBase, ColoredMap


@pytest.fixture(scope="function")
def imgfile():
    dirpath = Path("input")
    dirpath.mkdir(exist_ok=True)
    filepath = dirpath.joinpath("test.jpg")
    yield str(filepath)
    filepath.unlink(missing_ok=True)


class TestVisualizeBase(object):
    def test_base(self):
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
