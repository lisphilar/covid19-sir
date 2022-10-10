import warnings
import matplotlib
import pytest
from covsirphy import VisualizeBase


def test_base():
    warnings.filterwarnings("ignore", category=UserWarning)
    with VisualizeBase() as vb:
        with pytest.raises(NotImplementedError):
            vb.plot()


def test_file(imgfile):
    with VisualizeBase(filename=imgfile):
        pass


def test_setting(imgfile):
    with VisualizeBase(filename=imgfile) as vb:
        assert not vb.title
        assert isinstance(vb.ax, matplotlib.axes.Axes)
        vb.ax = vb.ax
        vb.title = "title"
        vb.tick_params(
            labelbottom=False, labelleft=False, left=False, bottom=False)
