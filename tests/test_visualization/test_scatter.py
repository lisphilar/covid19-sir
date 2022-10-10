import pytest
from covsirphy import ScatterPlot, scatter_plot
from covsirphy import UnExecutedError


class TestScatterPlot(object):
    def test_plot(self, japan_df, imgfile):
        df = japan_df.rename(columns={"Positive": "x", "Discharged": "y"})
        # Create a scatter plot
        scatter_plot(df, filename=imgfile)

    def test_error(self, japan_df, imgfile):
        df = japan_df.rename(columns={"Positive": "x", "Discharged": "y"})
        # Plotting not done
        with ScatterPlot(filename=imgfile) as sp:
            with pytest.raises(UnExecutedError):
                sp.line_straight()
        # Cannot show a legend
        with ScatterPlot(filename=imgfile) as sp:
            sp.plot(data=df)
            with pytest.raises(NotImplementedError):
                sp.legend()
            with pytest.raises(NotImplementedError):
                sp.legend_hide()

    @pytest.mark.skip(reason="Refer to https://github.com/lisphilar/covid19-sir/issues/1225")
    def test_colormap(self, japan_df, imgfile):
        df = japan_df.rename(columns={"Positive": "x", "Discharged": "y"})
        with ScatterPlot(filename=imgfile) as sp:
            with pytest.raises(KeyError):
                sp.plot(data=df, colormap="unknown")
