import pandas as pd
import pytest
from covsirphy import LinePlot, line_plot, Term, UnExecutedError


class TestLinePlot(object):
    def test_plot(self, japan_df, imgfile):
        df = japan_df.set_index("date").rename(columns={"Positive": Term.C})
        with LinePlot(filename=imgfile) as lp:
            lp.plot(data=df)
        with LinePlot(filename=imgfile) as lp:
            lp.plot(data=df[Term.C])
        with LinePlot(filename=imgfile) as lp:
            lp.plot(data=df, colormap="rainbow")
        with LinePlot(filename=imgfile) as lp:
            lp.plot(data=df, color_dict={Term.C: "blue"})
        with pytest.raises(KeyError):
            with LinePlot(filename=imgfile) as lp:
                lp.plot(data=df, colormap="unknown")

    def test_axis(self, japan_df, imgfile):
        df = japan_df.set_index("date").rename(columns={"Positive": Term.C})
        with LinePlot(filename=imgfile) as lp:
            lp.plot(data=df)
            lp.x_axis(x_logscale=True)
            lp.y_axis(y_logscale=True)
            lp.line(v=pd.Timestamp("01Jan2021"))
        with LinePlot(filename=imgfile) as lp:
            lp.plot(data=df)
            lp.y_axis(y_integer=True)
            lp.line(h=100_000)

    def test_legend(self, japan_df, imgfile):
        df = japan_df.set_index("date")
        with LinePlot(filename=imgfile) as lp:
            with pytest.raises(UnExecutedError):
                lp.legend()
            lp.plot(data=df)
            lp.legend_hide()
            lp.legend()

    def test_function(self, japan_df, imgfile):
        df = japan_df.set_index("date")
        line_plot(df=df, filename=imgfile, show_legend=True)
        line_plot(df=df, filename=imgfile, show_legend=False)
