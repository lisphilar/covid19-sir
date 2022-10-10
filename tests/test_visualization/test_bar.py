import pytest
from covsirphy import BarPlot, bar_plot, Term


class TestBarPlot(object):
    def test_plot(self, japan_df, imgfile):
        df = japan_df.set_index("date").rename(columns={"Positive": Term.C}).tail()
        with BarPlot(filename=imgfile) as bp:
            bp.plot(data=df[Term.C])
        with BarPlot(filename=imgfile) as bp:
            bp.plot(data=df, vertical=True)
        with BarPlot(filename=imgfile) as bp:
            bp.plot(data=df, vertical=False)
        with BarPlot(filename=imgfile) as bp:
            bp.plot(data=df, colormap="rainbow")
        with BarPlot(filename=imgfile) as bp:
            bp.plot(data=df, color_dict={Term.C: "blue"})
        with pytest.raises(KeyError):
            with BarPlot(filename=imgfile) as bp:
                bp.plot(data=df, colormap="unknown")

    def test_axis(self, japan_df, imgfile):
        df = japan_df.set_index("date").rename(columns={"Positive": Term.C}).tail()
        with BarPlot(filename=imgfile) as bp:
            pass
        with BarPlot(filename=imgfile) as bp:
            bp.plot(data=df)
            bp.y_axis(y_integer=True)
        with BarPlot(filename=imgfile) as bp:
            bp.plot(data=df)
            bp.x_axis(xlabel=Term.DATE)
            bp.y_axis(y_logscale=True)
            bp.line(h=100_000)
        with BarPlot(filename=imgfile) as bp:
            bp.plot(data=df, vertical=True)
            bp.line(v=100_000)

    def test_function(self, japan_df, imgfile):
        df = japan_df.set_index("date").rename(columns={"Positive": Term.C}).tail()
        bar_plot(df=df, filename=imgfile)
        bar_plot(df=df, filename=imgfile, show_legend=False)
