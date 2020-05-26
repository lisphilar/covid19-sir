#!/usr/bin/env python
# -*- coding: utf-8 -*-

from covsirphy.analysis.phase_series import PhaseSeries
from covsirphy.analysis.sr_change import ChangeFinder
from covsirphy.cleaning.word import Word
from covsirphy.cleaning.jhu_data import JHUData
from covsirphy.cleaning.population import Population
from covsirphy.phase.sr_data import SRData
from covsirphy.util.plotting import line_plot


class Scenario(Word):
    """
    Scenario analysis.
    """

    def __init__(self, jhu_data, pop_data, country, province=None):
        """
        @jhu_data <covsirphy.JHUData>: object of records
        @pop_data <covsirphy.Population>: Population object
        @country <str>: country name
        @province <str>: province name
        """
        # Records
        if not isinstance(jhu_data, JHUData):
            raise TypeError(
                "@jhu_data must be a instance of <covsirphy.JHUData>."
            )
        self.jhu_data = jhu_data
        self.clean_df = jhu_data.cleaned()
        # Population
        if not isinstance(pop_data, Population):
            raise TypeError(
                "@pop_data must be a instance of <covsirphy.Population>."
            )
        self.population = pop_data.value(country, province=province)
        # Area name
        self.country = country
        self.province = province
        if province is None:
            self.name = country
        else:
            self.name = f"{country}{self.SEP}{province}"
        # First date of the area
        sr_data = SRData(self.clean_df, country=country, province=province)
        df = sr_data.make(self.population)
        self.first_date = df.index.min().strftime(self.DATE_FORMAT)
        # Init
        self.phase_series = PhaseSeries(self.first_date, self.population)

    def records(self, show_figure=True, filename=None):
        """
        Return the records as a dataframe.
        @show_figure <bool>:
            - if True, show the records as a line-plot.
        @filename <str>: filename of the figure, or None (show figure)
        """
        df = self.jhu_data.subset(self.country, province=self.province)
        if not show_figure:
            return df
        line_plot(
            df.set_index(self.DATE).drop(self.C, axis=1),
            f"{self.name}: Cases over time",
            y_integer=True,
            filename=filename
        )
        return df

    def add_phase(self, start_date, end_date, population=None):
        """
        Add a new phase.
        @start_date <str>: start date of the new phase
        @end_date <str>: end date of the new phase
        @population <int>: population value of the start date
            - if None, the same as initial value
        @return self
        """
        if population is None:
            population = self.population
        self.phase_series.add(start_date, end_date, population)
        return self

    def summary(self):
        """
        Summarize the series of phases in a dataframe.
        @return <pd.DataFrame>:
            - as the same as PhaseSeries().summary()
        """
        return self.phase_series.summary()

    def trend(self, n_points=0,
              set_phases=True, include_init_phase=False,
              show_figure=True, filename=None, **kwargs):
        """
        Perform S-R trend analysis and set phases.
        @n_points <int>: the number of change points
        @set_phases <bool>:
            - if True and n_points is not 0, set phases automatically
            - if @include_init_phase is False, initial phase will not be used
        @include_init_phase <bool>: whether use initial phase or not
        @show_figure <bool>:
            - if True, show the records as a line-plot.
        @filename <str>: filename of the figure, or None (show figure)
        @kwargs: the other keyword arguments of ChangeFinder().run()
        @return self
        """
        finder = ChangeFinder(
            self.clean_df, self.population,
            country=self.country, province=self.province
        )
        finder.run(n_points=n_points, **kwargs)
        phase_series = finder.show(show_figure=show_figure, filename=filename)
        if n_points != 0 and set_phases:
            self.phase_series = phase_series
            if not include_init_phase:
                self.phase_series.delete("0th")
        return self
