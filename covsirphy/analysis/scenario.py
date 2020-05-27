#!/usr/bin/env python
# -*- coding: utf-8 -*-

from inspect import signature
from covsirphy.ode import ModelBase
from covsirphy.cleaning import JHUData, Population, Word
from covsirphy.phase import Estimator, SRData
from covsirphy.util import line_plot
from covsirphy.analysis.sr_change import ChangeFinder
from covsirphy.analysis.phase_series import PhaseSeries


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
        # First/last date of the area
        sr_data = SRData(self.clean_df, country=country, province=province)
        df = sr_data.make(self.population)
        self.first_date = df.index.min().strftime(self.DATE_FORMAT)
        self.last_date = df.index.max().strftime(self.DATE_FORMAT)
        # Init
        self.phase_series = PhaseSeries(
            self.first_date, self.last_date, self.population
        )
        self.tau = None
        # {phase: Estimator}
        self.estimator_dict = dict()

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

    def _estimate(self, model, phase, **kwargs):
        """
        Estimate the parameters of the model using the records.
        @phase <str>: phase name, like 1st, 2nd...
        @model <covsirphy.ModelBase>: ODE model
        @kwargs:
            - keyword arguments of the model parameter
            - keyword arguments of covsirphy.Estimator.run()
        @retun self
        """
        # Set parameters
        try:
            setting_dict = self.phase_series.to_dict()[phase]
        except KeyError:
            raise KeyError(f"{phase} phase has not been registered.")
        start_date = setting_dict[self.START]
        end_date = setting_dict[self.END]
        population = setting_dict[self.N]
        # Run estinmation
        print(f"{phase} phase with {model.NAME} model:")
        est_kwargs = {
            p: kwargs[p] for p in model.PARAMETERS if p in kwargs.keys()
        }
        if self.tau is not None:
            est_kwargs[self.TAU] = self.tau
        estimator = Estimator(
            self.clean_df, model, population,
            country=self.country, province=self.province,
            start_date=start_date, end_date=end_date,
            **est_kwargs
        )
        sign = signature(Estimator.run)
        run_params = list(sign.parameters.keys())
        run_kwargs = {k: v for (k, v) in kwargs.items() if k in run_params}
        estimator.run(**run_kwargs)
        est_df = estimator.summary(phase)
        est_dict = est_df.to_dict(orient="index")
        self.tau = est_dict[phase][self.TAU]
        self.phase_series.update(phase, **est_dict[phase])
        self.estimator_dict[phase] = estimator

    def estimate(self, model, phases=None, **kwargs):
        """
        Estimate the parameters of the model using the records.
        @phases <list[str]>: list of phase names, like 1st, 2nd...
            - if None, all past phase will be used
        @model <covsirphy.ModelBase>: ODE model
        @kwargs:
            - keyword arguments of the model parameter
            - keyword arguments of covsirphy.Estimator.run()
        @return self
        """
        # Check model
        if not issubclass(model, ModelBase):
            raise TypeError(
                "@model must be an ODE model <sub-class of cs.ModelBase>."
            )
        # Phase names
        phase_dict = self.phase_series.to_dict()
        past_phases = [k for (k, v) in phase_dict.items()]
        phases = past_phases[:] if phases is None else phases
        if not set(phases).issubset(set(past_phases)):
            for phase in phases:
                if phase not in past_phases:
                    raise KeyError(f"{phase} is not a past phase.")
        # Run hyperparameter estimation
        for phase in phases:
            self._estimate(model, phase, **kwargs)
        return self

    def estimate_history(self, phase, **kwargs):
        """
        Show the history of optimization.
        @phase <str>: phase name, like 1st, 2nd...
        @kwargs: keyword arguments of <covsirphy.Estimator.history()>
        """
        try:
            estimator = self.estimator_dict[phase]
        except KeyError:
            raise KeyError(f"Estimator of {phase} phase has not been registered.")
        estimator.history(**kwargs)

    def estimate_accuracy(self, phase, **kwargs):
        """
        Show the accuracy as a figure.
        @phase <str>: phase name, like 1st, 2nd...
        @kwargs: keyword arguments of <covsirphy.Estimator.accuracy()>
        """
        try:
            estimator = self.estimator_dict[phase]
        except KeyError:
            raise KeyError(f"Estimator of {phase} phase has not been registered.")
        estimator.accuracy(**kwargs)
