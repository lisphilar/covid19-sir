#!/usr/bin/env python
# -*- coding: utf-8 -*-

from inspect import signature
import matplotlib.pyplot as plt
from covsirphy.ode import ModelBase
from covsirphy.cleaning import JHUData, Population, Word
from covsirphy.phase import Estimator, SRData, NondimData
from covsirphy.util import line_plot
from covsirphy.analysis.phase_series import PhaseSeries
from covsirphy.analysis.simulator import ODESimulator
from covsirphy.analysis.sr_change import ChangeFinder


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
        # {model_name: model_class}
        self.model_dict = dict()

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

    def add_phase(self, end_date, population=None, model=None, **kwargs):
        """
        Add a new phase.
        The start date is the next date of the last registered phase.
        @end_date <str>: end date of the new phase
        @population <int>: population value of the start date
            - if None, the same as initial value
        @model <covsirphy.ModelBase>: ODE model
        @kwargs: keyword arguments of ODE model parameters
            - un-included parameters will be the same as the last phase
                - if model is not the same, None
                - tau is fixed as the last phase's value or 1440
        @return self
        """
        if population is None:
            population = self.population
        start_date = self.phase_series.next_date()
        if model is not None and not issubclass(model, ModelBase):
            raise TypeError(
                "@model must be an ODE model <sub-class of cs.ModelBase>."
            )
        param_dict = {self.TAU: self.tau}
        if model is not None:
            param_dict[self.ODE] = model.NAME
        summary_df = self.phase_series.summary()
        if model is None and self.ODE in summary_df.columns:
            model_name = summary_df.loc[summary_df.index[-1], self.ODE]
            if model_name is not None:
                param_dict[self.ODE] = model_name
                model = self.model_dict[model_name]
                model_param_dict = dict()
                for param in model.PARAMETERS:
                    model_param_dict[param] = summary_df.loc[summary_df.index[-1], param]
                model_param_dict.update(kwargs)
                param_dict.update(model_param_dict)
                model_instance = model(**model_param_dict)
                param_dict[self.RT] = model_instance.calc_r0()
                param_dict.update(model_instance.calc_days_dict(self.tau))
        self.phase_series.add(start_date, end_date, population, **param_dict)
        return self

    def clear(self, include_past=False):
        """
        Clear phase information.
        @include_past <bool>:
            - if True, include past phases.
            - future phase are always included
        return self
        """
        self.phase_series.clear(include_past=include_past)
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
        phase_est_dict = {self.ODE: model.NAME}
        phase_est_dict.update(est_df.to_dict(orient="index")[phase])
        self.tau = phase_est_dict[self.TAU]
        self.phase_series.update(phase, **phase_est_dict)
        self.estimator_dict[phase] = estimator
        self.model_dict[model.NAME] = model

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

    def predict(self, y0_dict=None, show_figure=True, filename=None):
        """
        Predict the number of cases.
        @y0_dict <doct[str]=float>:
            - dictionary of initial values or None
            - if model will be changed in the later phase, must be specified
        @show_figure <bool>:
            - if True, show the result as a figure.
        @filename <str>: filename of the figure, or None (show figure)
        @return <pd.DataFrame>
            - index <int>: reseted index
            - Date <pd.TimeStamp>: Observation date
            - Country <str>: country/region name
            - Province <str>: province/prefecture/state name
            - variables of the models <int>: Confirmed <int> etc.
        """
        # TODO: Refactoring, split this method
        df = self.phase_series.summary()
        # Future must be added in advance
        if self.FUTURE not in df[self.TENSE].unique():
            raise KeyError(
                "Future phases must be registered by Scenario.add_phase() in advance."
            )
        simulator = ODESimulator(
            self.country,
            province="-" if self.province is None else self.province
        )
        for phase in df.index:
            model_name = df.loc[phase, self.ODE]
            model = self.model_dict[model_name]
            start_obj = self.date_obj(df.loc[phase, self.START])
            end_obj = self.date_obj(df.loc[phase, self.END])
            step_n = int(((end_obj - start_obj).total_seconds() + 1) / 60 / self.tau)
            population = df.loc[phase, self.N]
            param_dict = df[model.PARAMETERS].to_dict(orient="index")[phase]
            if phase == "1st":
                # Calculate intial values
                nondim_data = NondimData(
                    self.clean_df, country=self.country,
                    province=self.province
                )
                nondim_df = nondim_data.make(model, population)
                init_index = [
                    date_obj for (date_obj, _)
                    in self.phase_series.phase_dict.items()
                    if date_obj == start_obj
                ][0]
                y0_dict_phase = {
                    v: nondim_df.loc[init_index, v] for v in model.VARIABLES
                }
            else:
                try:
                    y0_dict_phase = y0_dict.copy()
                except AttributeError:
                    y0_dict_phase = None
            simulator.add(
                model, step_n, population,
                param_dict=param_dict,
                y0_dict=y0_dict_phase
            )
        simulator.run()
        dim_df = simulator.dim(self.tau, df.loc["1st", self.START])
        # Show figure
        fig_cols_set = set(dim_df.set_index(self.DATE).columns) & set(self.FIG_COLUMNS)
        fig_cols = [col for col in self.FIG_COLUMNS if col in fig_cols_set]
        if dim_df[fig_cols].values.max() > self.population / 2:
            fig_cols.append(self.S)
        # TODO: add vertical lines to line-plot with tau and step_n
        line_plot(
            dim_df.set_index(self.DATE)[fig_cols],
            title=f"{self.name}: Predicted number of cases",
            filename=filename,
            y_integer=True
        )
        return dim_df

    def get(self, param, phase="last"):
        """
        Get the parameter value of the phase.
        @param <str>: parameter name (columns in self.summary())
        @phase <str>: phase name or 'last'
            - if 'last', the value of the last phase will be returned
        @return <str/int/float>
        """
        df = self.summary()
        if param not in df.columns:
            raise KeyError(f"@param must be in {', '.join(df.columns)}.")
        if phase == "last":
            phase = df.index[-1]
        return df.loc[phase, param]

    def param_history(self, targets=None, divide_by_first=True,
                      show_figure=True, filename=None, box_plot=True, **kwargs):
        """
        Return subset of summary.
        @targets <list[str]/str>: parameters to show (Rt etc.)
        @divide_by_first <bool>: if True, divide the values by 1st phase's values
        @box_plot <bool>: if True, box plot. if False, line plot.
        @show_figure <bool>:
            - if True, show the result as a figure.
        @filename <str>: filename of the figure, or None (show figure)
        @kwargs: keword arguments of pd.DataFrame.plot or line_plot()
        @return <pd.DataFrame>
        """
        if filename is not None:
            plt.switch_backend("Agg")
        df = self.summary()
        model_param_nest = [m.PARAMETERS for m in self.model_dict.values()]
        model_parameters = list(set(sum(model_param_nest, list())))
        model_day_nest = [m.DAY_PARAMETERS for m in self.model_dict.values()]
        model_day_params = list(set(sum(model_day_nest, list())))
        selectable_cols = [
            self.N, *model_parameters, self.RT, *model_day_params
        ]
        targets = [targets] if isinstance(targets, str) else targets
        targets = selectable_cols if targets is None else targets
        if not set(targets).issubset(set(selectable_cols)):
            raise KeyError(
                f"@targets must be a subset of {', '.join(selectable_cols)}."
            )
        df = df.loc[:, targets]
        if divide_by_first:
            df = df / df.iloc[0, :]
            title = f"{self.name}: Ratio to 1st phase parameters"
        else:
            title = f"{self.name}: History of parameter values"
        if box_plot:
            df.plot.bar(title=title)
            plt.xticks(rotation=0)
            if divide_by_first or self.RT in targets:
                plt.axhline(y=1.0, color="black", linestyle=":")
            plt.legend(
                bbox_to_anchor=(1.02, 0), loc="lower left", borderaxespad=0
            )
            plt.tight_layout()
            if filename is None:
                plt.show()
                return df
            plt.savefig(
                filename, bbox_inches="tight", transparent=False, dpi=300
            )
            plt.clf()
            return df
        _df = df.reset_index(drop=True)
        _df.index = _df.index + 1
        h = 1.0 if divide_by_first else None
        line_plot(
            _df, title=title,
            xlabel="Phase", ylabel=str(), math_scale=False, h=h,
            show_figure=show_figure, filename=filename
        )

