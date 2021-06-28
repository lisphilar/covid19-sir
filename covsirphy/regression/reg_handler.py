#!/usr/bin/env python
# -*- coding: utf-8 -*-

from covsirphy.util.error import UnExpectedReturnValueError, NotIncludedError
from covsirphy.util.evaluator import Evaluator
from covsirphy.util.term import Term
from covsirphy.ode.mbase import ModelBase
from covsirphy.regression.feature_engineer import _FeatureEngineer
from covsirphy.regression.param_elastic_net import _ParamElasticNetRegressor
from covsirphy.regression.param_decision_tree import _ParamDecisionTreeRegressor
from covsirphy.regression.param_lightgbm import _ParamLightGBMRegressor
from covsirphy.regression.param_svr import _ParamSVRegressor


class RegressionHandler(Term):
    """
    Handle regressors to predict parameter values of ODE models.
    With .fit() method, the best regressor will be selected based on the scores with test dataset.

    Args:
        data (pandas.DataFrame):
            Index
                Date (pandas.Timestamp): observation date
            Columns
                - parameter values
                - the number of cases
                - indicators
        model (covsirphy.ModelBase): ODE model
        kwargs: keyword arguments of sklearn.model_selection.train_test_split()

    Note:
        If @seed is included in kwargs, this will be converted to @random_state.

    Note:
        default values regarding sklearn.model_selection.train_test_split() are
        test_size=0.2, random_state=0, shuffle=False.
    """

    def __init__(self, data, model, **kwargs):
        # ODE parameter values
        self._ensure_subclass(model, ModelBase, name="model")
        self._parameters = model.PARAMETERS[:]
        # Set datasets (create _FeatureEngineer instance)
        self._ensure_dataframe(data, name="data", columns=self._parameters, time_index=True)
        df = data.drop([self.C, self.CI, self.F, self.R, self.S], axis=1, errors="ignore")
        X = df.drop(self._parameters, axis=1)
        Y = df.loc[:, self._parameters]
        self._engineer = _FeatureEngineer(X, Y)
        # Keyword arguments
        self._kwargs = kwargs.copy()
        # All regressors {name: RegressorBase}
        self._reg_dict = {}
        # The best regressor name and determined delay period
        self._best = None
        # Delay period
        self._delay_candidates = []
        # Backward compatibility, version < 2.21.0-alpha
        if "delay" in kwargs:
            self.feature_engineering(tools=["delay"], delay=kwargs["delay"])

    def _convert_delay_value(self, delay):
        """
        Convert delay value to candidate list of delay periods.

        Args:
            delay (int or tuple(int, int) or None): exact (or value range of) delay period [days]

        Raises:
            ValueError: @delay is None

        Returns:
            list[int]: candidates of delay periods
        """
        # Delay period
        if delay is None:
            raise ValueError(
                "@delay must be integer or tuple(int, int) when @tools is None or includes 'delay'.")
        if isinstance(delay, tuple):
            delay_min, delay_max = delay
            self._ensure_natural_int(delay_min, name="delay[0]")
            self._ensure_natural_int(delay_max, name="delay[1]")
            return list(range(delay_min, delay_max + 1))
        return [self._ensure_natural_int(delay, name="delay")]

    def feature_engineering(self, engineering_tools=None, delay=None):
        """
        Perform feature engineering of X dataset.

        Args:
            engineering_tools (list[str]): list of the feature engineering tools or None (all tools)
            delay (int or tuple(int, int) or None): exact (or value range of) delay period [days]

        Raises:
            ValueError: @delay is None when @tools is None or 'delay' is included in @engineering_tools
            NotIncludedError: @delay was not included in @engineering_tools

        Note:
            All tools and names are
            - "elapsed": alculate elapsed days from the last change point of indicators
            - "log": add log-transforemd indeciator values
            - "delay": add delayed (lagged) variables with @delay (must not be None)

        Note:
            "delay" must be included in the tools because delay is required to create target X.

        Note:
            "delay" will be applied to all indicators, including features created by the other tools.
        """
        # Delay period
        if engineering_tools is None or "delay" in engineering_tools:
            self._delay_candidates = self._convert_delay_value(delay)
        else:
            raise NotIncludedError(name="engineering_tools", value="delay")
        # Tools of feature engineering
        tool_dict = {
            "elapsed": (self._engineer.add_elapsed, {}),
            "log": (self._engineer.log_transform, {}),
            "delay": (self._engineer.apply_delay, {"delay_values": self._delay_candidates}),
        }
        all_tools = list(tool_dict.keys())
        selected_tools = self._ensure_list(
            engineering_tools or all_tools, candidates=all_tools, name="tools")
        # Perform feature engineering
        for name in selected_tools:
            method, arg_dict = tool_dict[name]
            method(**arg_dict)

    def fit(self, metric, regressors=None):
        """
        Fit regressors and select the best regressor based on the scores with test dataset.

        Args:
            metric (str): metric name to select the best regressor
            regressor (list[str]): list of regressors selected from en, dt, lgbm, svr (refer to note)

        Raises:
            ValueError: un-expected parameter values were predcited by all regressors, out of range (0, 1)

        Returns:
            float: the best score

        Note:
            All regressors are here.
            - "en": Indicators -> Parameters with Elastic Net
            - "dt": Indicators -> Parameters with Decision Tree Regressor
            - "lgbm": Indicators -> Parameters with Light Gradient Boosting Machine Regressor
            - "svr": Indicators -> Parameters with Epsilon-Support Vector Regressor
        """
        data_dict = self._engineer.split(**self._kwargs)
        # Select regressors
        all_reg_dict = {
            "en": _ParamElasticNetRegressor,
            "dt": _ParamDecisionTreeRegressor,
            "lgbm": _ParamLightGBMRegressor,
            "svr": _ParamSVRegressor,
        }
        sel_regressors = [
            reg for (name, reg) in all_reg_dict.items() if regressors is None or name in regressors]
        approach_dict = {reg.DESC: reg(**data_dict) for reg in sel_regressors}
        # Predicted all parameter values must be >= 0
        self._reg_dict = {
            k: v for (k, v) in approach_dict.items()
            if v.predict().ge(0).all().all() and v.predict().le(1).all().all()}
        if not self._reg_dict:
            raise UnExpectedReturnValueError(
                name="ODE parameter values", value=None, plural=True,
                message="Values are out of range (0, 1) with all regressors")
        # Select the best regressor with the metric
        score_dict = {k: v.score_test(metric=metric) for (k, v) in self._reg_dict.items()}
        self._best, score = Evaluator.best_one(score_dict, metric=metric)
        return score

    def to_dict(self, metric):
        """
        Return information regarding the best regressor.

        Args:
            metric (str): metric name to select the best regressor

        Returns:
            dict(str, object): regressor information of the best model, including
                - best (str): description of the selected approach
                - scaler (object): scaler class
                - regressor (object): regressor class
                - alpha (float): alpha value used in Elastic Net regression
                - l1_ratio (float): l1_ratio value used in Elastic Net regression
                - score_name (str): scoring method (specified with @metric or @metrics)
                - score_train (float): score with train dataset
                - score_test (float): score with test dataset
                - dataset (dict[numpy.ndarray]): X_train, X_test, y_train, y_test, X_target
                - intercept (pandas.DataFrame): intercept and coefficients (Index ODE parameters, Columns indicators)
                - coef (pandas.DataFrame): intercept and coefficients (Index ODE parameters, Columns indicators)
                - delay (list[int]): list of delay period [days]
        """
        fit_dict = {"best": self._best}
        fit_dict.update(self._reg_dict[self._best].to_dict(metric=metric))
        fit_dict["delay"] = self._delay_candidates
        return fit_dict

    def predict(self):
        """
        Predict parameter values of the ODE model using the best regressor.

        Returns:
            pandas.DataFrame:
                Index
                    Date (pandas.Timestamp): future dates
                Columns
                    (float): parameter values (4 digits)
        """
        return self._reg_dict[self._best].predict()

    def pred_actual_plot(self, metric, filename=None):
        """
        Create a scatter plot (predicted vs. actual parameter values).

        Args:
            metric (str): metric name, refer to covsirphy.Evaluator.score()
            fileaname (str): filename of the figure or None (display)
        """
        return self._reg_dict[self._best].pred_actual_plot(metric=metric, filename=filename)
