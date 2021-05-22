#!/usr/bin/env python
# -*- coding: utf-8 -*-

from covsirphy.util.error import UnExpectedReturnValueError
from covsirphy.util.evaluator import Evaluator
from covsirphy.util.term import Term
from covsirphy.ode.mbase import ModelBase
from covsirphy.regression.param_elastic_net import _ParamElasticNetRegressor
from covsirphy.regression.param_decision_tree import _ParamDecisionTreeRegressor
from covsirphy.regression.rate_elastic_net import _RateElasticNetRegressor
from covsirphy.regression.rate_decision_tree import _RateDecisionTreeRegressor


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
        delay (int): delay period [days]
        kwargs: keyword arguments of sklearn.model_selection.train_test_split(test_size=0.2, random_state=0)

    Note:
        If @seed is included in kwargs, this will be converted to @random_state.
    """

    def __init__(self, data, model, delay, **kwargs):
        # Dataset
        self._data = self._ensure_dataframe(data, name="data", time_index=True)
        # ODE parameter values
        self._ensure_subclass(model, ModelBase, name="model")
        self._parameters = model.PARAMETERS[:]
        # Delay period
        self._delay = self._ensure_natural_int(delay, name="delay")
        # Keyword arguments
        self._kwargs = kwargs.copy()
        # All regressors {name: RegressorBase}
        self._reg_dict = {}
        # The best regressor name
        self._best = None

    def fit(self, metric):
        """
        Fit regressors and select the best regressor based on the scores with test dataset.

        Args:
            metric (str): metric name to select the best regressor

        Raises:
            ValueError: un-expected parameter values were predcited by all regressors, out of range (0, 1)

        Returns:
            float: the best score

        Note:
            All regressors are here.
            - Indicators -> Parameters with Elastic Net
            - Indicators -> Parameters with Decision Tree Regressor
            - Indicators(n)/Indicators(n-1) -> Parameters(n)/Parameters(n-1) with Elastic Net
            - Indicators(n)/Indicators(n-1) -> Parameters(n)/Parameters(n-1) with Decision Tree Regressor
        """
        # All approaches
        approach_dict = {
            _ParamElasticNetRegressor.DESC: self._fit_param_reg(_ParamElasticNetRegressor),
            _ParamDecisionTreeRegressor.DESC: self._fit_param_reg(_ParamDecisionTreeRegressor),
            _RateElasticNetRegressor.DESC: self._fit_param_reg(_RateElasticNetRegressor),
            _RateDecisionTreeRegressor.DESC: self._fit_param_reg(_RateDecisionTreeRegressor),
        }
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

    def _fit_param_reg(self, regressor_class):
        """
        Fit a regressor which uses ODE parameter values as y.

        Args:
            regressor_class (covsirphy.regression.regbase.RegressorBase): regression class

        Returns:
            covsirphy.regression.regbase.RegressorBase: fitted regressor
        """
        df = self._data.drop([self.C, self.CI, self.F, self.R, self.S], axis=1, errors="ignore")
        self._ensure_dataframe(df, name="data", columns=self._parameters)
        X = df.drop(self._parameters, axis=1)
        y = df.loc[:, self._parameters]
        return regressor_class(X=X, y=y, delay=self._delay, **self._kwargs)

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
                - delay (int): delay period
        """
        fit_dict = {"best": self._best}
        fit_dict.update(self._reg_dict[self._best].to_dict(metric=metric))
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
