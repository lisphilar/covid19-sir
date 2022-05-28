#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
from pandas.testing import assert_frame_equal, assert_index_equal, assert_series_equal
import pytest
from covsirphy import ODEModel, SIRModel
from covsirphy import Term, Validator


def test_not_implemented():
    model = ODEModel.from_sample()
    with pytest.raises(NotImplementedError):
        model.solve()
    with pytest.raises(NotImplementedError):
        model.transform(data=pd.DataFrame())
    with pytest.raises(NotImplementedError):
        model.inverse_transform(data=pd.DataFrame)
    with pytest.raises(NotImplementedError):
        model.r0()
    with pytest.raises(NotImplementedError):
        model.dimensional_parameters()
    with pytest.raises(NotImplementedError):
        model._param_quantile(data=pd.DataFrame(), q=0.5)


@pytest.mark.parametrize("model_class", [SIRModel])
class TestODEModel(object):
    def test_special(self, model_class):
        model = model_class.from_sample()
        assert str(model) == model._NAME
        assert model == model_class(**model.to_dict())
        assert model == eval(repr(model))

    @pytest.mark.parametrize("tau", [720, 1440])
    def test_solve(self, model_class, tau):
        model = model_class.from_sample(tau=tau)
        df = model.solve()
        assert df.iloc[0].to_dict() == model.to_dict()["initial_dict"]
        Validator(df, name="analytical solution").dataframe(time_index=True, columns=model_class._VARIABLES)
        assert df.index.name == Term.DATE

    @pytest.mark.parametrize("tau", [720, 1440, None])
    def test_transform(self, model_class, tau):
        model = model_class.from_sample(tau=tau or 360)
        start_date = model.to_dict()["date_range"][0]
        solved_df = model.solve()
        actual_df = model_class.inverse_transform(solved_df, tau=tau, start_date=start_date)
        assert_frame_equal(model_class.transform(actual_df), solved_df)
        trans_df = model_class.transform(actual_df, tau=tau)
        assert_frame_equal(trans_df.reset_index(drop=True), solved_df.reset_index(drop=True))

    def test_r0(self, model_class):
        model = model_class.from_sample()
        assert model.r0() > 0
        with pytest.raises(ZeroDivisionError):
            _dict = model.to_dict()
            _dict.update(param_dict={param: 0 for param in _dict["param_dict"].keys()})
            model_class(**_dict).r0()

    def test_dimensional_parameters(self, model_class):
        model = model_class.from_sample()
        assert model.dimensional_parameters()
        with pytest.raises(ZeroDivisionError):
            _dict = model.to_dict()
            _dict.update(param_dict={param: 0 for param in _dict["param_dict"].keys()})
            model_class(**_dict).dimensional_parameters()

    @pytest.mark.parametrize("tau", [720, 1440])
    def test_from_data(self, model_class, tau):
        sample_model = model_class.from_sample(tau=tau)
        sample_df = sample_model.solve()
        sample_dict = sample_model.to_dict()
        trans_df = model_class.inverse_transform(sample_df).reset_index()
        model = model_class.from_data(data=trans_df, param_dict=sample_dict["param_dict"], tau=tau, digits=None)
        assert model.to_dict(with_estimation=True)["estimation_dict"]["method"] == "not_performed"
        solved_df = model.solve()
        assert_index_equal(solved_df.index, sample_df.index)
        assert_series_equal(solved_df.iloc[0], sample_df.iloc[0])

    @pytest.mark.parametrize("tau", [720, 1440])
    @pytest.mark.parametrize("q", [0.5])
    def test_from_data_with_quantile(self, model_class, tau, q):
        sample_model = model_class.from_sample(tau=tau)
        sample_df = sample_model.solve()
        trans_df = model_class.inverse_transform(sample_df).reset_index()
        model = model_class.from_data_with_quantile(data=trans_df, tau=tau, q=q, digits=2)
        assert model.to_dict(with_estimation=True)["estimation_dict"]["method"] == "with_quantile"
        solved_df = model.solve()
        assert_index_equal(solved_df.index, sample_df.index)
        assert_series_equal(solved_df.iloc[0], sample_df.iloc[0])

    @pytest.mark.parametrize("tau", [720, 1440])
    @pytest.mark.parametrize("metric", ["RMSLE"])
    def test_from_data_with_optimization(self, model_class, tau, metric):
        sample_model = model_class.from_sample(tau=tau)
        sample_df = sample_model.solve()
        trans_df = model_class.inverse_transform(sample_df).reset_index()
        model = model_class.from_data_with_optimization(data=trans_df, tau=tau, metric=metric, digits=4)
        assert model.to_dict(with_estimation=True)["estimation_dict"]["method"] == "with_optimization"
        solved_df = model.solve()
        assert_index_equal(solved_df.index, sample_df.index)
        assert_series_equal(solved_df.iloc[0], sample_df.iloc[0])
