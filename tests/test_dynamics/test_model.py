#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
from pandas.testing import assert_frame_equal
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


@pytest.mark.parametrize("model_class", [SIRModel])
class TestODEModel(object):
    def test_special(self, model_class):
        model = model_class.from_sample()
        assert str(model) == model._NAME
        assert model == model_class(**model.to_dict())
        assert model == eval(repr(model))

    def test_solve(self, model_class):
        model = model_class.from_sample()
        df = model.solve()
        Validator(df, name="analytical solution").dataframe(time_index=True, columns=model_class._VARIABLES)
        assert df.index.name == Term.DATE

    def test_transform(self, model_class):
        model = model_class.from_sample()
        solved_df = model.solve()
        actual_df = model_class.inverse_transform(solved_df)
        assert_frame_equal(model_class.transform(actual_df), solved_df)

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
