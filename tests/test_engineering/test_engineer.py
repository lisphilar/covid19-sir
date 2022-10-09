#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
from pandas.testing import assert_frame_equal
import pytest
from covsirphy import DataEngineer, Term, Dynamics, SIRFModel
from covsirphy.util.error import NotIncludedError


class TestDataEngineer(object):
    def test_from_sample_data(self):
        dynamics = Dynamics.from_sample(model=SIRFModel)
        data = dynamics.simulate().reset_index()
        data.insert(0, "Model", SIRFModel.name())
        engineer = DataEngineer(layers=["Model"], country=None)
        engineer.register(data, citations="Simulated data")
        assert set(engineer.all().columns) == {"Model", Term.DATE, Term.S, Term.CI, Term.F, Term.R}
        assert engineer.citations() == ["Simulated data"]
        engineer.inverse_transform()
        assert set(engineer.all().columns) == {"Model", Term.DATE, Term.S, Term.CI, Term.F, Term.R, Term.N, Term.C}
        assert len(engineer.subset(geo=SIRFModel.name())[0]) == len(data)

    def test_operations(self):
        dynamics = Dynamics.from_sample(model=SIRFModel)
        data = dynamics.simulate().reset_index()
        data.insert(0, "Model", SIRFModel.name())
        engineer = DataEngineer(layers=["Model"], country=None)
        engineer.register(data, citations="Simulated data")
        engineer.inverse_transform()
        # Diff
        engineer.diff(column=Term.C, suffix="_diff", freq="D")
        assert f"{Term.C}_diff" in engineer.all()
        # Addition
        engineer.add(columns=[Term.F, Term.R])
        assert f"{Term.F}+{Term.R}" in engineer.all()
        # Multiplication
        engineer.mul(columns=[Term.C, Term.R])
        assert f"{Term.C}*{Term.R}" in engineer.all()
        # Subtraction
        engineer.sub(minuend=Term.C, subtrahend=Term.R)
        assert f"{Term.C}-{Term.R}" in engineer.all()
        # Division and assign
        engineer.assign(Tests=lambda x: x[Term.C] * 10)
        engineer.div(numerator=Term.C, denominator="Tests", new="Positive_rate")
        engineer.assign(**{"Positive_rate_%": lambda x: x["Positive_rate"] * 100})
        assert engineer.all()["Positive_rate_%"].unique() == 10

    def test_with_actual_data(self, imgfile):
        engineer = DataEngineer()
        engineer.download(databases=["japan", "covid19dh", "owid"])
        all_df = engineer.all()
        layer_df = engineer.layer()
        assert_frame_equal(all_df, layer_df, check_dtype=False, check_categorical=False)
        engineer.choropleth(geo=None, variable=Term.C, filename=imgfile)
        engineer.clean()
        engineer.transform()
        assert len(engineer.subset(geo="Japan", complement=False)) == len(engineer.subset(geo="Japan"))
        df, *_ = engineer.subset_alias(alias="UK", geo="UK")
        assert_frame_equal(engineer.subset_alias(alias="UK")[0], df)
        assert engineer.subset_alias(alias=None)
        assert isinstance(DataEngineer.recovery_period(data=df), int)

    def test_variables_alias(self):
        engineer = DataEngineer()
        assert engineer.variables_alias(alias=None)
        engineer.variables_alias(alias="nc", variables=[Term.N, Term.C])
        assert engineer.variables_alias(alias="nc") == [Term.N, Term.C]
        with pytest.raises(NotIncludedError):
            engineer.variables_alias(alias="unknown")

    def test_resample_with_date_range(self):
        eng = DataEngineer(layers=["Country"], country=["Country"])
        eng.download(databases=["wpp"])
        eng.clean(kinds=["resample"], date_range=("01Jan2022", "19Sep2022"))
        df = eng.all()
        assert df["Date"].min() >= pd.to_datetime("01Jan2022")
        assert df["Date"].max() <= pd.to_datetime("19Sep2022")
