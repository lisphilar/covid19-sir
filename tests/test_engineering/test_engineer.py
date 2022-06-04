#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pandas.testing import assert_frame_equal
from covsirphy import DataEngineer, Term, Dynamics, SIRFModel


class TestDataEngineer(object):
    def test_from_sample_data(self):
        dynamics = Dynamics.from_sample(model=SIRFModel)
        data = dynamics.simulate().reset_index()
        data.insert(0, "Model", SIRFModel.name())
        engineer = DataEngineer(layers=["Model"], country=None, verbose=1)
        engineer.register(data, citations="Simulated data")
        assert set(engineer.all().columns) == {"Model", Term.DATE, Term.S, Term.CI, Term.F, Term.R}
        assert engineer.citations() == ["Simulated data"]
        engineer.inverse_transform()
        assert set(engineer.all().columns) == {"Model", Term.DATE, Term.S, Term.CI, Term.F, Term.R, Term.N, Term.C}

    def test_operations(self):
        dynamics = Dynamics.from_sample(model=SIRFModel)
        data = dynamics.simulate().reset_index()
        data.insert(0, "Model", SIRFModel.name())
        engineer = DataEngineer(layers=["Model"], country=None, verbose=1)
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
        engineer.download()
        all_df = engineer.all()
        layer_df = engineer.layer()
        assert all_df.shape == layer_df.shape
        engineer.choropleth(geo=None, variable=Term.C, filename=imgfile)
        engineer.clean()
        engineer.transform()
        df = engineer.subset_alias(alias="Japan", geo="Japan")
        assert assert_frame_equal(engineer.subset_alias(alias="Japan"), df)
