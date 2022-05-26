#!/usr/bin/env python
# -*- coding: utf-8 -*-

from covsirphy import DataEngineer, Term, Dynamics, SIRF


class TestDataEngineer(object):
    def test_from_sample_data(self):
        dynamics = Dynamics.from_sample(model=SIRF)
        data = dynamics.simulate()
        data.insert(0, "Model", SIRF.NAME)
        engineer = DataEngineer(layers=["Model"], country=None, verbose=1)
        engineer.register(data, citations="Simulated data")
        assert set(engineer.all().columns) == {"Model", Term.DATE, Term.S, Term.CI, Term.F, Term.R}
        assert engineer.citations() == ["Simulated data"]
        engineer.transform_inverse()
        assert set(engineer.all().columns) == {"Model", Term.DATE, Term.S, Term.CI, Term.F, Term.R, Term.N, Term.C}
