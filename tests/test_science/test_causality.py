#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pytest
from covsirphy import Causality, SIRFModel, DataEngineer, Term


@pytest.fixture(scope="module")
def snr():
    return Causality.auto_build(geo="Japan", model=SIRFModel)


class TestCausality(object):
    def test_predict(self, snr):
        engineer = DataEngineer()
        engineer.download()
        engineer.clean()
        X, *_ = engineer.subset(geo="Japan", variables=[Term.TESTS, Term.VAC])
        snr.predict(days=30, name="Baseline", X=X)
        snr.rename(old="Baseline_Multivariate_Likely", new="Likely")
        snr.represent(q=(0.1, 0.9), variable="Confirmed", excluded=["Baseline", "Likely"])
