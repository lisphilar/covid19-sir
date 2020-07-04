#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import pytest
from covsirphy import ODESimulator, Word
from covsirphy import SIR, SIRD, SIRF


class TestODESimulator(object):
    @pytest.mark.parametrize(
        "model",
        # TODO: Add SIRFV and SEWIRF to solve #18
        [SIR, SIRD, SIRF]
    )
    def test_simulation(self, model):
        # Setting
        eg_tau = 1440
        start_date = "22Jan2020"
        # Simulation
        simulator = ODESimulator(
            country="Example", province=model.NAME
        )
        simulator.add(model=model, **model.EXAMPLE)
        simulator.run()
        nondim_df = simulator.non_dim()
        assert isinstance(nondim_df, pd.DataFrame)
        nondim_cols = [Word.TS, *list(model.VAR_DICT.keys())]
        assert set(nondim_df.columns) == set(nondim_cols)
        dim_df = simulator.dim(tau=eg_tau, start_date=start_date)
        assert isinstance(dim_df, pd.DataFrame)
        dim_cols = [*Word.STR_COLUMNS, *model.VARIABLES]
        assert set(dim_df.columns) == set(dim_cols)
