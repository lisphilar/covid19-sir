#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pytest
import pandas as pd
from covsirphy import Term, LinelistData


class TestLinelistData(object):

    def test_raw(self, linelist_data):
        assert isinstance(linelist_data.raw, pd.DataFrame)

    def test_linelist(self, linelist_data):
        with pytest.raises(NotImplementedError):
            linelist_data.total()
        assert isinstance(linelist_data.cleaned(), pd.DataFrame)
        assert isinstance(linelist_data.citation, str)

    @pytest.mark.parametrize("country", ["Japan", "Germany"])
    @pytest.mark.parametrize("province", [None, "Tokyo"])
    def test_subset(self, linelist_data, country, province):
        if (country, province) == ("Germany", "Tokyo"):
            with pytest.raises(KeyError):
                linelist_data.subset(country=country, province=province)
        else:
            df = linelist_data.subset(country=country, province=province)
            column_set = set(df) | set([Term.COUNTRY, Term.PROVINCE])
            assert column_set == set(LinelistData.LINELIST_COLS)

    @pytest.mark.parametrize("outcome", ["Recovered", "Fatal", "Confirmed"])
    def test_closed(self, linelist_data, outcome):
        if outcome in ["Recovered", "Fatal"]:
            linelist_data.closed(outcome=outcome)
        else:
            with pytest.raises(KeyError):
                linelist_data.closed(outcome=outcome)

    def test_recovery_period(self, linelist_data):
        assert isinstance(linelist_data.recovery_period(), int)
