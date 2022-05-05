#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pytest
from covsirphy import DataEngineer, UnExecutedError, SubsetNotFoundError


class TestDataEngineer(object):
    def test_clean(self, japan_df):
        engineer = DataEngineer(data=japan_df, layers=["Country", "Prefecture"], date="date")
        with pytest.raises(UnExecutedError):
            engineer.clean(kinds=["resample"])
        engineer.clean(kinds=None)
        df = engineer.all()
        assert not df.isna().values.any()
        assert df.columns.tolist() == ["Country", "Prefecture", "date", "Positive", "Tested", "Discharged", "Fatal"]

    def test_transform(self, c_df):
        df = c_df.copy()
        df["Population"] = 125_190_000
        engineer = DataEngineer(data=df, layers=["Country"], date="date")
        engineer.clean(kinds=None)
        new_dict = {
            "susceptible": "Susceptible", "infected": "Infected",
        }
        col_dict = {
            "population": "Population", "confirmed": "Positive", "fatal": "Fatal", "recovered": "Discharged",
        }
        engineer.transform(new_dict, **col_dict)
        df = engineer.all()
        assert {"Susceptible", "Infected"}.issubset(df.columns)

    def test_diff(self, c_df):
        df = c_df.copy()
        engineer = DataEngineer(data=df, layers=["Country"], date="date")
        engineer.clean(kinds=None)
        engineer.diff(column="Tested", suffix="_diff", freq="D")
        df = engineer.all()
        assert {"Tested_diff"}.issubset(df.columns)

    def test_complement(self, c_df):
        col_dict = {"confirmed": "Positive", "fatal": "Fatal", "recovered": "Discharged", "tests": "Tested"}
        engineer = DataEngineer(data=c_df, layers=["Country"], date="date")
        engineer.clean(kinds=None)
        with pytest.raises(ValueError):
            engineer.complement_assess(address=["Japan", "Tokyo"], col_dict=col_dict)
        with pytest.raises(SubsetNotFoundError):
            engineer.complement_assess(address=["Moon"], col_dict=col_dict)
        with pytest.raises(ValueError):
            engineer.complement_force(address=["Japan", "Tokyo"], procedures=None)
        with pytest.raises(SubsetNotFoundError):
            engineer.complement_force(address=["Moon"], procedures=None)
        procedures = engineer.complement_assess(address="Japan", col_dict=col_dict)
        log_df = engineer.complement_force(address="Japan", procedures=procedures)
        assert len(log_df) == len(procedures)
