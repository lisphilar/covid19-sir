#!/usr/bin/env python
# -*- coding: utf-8 -*-

from datetime import datetime
import pytest
import pandas as pd
from pandas.testing import assert_frame_equal
from covsirphy import Validator, ModelBase, SIR, SIRF
from covsirphy import NAFoundError, NotIncludedError, NotSubclassError, UnExpectedTypeError, EmptyError
from covsirphy import UnExpectedValueRangeError, UnExpectedValueError, UnExpectedLengthError


class TestValidator(object):
    def test_subclass(self):
        v = Validator(SIRF, name="model")
        with pytest.raises(NotSubclassError):
            v.subclass(SIR)
        assert v.subclass(ModelBase) == SIRF

    def test_instance(self):
        v = Validator("covsirphy")
        with pytest.raises(UnExpectedTypeError):
            v.instance(int)
        assert v.instance(str) == "covsirphy"

    def test_dataframe(self):
        with pytest.raises(UnExpectedTypeError):
            Validator("string").dataframe()
        with pytest.raises(EmptyError):
            Validator(pd.DataFrame()).dataframe(empty_ok=False)
        df = pd.DataFrame({"A": [1, 2], "B": [3, 4], "C": pd.date_range("01Jan2022", "02Jan2022", freq="D")})
        v = Validator(df, "dataframe")
        with pytest.raises(UnExpectedTypeError):
            v.dataframe(time_index=True)
        with pytest.raises(NotIncludedError):
            v.dataframe(columns=["D"])
        assert_frame_equal(v.dataframe(), df)
        assert_frame_equal(v.dataframe(columns=["A"]), df)

    def test_float(self):
        assert Validator(None).float(default=1.2) == 1.2
        with pytest.raises(UnExpectedTypeError):
            Validator("string").float()
        with pytest.raises(UnExpectedValueRangeError):
            Validator(1.2).float(value_range=(0, 1.1))
        with pytest.raises(UnExpectedValueRangeError):
            Validator(1.2).float(value_range=(1.5, 2))
        assert Validator(1.2).float() == 1.2

    def test_int(self):
        assert Validator(None).int(default=2) == 2
        with pytest.raises(UnExpectedTypeError):
            Validator("string").int()
        with pytest.raises(UnExpectedTypeError):
            Validator(1.2).int()
        with pytest.raises(UnExpectedValueRangeError):
            Validator(2).int(value_range=(0, 1))
        with pytest.raises(UnExpectedValueRangeError):
            Validator(2).int(value_range=(3, 4))
        assert Validator(2.5).int(round_ok=True) == 2

    def test_tau(self):
        assert Validator(None).tau(default=720) == 720
        with pytest.raises(UnExpectedValueError):
            Validator(11).tau()
        assert Validator(360).tau() == 360

    def test_date(self):
        assert Validator(None).date(default=pd.Timestamp("01Jan2022")) == pd.Timestamp("01Jan2022")
        assert Validator(pd.Timestamp("01Jan2022")).date() == pd.Timestamp("01Jan2022")
        assert Validator(datetime(year=2022, month=1, day=1)).date() == pd.Timestamp("01Jan2022")
        assert Validator("01Jan2022").date() == pd.Timestamp("01Jan2022")
        with pytest.raises(UnExpectedTypeError):
            Validator("hello").date()
        v = Validator("2022-01-01")
        with pytest.raises(UnExpectedValueRangeError):
            v.date(value_range=(None, pd.Timestamp("2021-12-31")))
        with pytest.raises(UnExpectedValueRangeError):
            v.date(value_range=(pd.Timestamp("2022-02-01"), None))

    def test_sequence(self):
        assert Validator(None).sequence(default=[1, 2]) == [1, 2]
        with pytest.raises(UnExpectedTypeError):
            Validator(1).sequence()
        with pytest.raises(UnExpectedTypeError):
            Validator([[1, 2], 2]).sequence(flatten=True)
        assert Validator([[1, 2], [3, 4]]).sequence(flatten=True) == [1, 2, 3, 4]
        assert Validator([1, 2, 4, 2, 2]).sequence(unique=True) == [1, 2, 4]
        v = Validator([1, 2, 3])
        with pytest.raises(UnExpectedValueError):
            v.sequence(candidates=[1, 4, 5])
        assert v.sequence(candidates=[1, 2, 3, 4]) == [1, 2, 3]
        with pytest.raises(UnExpectedLengthError):
            v.sequence(length=2)
        assert v.sequence(length=3) == [1, 2, 3]

    def test_dict(self):
        assert Validator(None).dict(default={1: 2}) == {1: 2}
        with pytest.raises(UnExpectedTypeError):
            Validator(1).dict()
        with pytest.raises(NAFoundError):
            Validator({1: 2}).dict(required_keys=[3, 4], errors="raise")
        assert Validator({1: 2}).dict(default={3: 5, 6: 8}, required_keys=[3, 4]) == {1: 2, 3: 5, 6: 8, 4: None}

    def test_kwargs(self):
        kwargs = {"target": SIR, "unnecessary_key": True}
        v = Validator(kwargs)
        result_dict = v.kwargs(functions=Validator, default={"name": "target name"})
        assert isinstance(Validator(**result_dict), Validator)
