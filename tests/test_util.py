#!/usr/bin/env python
# -*- coding: utf-8 -*-

import warnings
import pandas as pd
import pytest
from covsirphy import find_args, save_dataframe, Filer, StopWatch, Evaluator, Term, Geography
from covsirphy import UnExpectedValueError


class TestArgument(object):
    def test_find_args(self):
        assert find_args(Filer, directory="output") == {"directory": "output"}
        assert find_args([Filer, Filer.files], directory="output") == {"directory": "output"}


class TestFiler(object):
    def test_filing(self):
        with pytest.raises(ValueError):
            Filer("output", numbering="xxx")
        filer = Filer(directory="output", prefix="jpn", suffix=None, numbering="01")
        # Create filenames
        filer.png("records")
        filer.jpg("records")
        filer.json("records")
        filer.csv("records", index=True)
        # Check files
        assert len(filer.files(ext=None)) == 4
        assert len(filer.files(ext="png")) == 1
        assert len(filer.files(ext="jpg")) == 1
        assert len(filer.files(ext="json")) == 1
        assert len(filer.files(ext="csv")) == 1
        # Save CSV file
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        save_dataframe(pd.DataFrame(), filename=None, index=False)


class TestStopWatch(object):
    def test_stopwatch(self):
        stop_watch = StopWatch()
        assert isinstance(stop_watch.stop_show(), str)


class TestEvaluator(object):

    @pytest.mark.parametrize("metric", ["ME", "MAE", "MSE", "MSLE", "MAPE", "RMSE", "RMSLE", "R2"])
    def test_score_series(self, metric):
        assert metric in Evaluator.metrics()
        true = pd.Series([5, 10, 8, 6])
        pred = pd.Series([8, 12, 6, 5])
        evaluator = Evaluator(true, pred, on=None)
        score_metric = evaluator.score(metric=metric)
        score_metrics = evaluator.score(metrics=metric)
        assert score_metric == score_metrics
        assert isinstance(Evaluator.smaller_is_better(metric=metric), bool)
        best_tuple = Evaluator.best_one({"A": 1.0, "B": 1.5, "C": 2.0}, metric=metric)
        if metric == "R2":
            assert best_tuple == ("C", 2.0)
        else:
            assert best_tuple == ("A", 1.0)

    @pytest.mark.parametrize("metric", ["ME", "MAE", "MSE", "MSLE", "MAPE", "RMSE", "RMSLE", "R2"])
    @pytest.mark.parametrize("how", ["all", "inner"])
    @pytest.mark.parametrize("on", [None, "join_on"])
    def test_score_dataframe(self, metric, how, on):
        true = pd.DataFrame(
            {
                "join_on": [0, 1, 2, 3, 4, 5],
                "value": [20, 40, 30, 50, 90, 10]
            }
        )
        pred = pd.DataFrame(
            {
                "join_on": [0, 2, 3, 4, 6, 7],
                "value": [20, 40, 30, 50, 110, 55]
            }
        )
        evaluator = Evaluator(true, pred, how=how, on=on)
        if metric == "ME" and (how == "all" or on is None):
            with pytest.raises(ValueError):
                evaluator.score(metric=metric)
            return
        assert isinstance(evaluator.score(metric=metric), float)

    def test_error(self):
        with pytest.raises(TypeError):
            Evaluator([1, 2, 3], [2, 5, 7])
        true = pd.Series([5, 10, 8, 6])
        pred = pd.Series([8, 12, 6, 5])
        evaluator = Evaluator(true, pred, on=None)
        with pytest.raises(UnExpectedValueError):
            evaluator.score(metric="Unknown")
        with pytest.raises(UnExpectedValueError):
            evaluator.smaller_is_better(metric="Unknown")


class TestGeography(object):
    def test_layer(self):
        day0, day1 = pd.to_datetime("2022-01-01"), pd.to_datetime("2022-01-02")
        raw = pd.DataFrame(
            {
                Term.COUNTRY: [*["UK" for _ in range(4)], *["Japan" for _ in range(10)]],
                Term.PROVINCE: [
                    "England", "England", *[Term.NA for _ in range(4)],
                    *["Tokyo" for _ in range(4)], *["Kanagawa" for _ in range(4)]],
                Term.CITY: [
                    *[Term.NA for _ in range(8)], "Chiyoda", "Chiyoda", "Yokohama", "Yokohama", "Kawasaki", "Kawasaki"],
                Term.DATE: [day0, day1] * 7,
                Term.C: range(14),
            }
        )
        geography = Geography(layers=[Term.COUNTRY, Term.PROVINCE, Term.CITY])
        # When `geo=None` or `geo=(None,)`, returns country-level data.
        df = pd.DataFrame(
            {
                Term.COUNTRY: ["UK", "UK", "Japan", "Japan"],
                Term.PROVINCE: [Term.NA for _ in range(4)],
                Term.CITY: [Term.NA for _ in range(4)],
                Term.DATE: [day0, day1] * 2,
                Term.C: [2, 3, 4, 5],
            }
        )
        assert geography.layer(data=raw, geo=None).equals(df)
        assert geography.layer(data=raw, geo=(None,)).equals(df)
        # When `geo=("Japan",)`, returns province-level data in Japan.
        df = pd.DataFrame(
            {
                Term.COUNTRY: ["Japan" for _ in range(2)],
                Term.PROVINCE: ["Tokyo", "Tokyo"],
                Term.CITY: [Term.NA for _ in range(2)],
                Term.DATE: [day0, day1],
                Term.C: [6, 7],
            }
        )
        assert geography.layer(data=raw, geo=("Japan",)).equals(df)
        # When `geo=(["Japan", "UK"],)`, returns province-level data in Japan and UK.
        df = pd.DataFrame(
            {
                Term.COUNTRY: ["UK", "UK", "Japan", "Japan"],
                Term.PROVINCE: ["England", "England", "Tokyo", "Tokyo"],
                Term.CITY: [Term.NA for _ in range(4)],
                Term.DATE: [day0, day1] * 2,
                Term.C: [0, 1, 6, 7],
            }
        )
        assert geography.layer(data=raw, geo=(["Japan", "UK"],)).equals(df)
        # When `geo=("Japan", "Kanagawa")`, returns city-level data in Kanagawa/Japan.
        df = pd.DataFrame(
            {
                Term.COUNTRY: ["Japan" for _ in range(4)],
                Term.PROVINCE: ["Kanagawa" for _ in range(4)],
                Term.CITY: ["Yokohama", "Yokohama", "Kawasaki", "Kawasaki"],
                Term.DATE: [day0, day1] * 2,
                Term.C: [10, 11, 12, 13],
            }
        )
        assert geography.layer(data=raw, geo=("Japan", "Kanagawa")).equals(df)
        # When `geo=("Japan", ["Tokyo", "Kanagawa"])`, returns city-level data in Tokyo/Japan and Kanagawa/Japan.
        df = pd.DataFrame(
            {
                Term.COUNTRY: ["Japan" for _ in range(6)],
                Term.PROVINCE: ["Tokyo", "Tokyo", *["Kanagawa" for _ in range(4)]],
                Term.CITY: ["Chiyoda", "Chiyoda", "Yokohama", "Yokohama", "Kawasaki", "Kawasaki"],
                Term.DATE: [day0, day1] * 3,
                Term.C: [8, 9, 10, 11, 12, 13],
            }
        )
        assert geography.layer(data=raw, geo=("Japan", ["Tokyo", "Kanagawa"])).equals(df)
        # Errors
        with pytest.raises(TypeError):
            geography.layer(data=raw, geo="a")
        with pytest.raises(TypeError):
            geography.layer(data=raw, geo=(1,))
        with pytest.raises(ValueError):
            geography.layer(data=raw, geo=("The Earth", "Japan", "Tokyo", "Chiyoda"))

    def test_filter(self):
        day0, day1 = pd.to_datetime("2022-01-01"), pd.to_datetime("2022-01-02")
        raw = pd.DataFrame(
            {
                Term.COUNTRY: [*["UK" for _ in range(4)], *["Japan" for _ in range(10)]],
                Term.PROVINCE: [
                    "England", "England", *[Term.NA for _ in range(4)],
                    *["Tokyo" for _ in range(4)], *["Kanagawa" for _ in range(4)]],
                Term.CITY: [
                    *[Term.NA for _ in range(8)], "Chiyoda", "Chiyoda", "Yokohama", "Yokohama", "Kawasaki", "Kawasaki"],
                Term.DATE: [day0, day1] * 7,
                Term.C: range(14),
            }
        )
        geography = Geography(layers=[Term.COUNTRY, Term.PROVINCE, Term.CITY])
        # When `geo = None` or `geo = (None,)`, returns all country-level data.
        df = pd.DataFrame(
            {
                Term.COUNTRY: ["UK", "UK", "Japan", "Japan"],
                Term.PROVINCE: [Term.NA for _ in range(4)],
                Term.CITY: [Term.NA for _ in range(4)],
                Term.DATE: [day0, day1] * 2,
                Term.C: [2, 3, 4, 5],
            }
        )
        assert geography.filter(data=raw, geo=None).equals(df)
        assert geography.filter(data=raw, geo=(None,)).equals(df)
        # When `geo = ("Japan",)`, returns country-level data in Japan.
        df = pd.DataFrame(
            {
                Term.COUNTRY: ["Japan", "Japan"],
                Term.PROVINCE: [Term.NA for _ in range(2)],
                Term.CITY: [Term.NA for _ in range(2)],
                Term.DATE: [day0, day1],
                Term.C: [4, 5],
            }
        )
        assert geography.filter(data=raw, geo=("Japan",)).equals(df)
        # When `geo = (["Japan", "UK"],)`, returns country-level data of Japan and UK.
        df = pd.DataFrame(
            {
                Term.COUNTRY: ["UK", "UK", "Japan", "Japan"],
                Term.PROVINCE: [Term.NA for _ in range(4)],
                Term.CITY: [Term.NA for _ in range(4)],
                Term.DATE: [day0, day1] * 2,
                Term.C: [2, 3, 4, 5],
            }
        )
        assert geography.filter(data=raw, geo=(["Japan", "UK"],)).equals(df)
        # When `geo = ("Japan", "Tokyo")`, returns province - level data of Tokyo/Japan.
        df = pd.DataFrame(
            {
                Term.COUNTRY: ["Japan", "Japan"],
                Term.PROVINCE: ["Tokyo", "Tokyo"],
                Term.CITY: [Term.NA for _ in range(2)],
                Term.DATE: [day0, day1],
                Term.C: [6, 7],
            }
        )
        assert geography.filter(data=raw, geo=("Japan", "Tokyo")).equals(df)
        # When `geo = ("Japan", ["Tokyo", "Kanagawa"])`, returns province-level data of Tokyo/Japan and Kanagawa/Japan.
        # Note that the example raw data does not include province-level data of Kanagawa/Japan.
        df = pd.DataFrame(
            {
                Term.COUNTRY: ["Japan", "Japan"],
                Term.PROVINCE: ["Tokyo", "Tokyo"],
                Term.CITY: [Term.NA for _ in range(2)],
                Term.DATE: [day0, day1],
                Term.C: [6, 7],
            }
        )
        assert geography.filter(data=raw, geo=("Japan", ["Tokyo", "Kanagawa"])).equals(df)
        # When `geo = ("Japan", "Kanagawa", "Yokohama")`, returns city-level data of Yokohama/Kanagawa/Japan.
        df = pd.DataFrame(
            {
                Term.COUNTRY: ["Japan", "Japan"],
                Term.PROVINCE: ["Kanagawa", "Kanagawa"],
                Term.CITY: ["Yokohama", "Yokohama"],
                Term.DATE: [day0, day1],
                Term.C: [10, 11],
            }
        )
        assert geography.filter(data=raw, geo=("Japan", "Kanagawa", "Yokohama")).equals(df)
        # When `geo = (("Japan", "Kanagawa", ["Yokohama", "Kawasaki"])`, returns city-level data of Yokohama/Kanagawa/Japan and Kawasaki / Kanagawa / Japan.
        df = pd.DataFrame(
            {
                Term.COUNTRY: ["Japan", "Japan", "Japan", "Japan"],
                Term.PROVINCE: ["Kanagawa", "Kanagawa", "Kanagawa", "Kanagawa"],
                Term.CITY: ["Yokohama", "Yokohama", "Kawasaki", "Kawasaki"],
                Term.DATE: [day0, day1] * 2,
                Term.C: [10, 11, 12, 13],
            }
        )
        assert geography.filter(data=raw, geo=("Japan", "Kanagawa", ["Yokohama", "Kawasaki"])).equals(df)
        # Errors
        with pytest.raises(TypeError):
            geography.filter(data=raw, geo="a")
        with pytest.raises(TypeError):
            geography.filter(data=raw, geo=(1,))
        with pytest.raises(ValueError):
            geography.filter(data=raw, geo=("The Earth", "Japan", "Tokyo", "Chiyoda"))
