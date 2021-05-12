#!/usr/bin/env python
# -*- coding: utf-8 -*-

import warnings
import pandas as pd
import pytest
from covsirphy import find_args, save_dataframe, Filer, StopWatch, Evaluator
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
