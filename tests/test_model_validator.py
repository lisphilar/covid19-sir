#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pytest
from covsirphy import ModelValidator, SIR


class TestModelValidator(object):

    @pytest.mark.parametrize("model", [SIR])
    def test_validation_sir(self, model):
        # Setting
        validator = ModelValidator(n_trials=4, step_n=10, seed=1)
        # Execute validation
        validator.run(model, timeout=10)
        validator.summary()
        with pytest.raises(ValueError):
            validator.run(model)
