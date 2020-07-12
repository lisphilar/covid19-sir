#!/usr/bin/env python
# -*- coding: utf-8 -*-

from covsirphy.cleaning.term import Term


class ModelBaseCommon(Term):
    # Quartile range of the parametes when setting initial values
    QUANTILE_RANGE = [0.3, 0.7]
    # Model name
    NAME = "ModelBaseCommon"

    def __init__(self):
        # Dictionary of non-dim parameters: {name: value}
        self.non_param_dict = dict()

    def __str__(self):
        return self.NAME

    def __repr__(self):
        if not self.non_param_dict:
            return self.NAME
        param_str = ", ".join(
            [f"{p}={v}" for (p, v) in self.non_param_dict.items()]
        )
        return f"{self.NAME} model with {param_str}"

    def __getitem__(self, key):
        """
        Args:
            key (str): parameter name
        """
        if key not in self.non_param_dict.keys():
            raise KeyError(f"key must be in {', '.join(self.PARAMETERS)}")
        return self.non_param_dict[key]
