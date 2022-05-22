#!/usr/bin/env python
# -*- coding: utf-8 -*-

from covsirphy.util.error import deprecate
from covsirphy.util.term import Term


class Estimator(Term):
    """
    Deprecated. Hyperparameter optimization of an ODE model.
    """

    @deprecate("Estimator", new="ODEHandler", version="2.19.1-zeta-fu1")
    def __init__(self, **kwargs):
        raise NotImplementedError("Please use ODEHandler()")


class Optimizer(Estimator):
    """
    This is deprecated. Please use Estimator class.
    """
    @deprecate("covsirphy.Estimator()", new="covsirphy.Estimator", version="2.17.0-eta")
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
