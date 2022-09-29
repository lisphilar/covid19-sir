#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pytest
import covsirphy as cs


class TestMeta(object):
    def test_version(self):
        assert isinstance(cs.__version__, str)
        assert isinstance(cs.get_version(), str)

    def test_citation(self):
        assert isinstance(cs.__citation__, str)
        assert isinstance(cs.get_citation(), str)

    def test_config(self):
        cs.config.logger(level=1)
        with pytest.raises(KeyError):
            cs.config.logger(level=-1)
