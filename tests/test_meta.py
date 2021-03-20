#!/usr/bin/env python
# -*- coding: utf-8 -*-

import covsirphy as cs


class TestMeta(object):
    def test_version(self):
        assert isinstance(cs.__version__, str)
        assert isinstance(cs.get_version(), str)

    def test_citation(self):
        assert isinstance(cs.__citation__, str)
        assert isinstance(cs.get_citation(), str)
