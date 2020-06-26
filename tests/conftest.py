#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pytest
from covsirphy import DataLoader


@pytest.fixture(autouse=True)
def data_loader():
    return DataLoader("input")
