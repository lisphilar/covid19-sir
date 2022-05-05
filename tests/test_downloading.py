#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pytest
from covsirphy import DataDownloader


class TestDataDownloader(object):
    @pytest.mark.parametrize(
        "country, province",
        [
            ("Japan", None),
            ("USA", None),
            ("USA", "Alabama"),
            ("UK", None),
            (None, None),
        ]
    )
    def test_download(self, country, province):
        downloader = DataDownloader()
        downloader.layer(country=country, province=province)
        assert downloader.citations()
