#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pytest
from covsirphy import DataDownloader, SubsetNotFoundError


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

    @pytest.mark.parametrize(
        "country, province",
        [
            ("UK", "London Region"),
        ]
    )
    def test_download_error(self, country, province):
        downloader = DataDownloader()
        with pytest.raises(SubsetNotFoundError):
            downloader.layer(country=country, province=province)
