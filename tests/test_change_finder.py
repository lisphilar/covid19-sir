#!/usr/bin/env python
# -*- coding: utf-8 -*-

from covsirphy import ChangeFinder


class TestChangeFinder(object):
    def test_find(self, data_loader):
        jhu_data = data_loader.jhu()
        clean_df = jhu_data.cleaned()
        population_data = data_loader.population()
        population = population_data.value("Italy")
        change_finder = ChangeFinder(
            clean_df, population, country="Italy"
        )
        change_finder.run()
        change_finder.show()
