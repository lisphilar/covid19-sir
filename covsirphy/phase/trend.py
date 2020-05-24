#!/usr/bin/env python
# -*- coding: utf-8 -*-

from covsirphy.cleaning.word import Word


class Trend(Word):
    """
    S-R trend analysis in a phase.
    """

    def __init__(self, clean_df, population,
                 country, province=None, **kwargs):
        """
        @clean_df <pd.DataFrame>: cleaned data
            - index <int>: reseted index
            - Date <pd.TimeStamp>: Observation date
            - Country <str>: country/region name
            - Province <str>: province/prefecture/sstate name
            - Confirmed <int>: the number of confirmed cases
            - Infected <int>: the number of currently infected cases
            - Fatal <int>: the number of fatal cases
            - Recovered <int>: the number of recovered cases
        @population <int>: total population in the place
        @country <str>: country name
        @province <str>: province name
        @kwargs: the other keyword arguments of NondimData.make()
            - @start_date <str>: start date, like 22Jan2020
            - @end_date <str>: end date, like 01Feb2020
        """
        self.population = population
        self.country = country
        self.province = province
        self.train_df = None
