#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from covsirphy.cleaning.cbase import CleaningBase


class JHUData(CleaningBase):
    """
    Class for data cleaning of JHU/ dataset.
    """

    def __init__(self, filename):
        super().__init__(filename)

    def cleaning(self):
        """
        Perform data cleaing of the raw data.
        This method overwrite super().cleaning() method.
        @return <pd.DataFrame>
            - index <int>: reseted index
            - Date <pd.TimeStamp>: Observation date
            - Country <str>: country/region name
            - Province <str>: province/prefecture/sstate name
            - Confirmed <int>: the number of confirmed cases
            - Infected <int>: the number of currently infected cases
            - Fatal <int>: the number of fatal cases
            - Recovered <int>: the number of recovered cases
        """
        df = self._raw.copy()
        # Rename the columns
        df = df.rename(
            {
                "ObservationDate": "Date",
                "Province/State": "Province",
                "Deaths": "Fatal"
            },
            axis=1
        )
        # Datetime columns
        df["Date"] = pd.to_datetime(df["Date"])
        # Country
        df["Country"] = df["Country/Region"].replace(
            {
                "Mainland China": "China",
                "Hong Kong SAR": "Hong Kong",
                "Taipei and environs": "Taiwan",
                "Iran (Islamic Republic of)": "Iran",
                "Republic of Korea": "South Korea",
                "Republic of Ireland": "Ireland",
                "Macao SAR": "Macau",
                "Russian Federation": "Russia",
                "Republic of Moldova": "Moldova",
                "Taiwan*": "Taiwan",
                "Cruise Ship": "Others",
                "United Kingdom": "UK",
                "Viet Nam": "Vietnam",
                "Czechia": "Czech Republic",
                "St. Martin": "Saint Martin",
                "Cote d'Ivoire": "Ivory Coast",
                "('St. Martin',)": "Saint Martin",
                "Congo (Kinshasa)": "Congo",
                "Congo (Brazzaville)": "Congo",
                "The, Bahamas": "Bahamas",
            }
        )
        # Province
        df["Province"] = df["Province"].fillna("-").replace(
            {
                "Cruise Ship": "Diamond Princess",
                "Diamond Princess cruise ship": "Diamond Princess"
            }
        )
        df.loc[df["Country"] == "Diamond Princess", [
            "Country", "Province"]] = ["Others", "Diamond Princess"]
        # Values
        df["Infected"] = df["Confirmed"] - df["Fatal"] - df["Recovered"]
        df[self.VALUE_COLUMNS] = df[self.VALUE_COLUMNS].astype(np.int64)
        df = df.loc[:, self.COLUMNS].reset_index(drop=True)
        return df

    def replace(self, use_df, country, province=None):
        """
        Replace a part of cleaned dataset with a dataframe.
        """
        # TODO: replacement with other datasets
        raise Exception("JHUData.replace() will be coded later!")
