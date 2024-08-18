import pandas as pd
from covsirphy.util.term import Term
from covsirphy.downloading._db import _DataBase


class _WPP(_DataBase):
    """
    Access "World Population Prospects by United nations" server.
    https://population.un.org/wpp/
    https://datahelpdesk.worldbank.org/knowledgebase/articles/898581-api-basic-call-structures
    """
    TOP_URL = "https://api.worldbank.org/v2/"
    # File title without extensions and suffix
    TITLE = "world-population-prospects"
    # Dictionary of column names
    COL_DICT = {
        "Time": "Year",
        "ISO3_code": Term.ISO3,
        "PopTotal": Term.N,
    }
    ALL_COLS = [Term.DATE, Term.ISO3, Term.PROVINCE, Term.CITY, Term.N]
    # Stdout when downloading (shown at most one time)
    STDOUT = "Retrieving datasets from World Population Prospects https://population.un.org/wpp/"
    # Citations
    CITATION = 'United Nations, Department of Economic and Social Affairs,' \
        ' Population Division (2022). World Population Prospects 2022, Online Edition.'

    def _country(self):
        """Returns country-level data.

        Returns:
            pandas.DataFrame:
                Index
                    reset index
                Columns
                    - Date (pandas.Timestamp): observation date
                    - ISO3 (str): country names
                    - Province (object): NAs
                    - City (object): NAs
                    - Population (numpy.float64): population values
        """
        url = f"{self.TOP_URL}country/all/indicator/SP.POP.TOTL?per_page=20000"
        df = pd.read_xml(url,  parser="etree")
        df[self.DATE] = pd.to_datetime(df["date"], format="%Y") + pd.offsets.DateOffset(months=6)
        df = df.rename(columns={"countryiso3code": Term.ISO3})
        df[self.PROVINCE] = self.NA
        df[self.CITY] = self.NA
        df[self.N] = df["value"]
        return df.loc[:, self.ALL_COLS].dropna(how="any")

    def _province(self, country):
        """Returns province-level data.

        Args:
            country (str): country name

        Returns:
            pandas.DataFrame:
                Index
                    reset index
                Columns
                    - Date (pandas.Timestamp): observation date
                    - ISO3 (str): country names
                    - Province (object): NAs
                    - City (object): NAs
                    - Population (numpy.float64): population values
        """
        return pd.DataFrame(columns=self.ALL_COLS)

    def _city(self, country, province):
        """Returns city-level data.

        Args:
            country (str): country name
            province (str): province/state/prefecture name

        Returns:
            pandas.DataFrame:
                Index
                    reset index
                Columns
                    - Date (pandas.Timestamp): observation date
                    - ISO3 (str): country names
                    - Province (object): NAs
                    - City (object): NAs
                    - Population (numpy.float64): population values
        """
        return pd.DataFrame(columns=self.ALL_COLS)
