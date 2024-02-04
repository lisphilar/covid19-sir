from __future__ import annotations
from pathlib import Path
import pandas as pd
from covsirphy.util.error import NotRegisteredError, SubsetNotFoundError
from covsirphy.util.validator import Validator
from covsirphy.util.term import Term
from covsirphy.gis.gis import GIS
from covsirphy.downloading._db_cs_japan import _CSJapan
from covsirphy.downloading._db_covid19dh import _COVID19dh
from covsirphy.downloading._db_owid import _OWID
from covsirphy.downloading._db_wpp import _WPP


class DataDownloader(Term):
    """Class to download datasets from the recommended data servers.

    Args:
        directory: directory to save downloaded datasets
        update_interval: update interval of downloading dataset

    Note:
        Location layers are fixed to ['ISO3', 'Province', 'City'].
    """
    LAYERS: list[str] = [Term.ISO3, Term.PROVINCE, Term.CITY]

    def __init__(self, directory: str | Path = "input", update_interval: int = 12, **kwargs) -> None:
        self._directory = directory
        self._update_interval = Validator(update_interval, "update_interval").int(value_range=(0, None))
        self._gis = GIS(layers=self.LAYERS, country=self.ISO3, date=self.DATE)

    def layer(self, country: str | None = None, province: str | None = None, databases: list[str] | None = None) -> pd.DataFrame:
        """Return the data at the selected layer.

        Args:
            country: country name or None
            province: province/state/prefecture name or None
            databases: databases to use or None (japan, covid19dh, owid).
                Candidates are as follows.

                - "japan": COVID-19 Dataset in Japan,
                - "covid19dh": COVID-19 Data Hub,
                - "owid": Our World In Data,
                - "wpp": World Population Prospects by United nations.

        Returns:
            A dataframe with reset index and the following columns.

                - Date (pandas.Timestamp): observation date
                - ISO3 (str): country names
                - Province (str): province/state/prefecture names
                - City (str): city names
                - Country (str): country names (the top level administration)
                - Province (str): province names (the 2nd level administration)
                - ISO3 (str): ISO3 codes
                - Confirmed (pandas.Int64): the number of confirmed cases
                - Fatal (pandas.Int64): the number of fatal cases
                - Recovered (pandas.Int64): the number of recovered cases
                - Population (pandas.Int64): population values
                - Tests (pandas.Int64): the number of tests
                - Product (pandas.Int64): vaccine product names
                - Vaccinations (pandas.Int64): cumulative number of vaccinations
                - Vaccinations_boosters (pandas.Int64): cumulative number of booster vaccinations
                - Vaccinated_once (pandas.Int64): cumulative number of people who received at least one vaccine dose
                - Vaccinated_full (pandas.Int64): cumulative number of people who received all doses prescribed by the protocol
                - School_closing
                - Workplace_closing
                - Cancel_events
                - Gatherings_restrictions
                - Transport_closing
                - Stay_home_restrictions
                - Internal_movement_restrictions
                - International_movement_restrictions
                - Information_campaigns
                - Testing_policy
                - Contact_tracing
                - Stringency_index

        Note:
            - When @country is None, country-level data will be returned.
            - When @country is a string and @province is None, province-level data in the country will be returned.
            - When @country and @province are strings, city-level data in the province will be returned.
        """
        db_dict = {
            "japan": _CSJapan,
            "covid19dh": _COVID19dh,
            "owid": _OWID,
            "wpp": _WPP,
        }
        all_databases = ["japan", "covid19dh", "owid"]
        selected = Validator(databases, "databases").sequence(default=all_databases, candidates=list(db_dict.keys()))
        self._gis = GIS(layers=self.LAYERS, country=self.ISO3, date=self.DATE)
        for database in selected:
            db = db_dict[database](
                directory=self._directory, update_interval=self._update_interval,)
            new_df = db.layer(country=country, province=province).convert_dtypes()
            if new_df.empty:
                continue
            self._gis.register(
                data=new_df, layers=self.LAYERS, date=self.DATE, citations=db.CITATION, convert_iso3=False)
        try:
            return self._gis.layer(geo=(country, province))
        except NotRegisteredError:
            raise SubsetNotFoundError(geo=(country, province)) from None

    def citations(self, variables: list[str] | None = None) -> list[str]:
        """
        Return citation list of the data sources.

        Args:
            variables: list of variables to collect or None (all available variables)

        Returns:
            citations
        """
        return self._gis.citations(variables=variables)
