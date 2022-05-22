#!/usr/bin/env python
# -*- coding: utf-8 -*-

from covsirphy.util.error import NotRegisteredError, SubsetNotFoundError
from covsirphy.util.validator import Validator
from covsirphy.util.term import Term
from covsirphy.gis.gis import GIS
from covsirphy.downloading.db_cs_japan import _CSJapan
from covsirphy.downloading.db_covid19dh import _COVID19dh
from covsirphy.downloading.db_google import _GoogleOpenData
from covsirphy.downloading.db_owid import _OWID


class DataDownloader(Term):
    """Class to download datasets from the recommended data servers.

    Args:
        directory (str or pathlib.Path): directory to save downloaded datasets
        update_interval (int): update interval of downloading dataset
        verbose (int): level of verbosity when downloading

    Note:
        If @verbose is 0, no descriptions will be shown.
        If @verbose is 1 or larger, URL and database name will be shown.

    Note:
        Location layers are fixed to ["ISO3", "Province", "City"]
    """
    LAYERS = [Term.ISO3, Term.PROVINCE, Term.CITY]

    def __init__(self, directory="input", update_interval=12, verbose=1):
        self._directory = directory
        self._update_interval = Validator(update_interval, "update_interval").int(value_range=(0, None))
        self._verbose = Validator(verbose, "verbose").int(value_range=(0, None))
        self._gis = GIS(layers=self.LAYERS, country=self.ISO3, date=self.DATE, verbose=self._verbose)

    def layer(self, country=None, province=None, databases=None):
        """Return the data at the selected layer.

        Args:
            country (str or None): country name or None
            province (str or None): province/state/prefecture name or None
            databases (list[str] or None): list of databases to use or None (all available databases).
                "japan": COVID-19 Dataset in Japan,
                "covid19dh": COVID-19 Data Hub,
                "google: COVID-19 Open Data by Google Cloud Platform,
                "owid": Our World In Data.

        Returns:
            pandas.DataFrame:
                Index
                    reset index
                Columns
                    - Date (pandas.Timestamp): observation date
                    - ISO3 (str): country names
                    - Province (str): province/state/prefecture names
                    - City (str): city names
                    - Country (str): country names (top level administration)
                    - Province (str): province names (2nd level administration)
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
                    - Mobility_grocery_and_pharmacy: % to baseline in visits (grocery markets, pharmacies etc.)
                    - Mobility_parks: % to baseline in visits (parks etc.)
                    - Mobility_transit_stations: % to baseline in visits (public transport hubs etc.)
                    - Mobility_retail_and_recreation: % to baseline in visits (restaurant, museums etc.)
                    - Mobility_residential: % to baseline in visits (places of residence)
                    - Mobility_workplaces: % to baseline in visits (places of work)

        Note:
            When @country is None, country-level data will be returned.

        Note:
            When @country is a string and @province is None, province-level data in the country will be returned.

        Note:
            When @country and @province are strings, city-level data in the province will be returned.
        """
        db_dict = {
            "japan": _CSJapan,
            "covid19dh": _COVID19dh,
            "google": _GoogleOpenData,
            "owid": _OWID,
        }
        all_databases = list(db_dict.keys())
        selected = Validator(databases, "databases").sequence(default=all_databases, candidates=all_databases)
        for database in selected:
            db = db_dict[database](
                directory=self._directory, update_interval=self._update_interval, verbose=self._verbose)
            new_df = db.layer(country=country, province=province).convert_dtypes()
            if new_df.empty:
                continue
            self._gis.register(
                data=new_df, layers=self.LAYERS, date=self.DATE, citations=db.CITATION, convert_iso3=False)
        try:
            return self._gis.layer(geo=(country, province))
        except NotRegisteredError:
            raise SubsetNotFoundError(geo=(country, province)) from None

    def citations(self, variables=None):
        """
        Return citation list of the data sources.

        Args:
            variables (list[str] or None): list of variables to collect or None (all available variables)

        Returns:
            list[str]: citation list
        """
        return self._gis.citations(variables=variables)
