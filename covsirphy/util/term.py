from __future__ import annotations
from collections import defaultdict
import logging
import warnings
from typing import ClassVar, Any
import country_converter as coco
import pandas as pd
from covsirphy.util.validator import Validator


class Term(object):
    """
    Term definition.
    """
    # Variables of SIR-derived model
    N: str = "Population"
    S: str = "Susceptible"
    C: str = "Confirmed"
    CI: str = "Infected"
    F: str = "Fatal"
    R: str = "Recovered"
    FR: str = "Fatal or Recovered"
    E: str = "Exposed"
    W: str = "Waiting"
    # PCR tests
    TESTS: str = "Tests"
    TESTS_DIFF: str = "Tests_diff"
    # Severity
    MODERATE: str = "Moderate"
    SEVERE: str = "Severe"
    # Vaccination
    VAC: str = "Vaccinations"
    V: str = "Vaccinated"
    V_ONCE: str = f"{V}_once"
    V_FULL: str = f"{V}_full"
    VAC_BOOSTERS: str = f"{VAC}_boosters"
    PRODUCT: str = "Product"
    # Column names
    DATE: str = "Date"
    START: str = "Start"
    END: str = "End"
    T: str = "Elapsed"
    TS: str = "t"
    TAU: str = "tau"
    COUNTRY: str = "Country"
    ISO3: str = "ISO3"
    PROVINCE: str = "Province"
    CITY: str = "City"
    STEP_N: str = "step_n"
    Y0_DICT: str = "y0_dict"
    PARAM_DICT: str = "param_dict"
    ID: str = "ID"
    _PH: str = "Phase_ID"
    _SIRF: list[str] = [S, CI, R, F]
    AREA_COLUMNS: list[str] = [COUNTRY, PROVINCE]
    STR_COLUMNS: list[str] = [DATE, *AREA_COLUMNS]
    COLUMNS: list[str] = [*STR_COLUMNS, C, CI, F, R]
    NLOC_COLUMNS: list[str] = [DATE, C, CI, F, R]
    SUB_COLUMNS: list[str] = [DATE, C, CI, F, R, S]
    VALUE_COLUMNS: list[str] = [C, CI, F, R]
    FIG_COLUMNS: list[str] = [CI, F, R, FR, V, E, W]
    MONO_COLUMNS: list[str] = [C, F, R]
    AREA_ABBR_COLS: list[str] = [ISO3, *AREA_COLUMNS]
    DSIFR_COLUMNS: list[str] = [DATE, S, CI, F, R]
    # Date format: 22Jan2020 etc.
    DATE_FORMAT: str = "%d%b%Y"
    DATE_FORMAT_DESC: str = "DDMmmYYYY"
    # Separator of country and province
    SEP: str = "/"
    # EDA
    RATE_COLUMNS: list[str] = [
        "Fatal per Confirmed",
        "Recovered per Confirmed",
        "Fatal per (Fatal or Recovered)"
    ]
    # Optimization
    A: str = "_actual"
    P: str = "_predicted"
    ACTUAL: str = "Actual"
    FITTED: str = "Fitted"
    # Phase name
    SUFFIX_DICT: ClassVar[defaultdict[int, str]] = defaultdict(lambda: "th")
    SUFFIX_DICT.update({1: "st", 2: "nd", 3: "rd"})
    # Summary of phases
    TENSE: str = "Type"
    PAST: str = "Past"
    FUTURE: str = "Future"
    INITIAL: str = "Initial"
    ODE: str = "ODE"
    RT: str = "Rt"
    RT_FULL: str = "Reproduction number"
    TRIALS: str = "Trials"
    RUNTIME: str = "Runtime"
    # Scenario analysis
    PHASE: str = "Phase"
    SERIES: str = "Scenario"
    MAIN: str = "Main"
    # Flag
    NA: str = "-"
    OTHERS: str = "Others"

    @classmethod
    def num2str(cls, num: int) -> str:
        """
        Convert numbers to 1st, 2nd etc.

        Args:
            num (int): number

        Returns:
            str
        """
        num = Validator(num, "num").int(value_range=(0, None))
        q, mod = divmod(num, 10)
        suffix = "th" if q % 10 == 1 else cls.SUFFIX_DICT[mod]
        return f"{num}{suffix}"

    @staticmethod
    def str2num(string: str, name: str = "phase names") -> int:
        """
        Convert 1st to 1 and so on.

        Args:
            string (str): like 1st, 2nd, 3rd,...
            name (str): name of the string

        Returns:
            int
        """
        try:
            return int(string[:-2])
        except ValueError as e:
            raise ValueError(
                f"Examples of {name} are 0th, 1st, 2nd..., but {string} was applied."
            ) from e

    @classmethod
    def _to_iso3(cls, name: str | list[str] | None) -> list[str]:
        """Convert country name(s) to ISO3 codes.

        Args:
            name (str or list[str] or None): country name(s)

        Returns:
            list[str]: ISO3 code(s)

        Note:
            "UK" will be converted to "GBR".

        Note:
            When the country was not found or None, it will not be converted.

        Examples:
            >>> Term._to_iso3("Japan")
            ['JPN']
            >>> Term._to_iso3("UK")
            ['GBR']
            >>> Term._to_iso3("Moon")
            ['Moon']
            >>> Term._to_iso3(None)
            ['---']
            >>> Term._to_iso3(["Japan", "UK", "Moon", None])
            ['JPN', 'GBR', 'Moon', '---']
        """
        logging.basicConfig(level=logging.CRITICAL)
        warnings.simplefilter("ignore", FutureWarning)
        names = [name] if (isinstance(name, str) or name is None) else name
        excepted_dict = {"UK": "GBR", None: cls.NA * 3}
        code_dict = {
            elem: excepted_dict[elem] if elem in excepted_dict else coco.convert(elem, to="ISO3", not_found=elem)
            for elem in set(names)
        }
        return [code_dict[elem] for elem in names]

    def _country_information(self) -> pd.DataFrame:
        """Return the raw data of country_converter library raw data as a dataframe.

        Returns:
            pandas.DataFrame:
                Index
                    reset index
                Columns:
                    - name_short: standard or short names
                    - ISO2: ISO2 codes
                    - ISO3: ISO3 codes
                    - Continent: continent names
                    - the other columns listed in country_converter library homepage.

        Note:
            Refer to https://github.com/konstantinstadler/country_converter
        """
        return coco.CountryConverter().data
