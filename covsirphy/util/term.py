from collections import defaultdict
import logging
import warnings
import country_converter as coco
from covsirphy.util.validator import Validator


class Term(object):
    """
    Term definition.
    """
    # Variables of SIR-derived model
    N = "Population"
    S = "Susceptible"
    C = "Confirmed"
    CI = "Infected"
    F = "Fatal"
    R = "Recovered"
    FR = "Fatal or Recovered"
    E = "Exposed"
    W = "Waiting"
    # PCR tests
    TESTS = "Tests"
    TESTS_DIFF = "Tests_diff"
    # Severity
    MODERATE = "Moderate"
    SEVERE = "Severe"
    # Vaccination
    VAC = "Vaccinations"
    V = "Vaccinated"
    V_ONCE = f"{V}_once"
    V_FULL = f"{V}_full"
    VAC_BOOSTERS = f"{VAC}_boosters"
    PRODUCT = "Product"
    # Column names
    DATE = "Date"
    START = "Start"
    END = "End"
    T = "Elapsed"
    TS = "t"
    TAU = "tau"
    COUNTRY = "Country"
    ISO3 = "ISO3"
    PROVINCE = "Province"
    CITY = "City"
    STEP_N = "step_n"
    Y0_DICT = "y0_dict"
    PARAM_DICT = "param_dict"
    ID = "ID"
    _PH = "Phase_ID"
    _SIRF = [S, CI, R, F]
    AREA_COLUMNS = [COUNTRY, PROVINCE]
    STR_COLUMNS = [DATE, *AREA_COLUMNS]
    COLUMNS = [*STR_COLUMNS, C, CI, F, R]
    NLOC_COLUMNS = [DATE, C, CI, F, R]
    SUB_COLUMNS = [DATE, C, CI, F, R, S]
    VALUE_COLUMNS = [C, CI, F, R]
    FIG_COLUMNS = [CI, F, R, FR, V, E, W]
    MONO_COLUMNS = [C, F, R]
    AREA_ABBR_COLS = [ISO3, *AREA_COLUMNS]
    DSIFR_COLUMNS = [DATE, S, CI, F, R]
    # Date format: 22Jan2020 etc.
    DATE_FORMAT = "%d%b%Y"
    DATE_FORMAT_DESC = "DDMmmYYYY"
    # Separator of country and province
    SEP = "/"
    # EDA
    RATE_COLUMNS = [
        "Fatal per Confirmed",
        "Recovered per Confirmed",
        "Fatal per (Fatal or Recovered)"
    ]
    # Optimization
    A = "_actual"
    P = "_predicted"
    ACTUAL = "Actual"
    FITTED = "Fitted"
    # Phase name
    SUFFIX_DICT = defaultdict(lambda: "th")
    SUFFIX_DICT.update({1: "st", 2: "nd", 3: "rd"})
    # Summary of phases
    TENSE = "Type"
    PAST = "Past"
    FUTURE = "Future"
    INITIAL = "Initial"
    ODE = "ODE"
    RT = "Rt"
    RT_FULL = "Reproduction number"
    TRIALS = "Trials"
    RUNTIME = "Runtime"
    # Scenario analysis
    PHASE = "Phase"
    SERIES = "Scenario"
    MAIN = "Main"
    # Flag
    NA = "-"
    OTHERS = "Others"

    @classmethod
    def num2str(cls, num):
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
    def str2num(string, name="phase names"):
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
    def _to_iso3(cls, name):
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

    def _country_information(self):
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
