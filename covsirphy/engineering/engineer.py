#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
from covsirphy.util.error import UnExecutedError, UnExpectedValueError, SubsetNotFoundError
from covsirphy.util.validator import Validator
from covsirphy.util.term import Term
from covsirphy.engineering.cleaner import _DataCleaner
from covsirphy.engineering.transformer import _DataTransformer
from covsirphy.engineering.complement import _ComplementHandler


class DataEngineer(Term):
    """Class for data engineering including cleaning, transforming and complementing.

    Args:
        data (pandas.DataFrame): raw data
            Index
                reset index
            Column
                - columns defined by @layers
                - column defined by @date
                - the other columns
        layers (list[str]): location layers of the data
        date (str): column name of observation dates of the data
    """
    # Logs of complement
    VARIABLE = "Variable"
    DESCRIPTION = "Description"
    SCORE = "Score"
    # Default values of keyword arguments of complement
    DEFAULT_COMPLEMENT_KWARGS = {
        "recovery_period": 17,
        "interval": 2,
        "max_ignored": 100,
        "max_ending_unupdated": 14,
        "upper_limit_days": 90,
        "lower_limit_days": 7,
        "upper_percentage": 0.5,
        "lower_percentage": 0.5,
    }

    def __init__(self, data, layers, date="Date"):
        self._layers = Validator(layers, "layers").sequence()
        self._date = str(date)
        self._id_cols = [*self._layers, self._date]
        self._df = Validator(data, "data").dataframe(columns=[*self._layers, self._date])

    def all(self):
        """Return all available data, converting dtypes with pandas.DataFrame.convert_dtypes().

        Returns:
            pandas.DataFrame:
                Index
                    reset index
                Column
                    - columns defined by @layers of _DataEngineer()
                    - (pandas.Timestamp): observation dates defined by @date of _DataEngineer()
                    - the other columns
        """
        all_cols = self._df.columns.tolist()
        columns = self._id_cols + sorted(set(all_cols) - set(self._id_cols), key=all_cols.index)
        return self._df.reindex(columns=columns).convert_dtypes()

    def clean(self, kinds=None, **kwargs):
        """Perform data cleaning.

        Args:
            kinds (list[str] or None): kinds of data cleaning with order or None (all available kinds as follows)
                - "convert_date": Convert dtype of date column to pandas.Timestamp.
                - "resample": Resample records with dates.
                - "fillna": Fill NA values with '-' (layers) and the previous values and 0.
            **kwargs: keyword arguments of data cleaning refer to note

        Returns:
            DataEngineer: self

        Note:
            For "convert_date", keyword arguments of pandas.to_datetime() including "dayfirst (bool): whether date format is DD/MM or not" can be used.
        """
        cleaner = _DataCleaner(data=self._df, layers=self._layers, date=self._date)
        kind_dict = {
            "convert_date": cleaner.convert_date,
            "resample": cleaner.resample,
            "fillna": cleaner.fillna,
        }
        all_kinds = list(kind_dict.keys())
        selected = all_kinds if kinds is None else Validator(kinds, "kind").sequence(candidates=all_kinds)
        for kind in selected:
            try:
                kind_dict[kind](**Validator(kwargs, "keyword arguments").kwargs(functions=kind_dict[kind], default=None))
            except UnExecutedError:
                raise UnExecutedError(
                    "DataEngineer.clean(kinds=['convert_date'])",
                    details=f"Column {self._date} was not a column of date") from None
        self._df = cleaner.all()

    def transform(self, new_dict=None, **kwargs):
        """Transform the data, calculating the number of susceptible and infected cases.

        Args:
            new_dict (dict[str, str]): new column names (if not included, will not be calculated)
                susceptible (str): the number of susceptible cases
                infected (str): the number of infected cases
            kwargs (dict[str, str]): dictionary of existed columns
                - population (str): total population
                - confirmed (str): the number of confirmed cases
                - fatal (str): the number of fatal cases
                - recovered (str): the number of recovered cases

        Note:
            Susceptible = Population - Confirmed.

        Note:
            Infected = Confirmed - Fatal - Recovered.
        """
        transformer = _DataTransformer(data=self._df, layers=self._layers, date=self._date)
        method_dict = {
            "susceptible": transformer.susceptible,
            "infected": transformer.infected,
        }
        for (variable, new_column) in (new_dict or {}).items():
            if variable in method_dict:
                method_dict[variable](new=new_column, **kwargs)
        self._df = transformer.all()

    def diff(self, column, suffix="_diff", freq="D"):
        """Calculate daily new cases with "x(x>0) = F(x) - F(x-1), x(0) = 0 when F is cumulative numbers".

        Args:
            column (str): column name of the cumulative numbers
            suffix (str): suffix if the column (new column name will be '{column}{suffix}')
            freq (str): offset aliases of shifting dates

        Note:
            Regarding @freq, refer to https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases
        """
        transformer = _DataTransformer(data=self._df, layers=self._layers, date=self._date)
        transformer.diff(column=column, suffix=suffix, freq=freq)
        self._df = transformer.all()

    def complement_assess(self, address, col_dict, **kwargs):
        """Assess the data of a location and list-up necessary complement procedures.

        Args:
            address (list[str] or str): list/str to specify location, like ["Japan", "Tokyo"] when layers=["Country", "Prefecture"]
            col_dict (dict[str, str]): dictionary of column names
                - confirmed (str): column of the number of confirmed cases
                - fatal (str): column of the number of fatal cases
                - recovered (str): column of the number of recovered cases
                - tests (str): column of the number of tests
            kwargs: Keyword arguments of the following
                recovery_period (int): expected value of recovery period [days]
                interval (int): expected update interval of the number of recovered cases [days]
                max_ignored (int): max number of confirmed cases to be ignored [cases]
                max_ending_unupdated (int): Max number of days to apply full complement, where max recovered cases are not updated [days]
                upper_limit_days (int): maximum number of valid partial recovery periods [days]
                lower_limit_days (int): minimum number of valid partial recovery periods [days]
                upper_percentage (float): fraction of partial recovery periods with value greater than upper_limit_days
                lower_percentage (float): fraction of partial recovery periods with value less than lower_limit_days

        Raises:
            ValueError: the length of @address is not the same as @layers of DataEngineer
            SubsetNotFoundError: no records were found with the address

        Returns:
            list[tuple(str or list[str], str, dict[str])]: list of the combinations of variables (columns), rules and conditions

        Note:
            Available rules are
                - monotonic increasing complemented confirmed data,
                - monotonic increasing complemented fatal data,
                - monotonic increasing complemented recovered data,
                - monotonic increasing complemented tests data,
                - fully complemented recovered data,
                - partially complemented recovered data (internal),
                - partially complemented tests data (internal),
                - partially complemented recovered data (ending),
                - partially complemented tests data (ending).

        Note:
            Regarding @address, None is equal to '-' (NA).

        Note:
            Default values of **kwargs can be confirmed with class variable DataEngineer.DEFAULT_COMPLEMENT_KWARGS.
        """
        comp_kwargs = self.DEFAULT_COMPLEMENT_KWARGS.copy()
        comp_kwargs.update(kwargs)
        # Location
        address_converted = Validator([address] if isinstance(address, str) else address, "address").sequence()
        if len(address_converted) != len(self._layers):
            raise ValueError(f"@address ({address}) must have the same length with layers ({self._layers}), but not.")
        df = self._df.copy()
        for location, layer in zip(address_converted, self._layers):
            df = df.loc[df[layer] == (location or self.NA)]
        if df.empty:
            raise SubsetNotFoundError(geo=address)
        # Set-up handler with variables
        c_dict = dict.fromkeys(["confirmed", "fatal", "recovered", "tests"])
        c_dict.update(col_dict)
        cfr = [c_dict["confirmed"], c_dict["fatal"], c_dict["recovered"]]
        confirmed, _, recovered, tests = *cfr, c_dict["tests"]
        handler = _ComplementHandler(data=df, date=self._date)
        procedures = [
            ([col], f"monotonic increasing complemented {key} data", None)
            for (key, col) in c_dict.items() if col is not None and handler.assess_monotonic_increase(col)]
        if None not in cfr and handler.assess_recovered_full(*cfr, **comp_kwargs):
            procedures.append((cfr, "fully complemented recovered data", comp_kwargs))
        if None not in [confirmed, recovered] and handler.assess_partial_internal(recovered, **comp_kwargs):
            procedures.append(([confirmed, recovered], "partially complemented recovered data (internal)", comp_kwargs))
        if None not in [confirmed, tests] and handler.assess_partial_internal(tests, **comp_kwargs):
            procedures.append(([confirmed, tests], "partially complemented tests data (internal)", comp_kwargs))
        if None not in cfr and handler.assess_partial_ending(recovered, **comp_kwargs):
            procedures.append((cfr, "partially complemented recovered data (ending)", comp_kwargs))
        if tests is not None and handler.assess_partial_ending(tests, **comp_kwargs):
            procedures.append(([tests], "partially complemented tests data (ending)", comp_kwargs))
        return procedures

    def complement_force(self, address, procedures):
        """Perform complement per the operation procedures.

        Args:
            address (list[str] or str): list/str to specify location, like ["Japan", "Tokyo"] when layers=["Country", "Prefecture"]
            procedures (list[tuple(str, str)]): list of the combinations of variables (columns), rules and conditions

        Raises:
            ValueError: the length of @address is not the same as @layers of DataEngineer
            SubsetNotFoundError: no records were found with the address

        Returns:
            pandas.DataFrame: logs of complement
                Index
                    reset index
                Columns
                    - Variable (str): variable names
                    - Description (str): description of complement procedures
                    - Score (str): levels of complement (high score means more significant change) or 0 (not complemented)

        Note:
            Regarding rules, please refer to DataEngineer.complement_assess().

        Note:
            Scores are
            - "monotonic increasing complemented confirmed data": 1,
            - "monotonic increasing complemented fatal data": 1,
            - "monotonic increasing complemented recovered data": 1,
            - "monotonic increasing complemented tests data": 1,
            - "fully complemented recovered data": 3,
            - "partially complemented recovered data (internal)": 2,
            - "partially complemented tests data (internal)": 2,
            - "partially complemented recovered data (ending)": 2,
            - "partially complemented tests data (ending)": 2.
        """
        # Location
        address_converted = Validator([address] if isinstance(address, str) else address, "address").sequence()
        if len(address_converted) != len(self._layers):
            raise ValueError(f"@address ({address}) must have the same length with layers ({self._layers}), but not.")
        df, remain_df = self._df.copy(), self._df.copy()
        for location, layer in zip(address_converted, self._layers):
            df = df.loc[df[layer] == (location or self.NA)]
            remain_df = remain_df.loc[remain_df[layer] != (location or self.NA)]
        if df.empty:
            raise SubsetNotFoundError(geo=address)
        # Complements
        handler = _ComplementHandler(data=self._df, date=self._date)
        method_dict = {
            "monotonic increasing complemented confirmed data": {"method": handler.force_monotonic_increase, "score": 1},
            "monotonic increasing complemented fatal data": {"method": handler.force_monotonic_increase, "score": 1},
            "monotonic increasing complemented recovered data": {"method": handler.force_monotonic_increase, "score": 1},
            "monotonic increasing complemented tests data": {"method": handler.force_monotonic_increase, "score": 1},
            "fully complemented recovered data": {"method": handler.force_recovered_full, "score": 3},
            "partially complemented recovered data (internal)": {"method": handler.force_partial_internal, "score": 2},
            "partially complemented tests data (internal)": {"method": handler.force_partial_internal, "score": 2},
            "partially complemented recovered data (ending)": {"method": handler.force_recovered_partial_ending, "score": 2},
            "partially complemented tests data (ending)": {"method": handler.force_tests_partial_ending, "score": 2},
        }
        forced_dict = {}
        for i, (cols, description, comp_kwargs) in enumerate(procedures):
            if description not in method_dict:
                raise UnExpectedValueError(name="procedure", value=description, candidates=list(method_dict.keys()))
            is_forced = method_dict[description]["method"](*cols, **(comp_kwargs or {}))
            score = method_dict[description]["score"] if is_forced else 0
            forced_dict[i] = [cols[-1], description, score]
        # Logs
        log_columns = [self.VARIABLE, self.DESCRIPTION, self.SCORE]
        log_df = pd.DataFrame.from_dict(forced_dict, orient="index", columns=log_columns)
        self._df = pd.concat([handler.all(), remain_df], axis=0, ignore_index=True)
        return log_df
