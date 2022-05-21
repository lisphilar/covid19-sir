#!/usr/bin/env python
# -*- coding: utf-8 -*-

from inspect import signature
import pandas as pd
from covsirphy.util.error import NAFoundError, NotIncludedError, NotSubclassError, UnExpectedTypeError, EmptyError
from covsirphy.util.error import UnExpectedValueRangeError, UnExpectedValueError


class Validator(object):
    """Validate objects and arguments.

    Args:
        target (object): target object to validate
        name (str): name of the target shown in error code
    """

    def __init__(self, target, name="target"):
        self._target = target
        self._name = str(name)

    def subclass(self, parent):
        """Ensure the target is a subclass of the parent class.

        Args:
            parent (object): parent class

        Raises:
            NotSubclassError: the target is not the subclass

        Returns:
            object: the target itself
        """
        if issubclass(self._target, parent):
            return self._target
        raise NotSubclassError(self._name, self._target, parent)

    def instance(self, expected):
        """Ensure that the target is an instance of a specified class.

        Args:
            expected (object): expected class

        Raises:
            UnExpectedTypeError: the target is not an instance of the class

        Returns:
            object: the target itself
        """
        if isinstance(self._target, expected):
            return self._target
        raise UnExpectedTypeError(self._name, self._target, expected)

    def dataframe(self, time_index=False, columns=None, empty_ok=True):
        """Ensure the target is a dataframe.

        Args:
            time_index (bool): if True, the dataframe must has DatetimeIndex
            columns (list[str] or None): the columns the dataframe must have
            empty_ok (bool): whether give permission to empty dataframe or not

        Raises:
            UnExpectedTypeError: the target is not a dataframe or that has un-expected-type index
            EmptyError: empty when @empty_ok is False
            NotIncludedError: expected columns were not included

        Returns:
            pandas.DataFrame: the target itself
        """
        if not isinstance(self._target, pd.DataFrame):
            raise UnExpectedTypeError(self._name, self._target, pd.DataFrame)
        df = self._target.copy()
        if not empty_ok and df.empty:
            raise EmptyError(name=self._name)
        if time_index and not isinstance(df.index, pd.DatetimeIndex):
            raise UnExpectedTypeError(f"Index of {self._name}", df.index, pd.DatetimeIndex)
        if columns is None:
            return df
        if not set(columns).issubset(df.columns):
            expected_cols = sorted(set(columns) - set(df.columns), key=columns.index)
            for col in expected_cols:
                raise NotIncludedError(
                    col, "column list of '{self._name}'", details="The dataframe has {', '.join(df.columns.tolist())} as columns.")
        return df

    def float(self, value_range=(0, None), default=None):
        """Convert a value to a float value.

        Args:
            value_range (tuple(int or None, int or None)): value range, None means un-specified
            default (float or None): default value when the target is None

        Raises:
            UnExpectedTypeError: the target cannot be converted to a float value
            UnExpectedValueRangeError: the value is out of value range

        Returns:
            float or None: converted float value or None (when both of the target and @default are None)
        """
        if self._target is None:
            return None if default is None else Validator(default, name="default").float(value_range=value_range)
        try:
            value = float(self._target)
        except ValueError:
            raise UnExpectedTypeError(self._name, self._target, float) from None
        if (value < (value_range[0] or value)) or (value > (value_range[1] or value)):
            raise UnExpectedValueRangeError(self._name, value, value_range)
        return value

    def int(self, value_range=(0, None), default=None, round_ok=False):
        """Convert a value to an integer.

        Args:
            value_range (tuple(int or None, int or None)): value range, None means un-specified
            default (int or None): default value when the target is None
            round_ok (bool): whether ignore round-off error

        Raises:
            UnExpectedTypeError: the target cannot be converted to an integer or round-off error exists when @round_ok is False
            UnExpectedValueRangeError: the value is out of value range

        Returns:
            int or None: converted float value or None (when both of the target and @default are None)
        """
        if self._target is None:
            return None if default is None else Validator(default, name="default").int(value_range=value_range, round_ok=round_ok)
        try:
            value = int(self._target)
        except ValueError:
            raise UnExpectedTypeError(self._name, self._target, int) from None
        if value != self._target and not round_ok:
            raise UnExpectedTypeError(
                self._name, self._target, int, details=f"This is because we cannot ignore round-off error, | {self._target} - {value} | > 0")
        if (value < (value_range[0] or value)) or (value > (value_range[1] or value)):
            raise UnExpectedValueRangeError(self._name, value, value_range)
        return value

    def tau(self, default=None):
        """Validate the value can be used as tau value [min].

        Args:
            default (int or None): default value when the target is None

        Raises:
            UnExpectedTypeError: the target cannot be converted to an integer
            UnExpectedValueRangeError: the value is out of value range

        Returns:
            int or None: converted float value or None (when both of the target and @default are None)
        """
        if self._target is None:
            return None if default is None else Validator(default, name="default").tau()
        value = self.int(value_range=(0, 1440), round_ok=False)
        if 1440 % value == 0:
            return value
        divisors = [str(i) for i in range(1, 1441) if 1440 % i == 0]
        raise UnExpectedValueError(
            self._name, value, divisors,
            details="Tau value [min], a divisor of 1440 [min] is a parameter used to convert actual time to time steps (without units)")

    def date(self, value_range=(None, None), default=None):
        """Convert a value to a date object.

        Args:
            value_range (tuple(int or None, int or None)): value range, None means un-specified
            default (pandas.Timestamp or None): default value when the target is None

        Raises:
            UnExpectedTypeError: the target cannot be converted to a date object
            UnExpectedValueRangeError: the value is out of value range

        Returns:
            pandas.Timestamp or None: converted date or None (when both of the target and @default are None)
        """
        if self._target is None:
            return None if default is None else Validator(default, name="default").date(value_range=value_range)
        if isinstance(self._target, pd.Timestamp):
            value = self._target.replace(hour=0, minute=0, second=0, microsecond=0)
        else:
            try:
                value = pd.to_datetime(self._target).replace(hour=0, minute=0, second=0, microsecond=0)
            except ValueError:
                raise UnExpectedTypeError(self._name, self._target, pd.Timestamp) from None
        if (value < (value_range[0] or value)) or (value > (value_range[1] or value)):
            raise UnExpectedValueRangeError(self._name, value, value_range)
        return value

    def sequence(self, default=None, flatten=False, unique=False, candidates=None):
        """Convert a sequence (list, tuple) to a list.

        Args:
            default (list[object] or None): default value when the target is None
            flatten (bool): whether flatten the sequence or not
            unique (bool): whether remove duplicated values or not, the first value will remain
            candidates (list[object] or tuple(object) or iter or None): list of candidates or None (no limitations)

        Raises:
            UnExpectedTypeError: the target cannot be converted to a list or failed in flattening
            UnExpectedValueError: the target has a value which is not included in the candidates

        Returns:
            list[object] or None: converted list or None (when both of the target and @default are None)
        """
        if self._target is None:
            return None if default is None else Validator(default, name="default").sequence(flatten=flatten, unique=unique, candidates=candidates)
        if not isinstance(self._target, (list, tuple)):
            raise UnExpectedTypeError(
                self._name, self._target, list, details="A tuple can be used, but it will be converted to a list.")
        if flatten:
            try:
                targets = sum(self._target, [])
            except TypeError:
                for value in [value for value in self._target if not isinstance(value, list)]:
                    raise UnExpectedTypeError(
                        f"A value of {self._name}", value, list, details="This is required to flatten the sequence") from None
        else:
            targets = list(self._target)
        if unique:
            targets = sorted(set(targets), key=targets.index)
        if candidates is None or set(targets).issubset(candidates):
            return targets
        for value in (set(targets) - set(candidates)):
            raise UnExpectedValueError(self._name, value, [str(c) for c in candidates])

    def dict(self, default=None, required_keys=None, errors="coerce"):
        """Ensure the target is a dictionary.

        Args:
            default (dict[str, object] or None): default value, when the target is None or key is not included in the target
            required_keys (list): keys which must be included
            errors (str): "coerce" or "raise"

        Raises:
            UnExpectedTypeError: the target is not a dictionary
            NAFoundError: values of the required keys are not specified when @errors="coerce"

        Returns:
            dict[str, object]: the target is self with default values and required keys

        Note:
            All keys of @default will be included and the target will overwrite it.

        Note:
            If some keys of @required_keys are not included and @errors="coerce", None will be set as the values of the keys.
        """
        if self._target is not None and not isinstance(self._target, dict):
            raise UnExpectedTypeError(self._name, self._target, dict)
        _dict = dict.fromkeys(required_keys or [])
        _dict.update(default or {})
        _dict.update(self._target or {})
        if required_keys is not None and errors != "coerce" and None in [_dict[key] for key in required_keys]:
            for key in [key for key in required_keys if _dict[key] is None]:
                raise NAFoundError(f"The value of key {key} in dictionary {self._name}")
        return _dict

    def kwargs(self, functions, default=None):
        """Find keyword arguments of the functions.

        Args:
            functions (list[function] or function): target functions
            default (dict[str, object] or None): default value when the target is None

        Raises:
            UnExpectedTypeError: the target is not a dictionary
            NAFoundError: values of the required keys are not specified when @errors="coerce"

        Returns:
            dict[str, object]: keyword arguments of the functions
        """
        _dict = self.dict(default=default, required_keys=None, errors="coerce")
        keywords_nest = [
            list(signature(func).parameters.keys()) for func in (functions if isinstance(functions, list) else [functions])]
        keywords_set = set(sum(keywords_nest, [])) - {"self", "cls"}
        return {k: v for k, v in _dict.items() if k in keywords_set}
