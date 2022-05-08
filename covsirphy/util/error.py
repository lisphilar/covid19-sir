#!/usr/bin/env python
# -*- coding: utf-8 -*-

from copy import deepcopy
import warnings


def deprecate(old, new=None, version=None, ref=None):
    """
    Decorator to raise deprecation warning.

    Args:
        old (str): description of the old method/function
        new (str or None): description of the new method/function
        version (str or None): version number, like 2.7.3-alpha
        ref (str or None): reference URL of the new method/function
    """
    def _deprecate(func):
        def wrapper(*args, **kwargs):
            version_str = "." if version is None else f", version >= {version}."
            message = "" if ref is None else f" Refer to {ref}."
            if new is None:
                comment = f"{old} was deprecated{version_str}{message}"
            else:
                comment = f"Please use {new} rather than {old}{version_str}{message}"
            warnings.warn(
                comment,
                DeprecationWarning,
                stacklevel=2
            )
            return func(*args, **kwargs)
        return wrapper
    return _deprecate


class SubsetNotFoundError(KeyError, ValueError):
    """
    Error when subset was failed with specified arguments.

    Args:
        geo (tuple(list[str] or tuple(str) or str) or str or None): location names to filter or None
        country (str): country name
        country_alias (str or None): country name used in the dataset
        province (str or None): province name
        start_date (str or None): start date, like 22Jan2020
        end_date (str or None): end date, like 01Feb2020
        date (str or None): specified date, like 22Jan2020
        message (str or None): the other messages
    """

    def __init__(self, geo=None, country=None, country_alias=None, province=None,
                 start_date=None, end_date=None, date=None, message=None):
        self.area = self._area(geo, country, country_alias, province)
        self.date = self._date(start_date, end_date, date)
        self.message = "" if message is None else f" {message}"

    def __str__(self):
        return f"No records{self.message} in {self.area}{self.date} were found."

    @staticmethod
    def _area(geo, country, country_alias, province):
        """
        Error when subset was failed with specified arguments.

        Args:
            geo (tuple(list[str] or tuple(str) or str) or str or None): location names to filter or None
            country (str): country name
            country_alias (str or None): country name used in the dataset
            province (str or None): province name

        Returns:
            str: area name
        """
        if geo is None and country is None:
            return "the world"
        if geo is not None:
            geo_converted = deepcopy(geo)
        elif province is None:
            geo_converted = (country if country_alias is None else f"{country} ({country_alias})",)
        else:
            geo_converted = (country if country_alias is None else f"{country} ({country_alias})", province)
        names = [
            info if isinstance(info, str) else "_".join(list(info))
            for info in ([geo_converted] if isinstance(geo_converted, str) else geo_converted)]
        return "/".join(names[::-1])

    @staticmethod
    def _date(start_date, end_date, date):
        """
        Error when subset was failed with specified arguments.

        Args:
            start_date (str or None): start date, like 22Jan2020
            end_date (str or None): end date, like 01Feb2020
            date (str or None): specified date, like 22Jan2020
        """
        if date is not None:
            return f" on {date}"
        start_str = "" if start_date is None else f" from {start_date}"
        end_str = "" if end_date is None else f" to {end_date}"
        return f"{start_str}{end_str}"


class ScenarioNotFoundError(KeyError):
    """
    Error when unregistered scenario name was specified.

    Args:
        name (str): scenario name
    """

    def __init__(self, name):
        self.name = name

    def __str__(self):
        return f"{self.name} scenario is not registered."


class UnExecutedError(AttributeError, NameError, ValueError):
    """
    Error when we have unexecuted methods that we need to run in advance.

    Args:
        method_name (str): method name to run in advance
        message (str or None): the other messages
    """

    def __init__(self, method_name, message=None):
        self.method_name = method_name
        self.message = "." if message is None else f" {message}."

    def __str__(self):
        return f"Please execute {self.method_name} in advance{self.message}"


class NotRegisteredError(UnExecutedError):
    """
    Error when no records have been registered yet.
    """
    pass


class NotRegisteredMainError(UnExecutedError):
    """
    Error when main datasets were not registered.
    """
    pass


class NotRegisteredExtraError(UnExecutedError):
    """
    Error when extra datasets were not registered.
    """
    pass


class PCRIncorrectPreconditionError(KeyError):
    """
    Error when checking preconditions in the PCR data.

    Args:
        country (str): country name
        province (str or None): province name
        message (str or None): the other messages
    """

    def __init__(self, country, province=None, message=None):
        self.area = self._area(country, province)
        self.message = "" if message is None else f" {message}"

    def __str__(self):
        return f"{self.message}{self.area}."

    @staticmethod
    def _area(country, province):
        """
        Error when PCR preconditions failed with specified arguments.

        Args:
            country (str): country name
            province (str or None): province name

        Returns:
            str: area name
        """
        if province == "-":
            province = None
        country_str = "" if province else f" in country {country}"
        province_str = "" if province is None else f" in province {province}"
        return f"{province_str}{country_str}"


class NotInteractiveError(ValueError):
    """
    Error when interactive shell is not used but forced to use it.

    Args:
        message (str or None): the other messages
    """

    def __init__(self, message=None):
        self.message = "" if message is None else f" {message}"

    def __str__(self):
        return f"Interactive shell is not used.{self.message}"


class UnExpectedValueError(ValueError):
    """
    Error when unexpected value was applied as the value of an argument.

    Args:
        name (str): argument name
        value (object): value user applied
        candidates (list[object]): candidates of the argument
        message (str or None): the other messages
    """

    def __init__(self, name, value, candidates, message=None):
        self.name = str(name)
        self.value = str(value)
        self.candidates_str = ", ".join(candidates)
        self.message = "" if message is None else f" {message}"

    def __str__(self):
        s1 = f"@{self.name} must be selected from '{self.candidates_str}',"
        s2 = f"but {self.value} was applied.{self.message}"
        return f"{s1} {s2}"


class UnExpectedReturnValueError(ValueError):
    """
    Error when unexpected value was returned.

    Args:
        name (str): argument name
        value (object): value user applied or None (will not be shown)
        plural (bool): whether plural or not
        message (str or None): the other messages
    """

    def __init__(self, name, value, plural=False, message=None):
        self.name = str(name)
        self.value = "" if value is None else f" ({value})"
        self.s = "s" if plural else ""
        self.be = "were" if plural else "was"
        self.message = "" if message is None else f" {message}"

    def __str__(self):
        return f"Un-expected value{self.s}{self.value} {self.be} returned as {self.name}. {self.message}."


class DBLockedError(ValueError):
    """
    Error when a database has been locked not as expected.

    Args:
        name (str): database name
        message (str or None): the other messages
    """

    def __init__(self, name, message=None):
        self.name = str(name)
        self.message = "" if message is None else f" {message}."

    def __str__(self):
        return f"{self.name} should NOT be locked, but locked.{self.message}"


class NotDBLockedError(ValueError):
    """
    Error when a database has NOT been locked not as expected.

    Args:
        name (str): database name
        message (str or None): the other messages
    """

    def __init__(self, name, message=None):
        self.name = str(name)
        self.message = "" if message is None else f" {message}."

    def __str__(self):
        return f"{self.name} should be locked, but NOT locked.{self.message}"


class _BaseException(Exception):
    """Basic class of exception.

    Args:
        message (str): main message of error, should be set in child classes
        details (str or None): details of error
    """

    def __init__(self, message, details=None):
        self._message = str(message)
        self._details = "" if details is None else f" {details}."

    def __str__(self):
        return f"{self._message}. {self._details}"


class AlreadyCalledError(_BaseException):
    """Error when a method has already been called and cannot be called any more.

    Args:
        name (str): the name of the method
        details (str or None): details of error
    """

    def __init__(self, name, details=None):
        message = f"{name} has already been called and cannot be called any more"
        super().__init__(message=message, details=details)


class NotIncludedError(_BaseException, ValueError):
    """Error when a necessary key was not included in a dictionary.

    Args:
        key_name (str): key name
        dict_name (str): dictionary name
        details (str or None): details of error
    """

    def __init__(self, key_name, dict_name, details=None):
        message = f"'{key_name}' was not included in the dictionary '{dict_name}'"
        super().__init__(message=message, details=details)
