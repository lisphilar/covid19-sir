#!/usr/bin/env python
# -*- coding: utf-8 -*-

from copy import deepcopy
from covsirphy.util.config import config


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
            config.warning(message=comment, category=DeprecationWarning)
            return func(*args, **kwargs)
        return wrapper
    return _deprecate


def experimental(name, version):
    """
    Decorator to raise ExperimentalWarning because the method/function is experimental.

    Args:
        name (str): description of the method/function
        version (str): version number, like 2.7.3-alpha
    """
    def _experimental(func):
        def wrapper(*args, **kwargs):
            comment = f"{name} can be used from {version}, but this is experimental." \
                "Its name and arguments may be changed later."
            config.warning(message=comment, category=ExperimentalWarning)
            return func(*args, **kwargs)
        return wrapper
    return _experimental


class _BaseWarning(Warning):
    """Basic class of warning.
    """
    pass


class ExperimentalWarning(Warning):
    """Class to explain the method/function is experimental and its name,
    features and arguments may changed later.
    """
    pass


class _BaseException(Exception):
    """Basic class of exception.

    Args:
        message (str): main message of error, should be set in child classes
        details (str or None): details of error
        log (str): description used by logger
    """

    def __init__(self, message, details=None, log="exception raised"):
        config.error(log)
        self._message = str(message)
        self._details = "" if details is None else f" {details}."

    def __str__(self):
        return f"{self._message}. {self._details}"


class _ValidationError(_BaseException):
    """Basic class of exception raised when validation.

    Args:
        name (str): name of the target
        message (str): main message of error, should be set in child classes
        details (str or None): details of error
        log (str): description used by logger
    """

    def __init__(self, name, message, details=None):
        log = f"validation of {name} failed"
        super().__init__(message=message, details=details, log=log)


class AlreadyCalledError(_BaseException):
    """Error when a method has already been called and cannot be called any more.

    Args:
        name (str): the name of the method
        details (str or None): details of error
    """

    def __init__(self, name, details=None):
        message = f"{name} has already been called and cannot be called any more"
        super().__init__(message=message, details=details)


class NotIncludedError(_ValidationError):
    """Error when a necessary key was not included in a container.

    Args:
        key_name (str): key name
        container_name (str): name of the container
        details (str or None): details of error
    """

    def __init__(self, key_name, container_name, details=None):
        message = f"'{key_name}' was not included in the '{container_name}'"
        super().__init__(name=key_name, message=message, details=details)


class NAFoundError(_ValidationError):
    """Error when NA values are included un-expectedly.

    Args:
        name (str): name of the target
        value (str or None): value of the target
        details (str or None): details of error
    """

    def __init__(self, name, value=None, details=None):
        message = f"'{name}' has NA(s) un-expectedly"
        if value is not None:
            message += f", '{value}'"
        super().__init__(name=name, message=message, details=details)


class NotEnoughDataError(_ValidationError):
    """Error when we do not have enough data for analysis.

    Args:
        name (str): name of the target
        value (str): value of the target
        required_n (int): required number of records
        details (str or None): details of error
    """

    def __init__(self, name, value, required_n, details=None):
        message = f"We need more than {required_n} records, but '{name}' has only {len(value)} records at this time"
        super().__init__(name=name, message=message, details=details)


class UnExpectedNoneError(_ValidationError):
    """Error when a value is None un-expectedly.

    Args:
        name (str): name of the target
        details (str or None): details of error
    """

    def __init__(self, name, details=None):
        message = f"'{name}' is None un-expectedly"
        super().__init__(name=name, message=message, details=details)


class NotNoneError(_ValidationError):
    """Error when a value must be None but not None un-expectedly.

    Args:
        name (str): name of the target
        value (str): value of the target
        details (str or None): details of error
    """

    def __init__(self, name, value, details=None):
        message = f"'{name}' must be None, but has value '{value}'"
        super().__init__(name=name, message=message, details=details)


class UnExecutedError(_BaseException):
    """
    Error when we have unexecuted methods that we need to run in advance.

    Args:
        name (str): method name to run in advance
        details (str or None): details of error
    """

    def __init__(self, name, details=None):
        message = f"Please execute {name} in advance"
        log = f"{name} not executed"
        super().__init__(message=message, details=details, log=log)


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


class NotSubclassError(_ValidationError):
    """Error when an object is not a subclass of the parent class un-expectedly.

    Args:
        name (str): name of the target
        target (object): target object
        parent (object): expected parent class
        details (str or None): details of error
    """

    def __init__(self, name, target, parent, details=None):
        message = f"'{name}' must be a sub-class of {parent}, but {type(target)} was applied"
        super().__init__(name=name, message=message, details=details)


class UnExpectedTypeError(_ValidationError):
    """Error when an object cannot be converted to an instance un-expectedly.

    Args:
        name (str): name of the target
        target (object): target object
        expected (object): expected type
        details (str or None): details of error
    """

    def __init__(self, name, target, expected, details=None):
        message = f"We could not convert '{name}' to an instance of {expected} because that of {type(target)} was applied"
        super().__init__(name=name, message=message, details=details)


class EmptyError(_ValidationError):
    """Error when the dataframe is empty un-expectedly.

    Args:
        name (str): name of the target
        details (str or None): details of error
    """

    def __init__(self, name, details=None):
        message = f"'Empty dataframe/series was applied as {name}' un-expectedly"
        super().__init__(name=name, message=message, details=details)


class UnExpectedValueRangeError(_ValidationError):
    """Error when the value is out of value range.

    Args:
        name (str): name of the target
        target (object): target object
        value_range (tuple(int or None, int or None)): value range, None means un-specified
        details (str or None): details of error
    """

    def __init__(self, name, target, value_range, details=None):
        _min, _max = value_range
        if _min is None:
            s = "is not in the expected value range" if _max is None else f"must be under or equal to {_max}"
        else:
            s = f"must be over or equal to {_min}" if _max is None else f"is not in the expected value range ({_min}, {_max})"
        message = f"'{name}' {s}, but {target} was applied"
        super().__init__(name=name, message=message, details=details)


class UnExpectedValueError(_ValidationError):
    """
    Error when unexpected value was applied as the value of an argument.

    Args:
        name (str): argument name
        value (object): value user applied
        candidates (list[object]): candidates of the argument
        details (str or None): details of error
    """

    def __init__(self, name, value, candidates, details=None):
        c_str = ", ".join(candidates)
        message = f"'{name}' must be selected from [{c_str}], but {value} was applied"
        super().__init__(name=name, message=message, details=details)


class UnExpectedLengthError(_ValidationError):
    """
    Error when a sequence has un-expended length.

    Args:
        name (str): argument name
        value (object): value user applied
        length (int): length of the sequence
        details (str or None): details of error
    """

    def __init__(self, name, value, length, details=None):
        message = f"The length of '{name}' must be {length}, but {len(value)} was applied"
        super().__init__(name=name, message=message, details=details)


class SubsetNotFoundError(_BaseException):
    """Error when subset was failed with specified arguments.

    Args:
        geo (tuple(list[str] or tuple(str) or str) or str or None): location names to filter or None
        country (str): country name
        country_alias (str or None): country name used in the dataset
        province (str or None): province name
        start_date (str or None): start date, like 22Jan2020
        end_date (str or None): end date, like 01Feb2020
        date (str or None): specified date, like 22Jan2020
        details (str or None): details of error
    """

    def __init__(self, geo=None, country=None, country_alias=None, province=None,
                 start_date=None, end_date=None, date=None, details=None):
        self.area = self._area(geo, country, country_alias, province)
        self.date = self._date(start_date, end_date, date)
        message = f"No records in {self.area}{self.date} were found"
        log = "data subsetting failed"
        super().__init__(message=message, details=details, log=log)

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
            info if isinstance(info, str) else "-" if info is None else "_".join(list(info))
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


class ScenarioNotFoundError(_BaseException):
    """Error when unregistered scenario name was specified.

    Args:
        name (str): scenario name
        details (str or None): details of error
    """

    def __init__(self, name, details=None):
        message = f"{name} scenario is not registered"
        log = "scenario selection failed"
        super().__init__(message=message, details=details, log=log)


class PCRIncorrectPreconditionError(_BaseException):
    """Error when checking preconditions in the PCR data.

    Args:
        country (str): country name
        province (str or None): province name
        details (str or None): details of error
    """

    def __init__(self, country, province=None, details=None):
        self.area = self._area(country, province)
        message = f"The dataset of {self.area} has too many missing values"
        super().__init__(message=message, details=details)

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


class NotInteractiveError(_BaseException):
    """Error when interactive shell is not used but forced to use it.

    Args:
        details (str or None): details of error
    """

    def __init__(self, details=None):
        message = "Interactive shell is not used."
        super().__init__(message=message, details=details)


class UnExpectedReturnValueError(_BaseException):
    """Error when unexpected value was returned.

    Args:
        name (str): argument name
        value (object): value user applied or None (will not be shown)
        plural (bool): whether plural or not
        details (str or None): details of error
    """

    def __init__(self, name, value, plural=False, details=None):
        self.value = "" if value is None else f" ({value})"
        self.s = "s" if plural else ""
        self.be = "were" if plural else "was"
        message = f"Un-expected value{self.s}{self.value} {self.be} returned as {name}"
        super().__init__(message=message, details=details)


class DBLockedError(_BaseException):
    """Error when a database has been locked not as expected.

    Args:
        name (str): database name
        details (str or None): details of error
    """

    def __init__(self, name, details=None):
        message = f"{name} should NOT be locked, but locked"
        super().__init__(message=message, details=details)


class NotDBLockedError(_BaseException):
    """Error when a database has NOT been locked not as expected.

    Args:
        name (str): database name
        details (str or None): details of error
    """

    def __init__(self, name, details=None):
        message = f"{name} should be locked, but NOT locked"
        super().__init__(message=message, details=details)
