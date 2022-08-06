#!/usr/bin/env python
# -*- coding: utf-8 -*-

from covsirphy.util.validator import Validator
from covsirphy.util.term import Term


class Alias(Term):
    """Remember and parse aliases, just like defaultdict.

    Args:
        target_class (object): class of targets or None (all objects)
    """

    def __init__(self, target_class=None):
        self._dict = {}
        self._target_class = target_class or object

    @classmethod
    def for_variables(cls):
        """Initialize covsirphy.Alias with preset of variable aliases.
        """
        class_obj = cls(target_class=list)
        _dict = {
            "N": [cls.N], "S": [cls.S], "T": [cls.TESTS], "C": [cls.C], "I": [cls.CI], "F": [cls.F], "R": [cls.R],
            "CFR": [cls.C, cls.F, cls.R],
            "CIRF": [cls.C, cls.CI, cls.R, cls.F],
            "SIRF": [cls.S, cls.CI, cls.R, cls.F],
            "CR": [cls.C, cls.R],
        }
        [class_obj.update(name, target) for name, target in _dict.items()]
        return class_obj

    def update(self, name, target):
        """Update target of the alias.

        Args:
            name (str): alias name
            targets (object): target to link with the name

        Return:
            covsirphy.Alias: self
        """
        Validator(name, "name", accept_none=False).instance(str)
        self._dict[name] = Validator(target, "target").instance(expected=self._target_class)
        return self

    def find(self, name, default=None):
        """Find the target of the alias.

        Args:
            name (str): alias name
            default (object): default value when not found

        Returns:
            object: target or default value
        """
        try:
            return self._dict.get(name, default)
        except TypeError:
            return default

    def all(self):
        """List up all targets of aliases.

        Returns:
            dict of {str: object}
        """
        return self._dict

    def delete(self, name):
        """Delete alias.

        Args:
            name (str): alias name

        Raises:
            KeyError: the alias has not been registered as an alias

        Return:
            covsirphy.Alias: self
        """
        try:
            del self._dict[name]
        except KeyError:
            raise KeyError(f"{name} has not been registered as an alias.") from None
        return self
