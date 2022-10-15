from __future__ import annotations
from typing_extensions import Any, Self
from covsirphy.util.validator import Validator
from covsirphy.util.term import Term


class Alias(Term):
    """Remember and parse aliases, just like defaultdict.

    Args:
        target_class: class of targets or None (all objects)
    """

    def __init__(self, target_class: Any = None) -> None:
        self._dict = {}
        self._target_class = target_class or object

    @classmethod
    def for_variables(cls) -> Self:
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

    def update(self, name: str, target: Any) -> Self:
        """Update target of the alias.

        Args:
            name: alias name
            targets: target to link with the name

        Return:
            updated Alias instance
        """
        Validator(name, "name", accept_none=False).instance(str)
        self._dict[name] = Validator(target, "target").instance(expected=self._target_class)
        return self

    def find(self, name: str, default: Any = None) -> Any:
        """Find the target of the alias.

        Args:
            name: alias name
            default: default value when not found

        Returns:
            the target or default value
        """
        try:
            return self._dict.get(name, default)
        except TypeError:
            return default

    def all(self) -> dict[str, Any]:
        """List up all targets of aliases.

        Returns:
            all aliases
        """
        return self._dict

    def delete(self, name: str) -> Self:
        """Delete alias.

        Args:
            name: alias name

        Raises:
            KeyError: the alias has not been registered as an alias

        Return:
            updated Alias instance
        """
        try:
            del self._dict[name]
        except KeyError:
            raise KeyError(f"{name} has not been registered as an alias.") from None
        return self
