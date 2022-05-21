#!/usr/bin/env python
# -*- coding: utf-8 -*-

from covsirphy.util.error import deprecate
from covsirphy.util.term import Term


class PhaseUnit(Term):
    """
    Save information of  a phase.

    Args:
        start_date (str): start date of the phase
        end_date (str): end date of the phase
        population (int): population value

    Examples:
        >>> unit1 = PhaseUnit("01Jan2020", "01Feb2020", 1000)
        >>> unit2 = PhaseUnit("02Feb2020", "01Mar2020", 1000)
        >>> unit3 = PhaseUnit("02Mar2020", "01Apr2020", 1000)
        >>> unit4 = PhaseUnit("02Mar2020", "01Apr2020", 1000)
        >>> unit5 = PhaseUnit("01Jan2020", "01Apr2020", 1000)
        >>> str(unit1)
        'Phase (01Jan2020 - 01Feb2020)'
        >>> unit4 == unit4
        True
        >>> unit1 != unit2
        True
        >>> unit1 < unit2
        True
        >>> unit3 > unit1
        True
        >>> unit3 < unit4
        False
        >>> unit3 <= unit4
        True
        >>> unit1 < "02Feb2020"
        True
        >>> unit1 <= "01Feb2020"
        True
        >>> unit1 > "31Dec2019"
        True
        >>> unit1 >= "01Jan2020"
        True
        >>> sorted([unit3, unit1, unit2]) == [unit1, unit2, unit3]
        True
        >>> str(unit1 + unit2)
        'Phase (01Jan2020 - 01Mar2020)'
        >>> str(unit5 - unit1)
        'Phase (02Feb2020 - 01Apr2020)'
        >>> str(unit5 - unit4)
        'Phase (01Jan2020 - 01Mar2020)'
        >>> set([unit1, unit3, unit4]) == set([unit1, unit3])
        True
    """

    @deprecate("PhaseUnit", new="ODEHandler", version="2.19.1-zeta-fu1")
    def __init__(self, start_date, end_date, population):
        raise NotImplementedError

    def __str__(self):
        if self._id_dict is None:
            header = "Phase"
        else:
            id_str = ', '.join(list(self._id_dict.values()))
            header = f"{id_str:>4} phase"
        return f"{header} ({self._start_date} - {self._end_date})"

    def __hash__(self):
        return hash((self._start_date, self._end_date))

    def __eq__(self, other):
        if not isinstance(other, PhaseUnit):
            raise NotImplementedError
        return self._start_date == other.start_date and self._end_date == other.end_date

    def __ne__(self, other):
        return not self.__eq__(other)

    def __lt__(self, other):
        # self < other
        end = self._ensure_date(self._end_date)
        if isinstance(other, str):
            sta_other = self._ensure_date(other)
        elif isinstance(other, PhaseUnit):
            sta_other = self._ensure_date(other.start_date)
        else:
            raise NotImplementedError
        return end < sta_other

    def __le__(self, other):
        # self <= other
        end = self._ensure_date(self._end_date)
        if isinstance(other, str):
            sta_other = self._ensure_date(other)
        elif isinstance(other, PhaseUnit):
            if self.__eq__(other):
                return True
            sta_other = self._ensure_date(other.start_date)
        else:
            raise NotImplementedError
        return end <= sta_other

    def __gt__(self, other):
        # self > other
        if isinstance(other, PhaseUnit) and self.__eq__(other):
            return False
        return not self.__le__(other)

    def __ge__(self, other):
        # self >= other
        return not self.__lt__(other)

    def __add__(self, other):
        if self < other:
            return PhaseUnit(self._start_date, other.end_date, self._population)
        raise NotImplementedError

    def __iadd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        sta = self._ensure_date(self._start_date)
        end = self._ensure_date(self._end_date)
        sta_other = self._ensure_date(other.start_date)
        end_other = self._ensure_date(other.end_date)
        if sta < sta_other and end == end_other:
            end_date = self.yesterday(other.start_date)
            return PhaseUnit(self._start_date, end_date, self._population)
        if sta == sta_other and end > end_other:
            start_date = self.tomorrow(other.end_date)
            return PhaseUnit(start_date, self._end_date, self._population)

    def __isub__(self, other):
        return self.__sub__(other)

    def __contains__(self, date):
        sta = self._ensure_date(self._start_date)
        end = self._ensure_date(self._end_date)
        date = self._ensure_date(date)
        return sta <= date <= end
