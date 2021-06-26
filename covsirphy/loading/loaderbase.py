#!/usr/bin/env python
# -*- coding: utf-8 -*-

from covsirphy.util.term import Term


class _LoaderBase(Term):
    """
    Basic class for data loading.
    """

    def __init__(self):
        pass

    def jhu(self):
        """
        Load the dataset regarding the number of cases.

        Returns:
            covsirphy.JHUData: dataset regarding the number of cases
        """
        raise NotImplementedError

    def population(self):
        """
        Load the dataset regarding population values.

        Returns:
            covsirphy.PopulationData: dataset regarding population values
        """
        raise NotImplementedError

    def oxcgrt(self):
        """
        Load the dataset regarding OxCGRT indicators.

        Returns:
            covsirphy.JHUData: dataset regarding OxCGRT data
        """
        raise NotImplementedError

    def japan(self):
        """
        Load the dataset of the number of cases in Japan.

        Returns:
            covsirphy.CountryData: dataset at country level in Japan
        """
        raise NotImplementedError

    def linelist(self):
        """
        Load linelist of case reports.

        Returns:
            covsirphy.CountryData: dataset at country level in Japan
        """
        raise NotImplementedError

    def pcr(self):
        """
        Load the dataset regarding the number of tests and confirmed cases.

        Returns:
            covsirphy.PCRData: dataset regarding the number of tests and confirmed cases
        """
        raise NotImplementedError

    def vaccine(self):
        """
        Load the dataset regarding vaccination.

        Returns:
            covsirphy.VaccineData: dataset regarding vaccines
        """
        raise NotImplementedError

    def pyramid(self):
        """
        Load the dataset regarding population pyramid.

        Returns:
            covsirphy.PopulationPyramidData: dataset regarding population pyramid
        """
        raise NotImplementedError

    def collect(self):
        """
        Collect data for scenario analysis and return them as a dictionary.

        Returns:
            dict(str, object):
                - jhu_data (covsirphy.JHUData)
                - extras (list[covsirphy.CleaningBase]):
                    - covsirphy.OXCGRTData
                    - covsirphy.PCRData
                    - covsirphy.VaccineData
        """
        return {
            "jhu_data": self.jhu(),
            "extras": [self.oxcgrt(), self.pcr(), self.vaccine()]
        }
