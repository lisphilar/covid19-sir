#!/usr/bin/env python
# -*- coding: utf-8 -*-

from covsirphy.util.term import Term
import pytest
from covsirphy import ExampleData, Scenario
from covsirphy import SIR, SIRD, SIRF, SEWIRF


class TestExampleData(object):
    def test_iso3(self):
        example_data = ExampleData()
        example_data.add(SIRF, country="Japan")
        assert example_data.country_to_iso3("Japan") == "JPN"
        example_data.add(SIRF, country="Moon")
        assert example_data.country_to_iso3("Moon") == "---"

    @pytest.mark.parametrize("model", [SIR, SIRD, SIRF, SEWIRF])
    def test_one_phase(self, model):
        example_data = ExampleData()
        example_data.add(model)
        with pytest.raises(ValueError):
            example_data.subset()
        # Cleaned dataset
        clean_df = example_data.cleaned()
        cols = [Term.ISO3, Term.DATE, Term.COUNTRY, Term.PROVINCE, Term.C, Term.CI, Term.F, Term.R, Term.N]
        assert set(clean_df.columns) == set(cols)
        # Subset
        subset_df = example_data.subset(model=model)
        assert subset_df.columns.tolist() == Term.SUB_COLUMNS
        example_data.subset(country=model.NAME)
        example_data.subset_complement(model=model)
        example_data.records(model=model)
        # Specialized
        specialized_df = example_data.specialized(model=model)
        assert specialized_df.columns.tolist() == [Term.DATE, *model.VARIABLES]
        # Non-dimensional
        nondim_df = example_data.non_dim(model=model)
        assert nondim_df.columns.tolist() == model.VARIABLES
        assert round(nondim_df.sum().sum()) == len(nondim_df)

    @pytest.mark.parametrize("model", [SIR, SIRD, SIRF, SEWIRF])
    def test_two_phases(self, model):
        example_data = ExampleData()
        example_data.add(model)
        example_data.add(model)
        example_data.subset(model=model)

    @pytest.mark.parametrize("model", [SIR, SIRD, SIRF])
    def test_r0(self, model):
        eg_dict = model.EXAMPLE.copy()
        # Calculate r0
        model_ins = model(eg_dict["population"], **eg_dict["param_dict"])
        eg_r0 = model_ins.calc_r0()
        # Set-up example dataset (from 01Jan2020 to 31Jan2020)
        area = {"country": "Theoretical"}
        example_data = ExampleData(tau=1440, start_date="01Jan2020")
        example_data.add(model, step_n=180, **area)
        # Check the nature of r0
        df = example_data.specialized(model, **area)
        x_max = df.loc[df[Term.CI].idxmax(), Term.S] / eg_dict["population"]
        assert round(x_max, 2) == round(1 / eg_r0, 2)

    def test_scenario(self):
        area = {"country": "Theoretical"}
        # Set-up example dataset (from 01Jan2020 to 31Jan2020)
        example_data = ExampleData(tau=1440, start_date="01Jan2020")
        example_data.add(SIRF, step_n=30, population=SIRF.EXAMPLE["population"], **area)
        # Set-up Scenario instance
        snl = Scenario(tau=1440, **area)
        snl.register(example_data)
        # Check records
        record_df = snl.records(variables="CFR")
        assert set(record_df.columns) == set([Term.DATE, Term.C, Term.F, Term.R])
        # Add a past phase to 31Jan2020 with parameter values
        snl.add(model=SIRF, **SIRF.EXAMPLE["param_dict"])
        # Check summary
        df = snl.summary()
        assert not df.empty
        assert len(df) == 1
        assert Term.RT in df
        # Main scenario
        snl.add(end_date="31Dec2020", name="Main")
        assert snl.get(Term.RT, phase="last", name="Main") == 2.50
        # Lockdown scenario
        snl.clear(name="Lockdown")
        rho_lock = snl.get("rho", phase="0th") * 0.5
        snl.add(end_date="31Dec2020", name="Lockdown", rho=rho_lock)
        assert snl.get(Term.RT, phase="last", name="Lockdown") == 1.25
        # Medicine scenario
        snl.clear(name="Medicine")
        kappa_med = snl.get("kappa", phase="0th") * 0.5
        sigma_med = snl.get("sigma", phase="0th") * 2
        snl.add(end_date="31Dec2020", name="Medicine", kappa=kappa_med, sigma=sigma_med)
        assert snl.get(Term.RT, phase="last", name="Medicine") == 1.31
        # Add vaccine scenario
        snl.clear(name="Vaccine")
        rho_vac = snl.get("rho", phase="0th") * 0.8
        kappa_vac = snl.get("kappa", phase="0th") * 0.6
        sigma_vac = snl.get("sigma", phase="0th") * 1.2
        snl.add(end_date="31Dec2020", name="Vaccine", rho=rho_vac, kappa=kappa_vac, sigma=sigma_vac)
        assert snl.get(Term.RT, phase="last", name="Vaccine") == 1.72
        # Description
        snl.describe()
        # History
        snl.history("Rt")
        snl.history("rho")
        snl.history("Infected")
        snl.history_rate(name="Medicine")
        snl.simulate(name="Vaccine")
