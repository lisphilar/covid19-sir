#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
When you use this file from the top directory of the repository with poetry, please run
cd example; poetry run ./scenario_analysis.py; cd ../
"""

from pathlib import Path
import covsirphy as cs


def main(country="Italy", province=None, file_prefix="ita"):
    """
    Run scenario analysis template.

    Args:
        country (str): country name
        pronvince (str or None): province name or None (country level)
        file_prefix (str): prefix of the filenames
    """
    # This script works with version >= 2.18.0-beta
    print(cs.get_version())
    # Create output directory in example directory
    code_path = Path(__file__)
    input_dir = code_path.parent.with_name("input")
    output_dir = code_path.with_name("output").joinpath(f"{code_path.stem}_{file_prefix}")
    output_dir.mkdir(exist_ok=True, parents=True)
    filer = cs.Filer(output_dir, prefix=file_prefix, numbering="01")
    # Load datasets
    data_loader = cs.DataLoader(input_dir)
    jhu_data = data_loader.jhu()
    population_data = data_loader.population()
    # Extra datasets
    oxcgrt_data = data_loader.oxcgrt()
    # Start scenario analysis
    snl = cs.Scenario(country=country, province=province)
    snl.register(jhu_data, population_data, extras=[oxcgrt_data])
    # Show records
    record_df = snl.records(**filer.png("records"))
    record_df.to_csv(**filer.csv("records", index=False))
    # Show S-R trend
    snl.trend(**filer.png("trend"))
    print(snl.summary())
    # Parameter estimation
    snl.estimate(cs.SIRF)
    # Score of parameter estimation
    metrics = ["MAE", "MSE", "MSLE", "RMSE", "RMSLE"]
    for metric in metrics:
        metric_name = metric.rjust(len(max(metrics, key=len)))
        print(f"{metric_name}: {snl.score(metric=metric)}")
    # Reproduction number in past phases
    snl.history("Rt", **filer.png("history_rt_past"))
    # Main scenario: parameters not changed
    snl.add(name="Main", end_date="31May2021")
    snl.simulate(name="Main", **filer.png("simulate_main"))
    snl.history_rate(name="Main", **filer.png("history-rate_main"))
    # Forecast scenario: Short-term prediction with regression and OxCGRT data
    fit_dict = snl.fit(name="Forecast")
    fit_dict.pop("coef").to_csv(**filer.csv("forecast_coef", index=True))
    del fit_dict["dataset"], fit_dict["intercept"]
    print(fit_dict)
    snl.predict(name="Forecast")
    snl.add(name="Forecast", end_date="31May2021")
    snl.simulate(name="Forecast", **filer.png("simulate_forecast"))
    snl.history_rate(name="Main", **filer.png("history-rate_forecast"))
    # Parameter history
    for item in ["Rt", "rho", "sigma", "Confirmed", "Infected", "Recovered", "Fatal"]:
        snl.history(item, **filer.png(f"history_{item}"))
    # Summary
    snl.summary().to_csv(**filer.csv("summary", index=True))


if __name__ == "__main__":
    main()
