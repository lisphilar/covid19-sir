#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
When you use this file from the top directory of the repository with poetry, please run
cd example; poetry run ./scenario_analysis.py; cd ../
"""

import os
import sys
try:
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
except Exception:
    pass
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
    print("This script works with version >= 2.21.0-gamma")
    print(cs.get_version())
    # Create output directory in example directory
    code_path = Path(__file__)
    input_dir = code_path.parent.with_name("input")
    output_dir = code_path.with_name("output").joinpath(f"{code_path.stem}_{file_prefix}")
    output_dir.mkdir(exist_ok=True, parents=True)
    filer = cs.Filer(output_dir, prefix=file_prefix, numbering="01")
    # Load datasets
    loader = cs.DataLoader(input_dir)
    data_dict = loader.collect()
    # Start scenario analysis and register datasets
    snl = cs.Scenario(country=country, province=province)
    snl.register(**data_dict)
    # Show records
    record_df = snl.records(**filer.png("records"))
    record_df.to_csv(**filer.csv("records", index=False))
    # Load information if available
    backupfile_dict = filer.json("backup")
    if Path(backupfile_dict["filename"]).exists():
        # Restore phase setting (Main scenario)
        print("Restore phase settings of Main scenario with backup JSON file.")
        snl.restore(**backupfile_dict)
        # Adjust file numbers
        filer.png("trend")
        filer.png("estimate_accuracy")
    else:
        # S-R trend analysis
        snl.trend(**filer.png("trend"))
        print(snl.summary())
        # Parameter estimation
        snl.estimate(cs.SIRF)
        snl.estimate_accuracy("10th", name="Main", **filer.png("estimate_accuracy"))
        # Backup
        snl.backup(**backupfile_dict)
    print(snl.summary(columns=["Type", "Start", "End", "Population", "Rt"]))
    # Score of parameter estimation
    metrics = ["MAE", "MSE", "MSLE", "RMSE", "RMSLE"]
    for metric in metrics:
        metric_name = metric.rjust(len(max(metrics, key=len)))
        print(f"{metric_name}: {snl.score(metric=metric)}")
    # Reproduction number in past phases
    snl.history("Rt", **filer.png("history_rt_past"))
    # Main scenario: parameters not changed
    snl.add(name="Main", days=60)
    snl.simulate(name="Main", **filer.png("simulate_main"))
    snl.history_rate(name="Main", **filer.png("history-rate_main"))
    # Forecast scenario: Short-term prediction with regression and OxCGRT data
    fit_dict = snl.fit(delay=(7, 31), name="Forecast", **filer.png("fit_plot"))
    del fit_dict["dataset"], fit_dict["intercept"], fit_dict["coef"]
    print(fit_dict)
    snl.predict(name="Forecast")
    snl.adjust_end()
    snl.simulate(name="Forecast", **filer.png("simulate_forecast"))
    snl.history_rate(name="Main", **filer.png("history-rate_forecast"))
    # Parameter history
    for item in ["Rt", "rho", "sigma", "Confirmed", "Infected", "Recovered", "Fatal"]:
        snl.history(item, **filer.png(f"history_{item}"))
    # Summary
    snl.summary().to_csv(**filer.csv("summary", index=True))


if __name__ == "__main__":
    main()
