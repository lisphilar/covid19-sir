#!/usr/bin/env python
# -*- coding: utf-8 -*-

import functools
from pathlib import Path
import warnings
import covsirphy as cs


def main():
    warnings.simplefilter("error")
    # Create output directory in example directory
    code_path = Path(__file__)
    input_dir = code_path.parent.with_name("input")
    output_dir = code_path.with_name("output").joinpath(code_path.stem)
    output_dir.mkdir(exist_ok=True, parents=True)
    # Load datasets
    data_loader = cs.DataLoader(input_dir)
    jhu_data = data_loader.jhu()
    population_data = data_loader.population()
    # Set country name
    country = "Italy"
    abbr = "ita"
    figpath = functools.partial(
        filepath, output_dir=output_dir, country=abbr, ext="jpg")
    # Start scenario analysis
    snl = cs.Scenario(jhu_data, population_data, country, tau=120)
    # Show records
    record_df = snl.records(filename=figpath("records"))
    save_df(record_df, "records", output_dir, abbr, use_index=False)
    # Daily new cases
    snl.records_diff(filename=figpath("records_diff"))
    # Show S-R trend
    snl.trend(filename=figpath("trend"))
    print(snl.summary())
    # Parameter estimation
    snl.estimate(cs.SIRF)
    # Show the history of optimization
    snl.estimate_history(phase="1st", filename=figpath("estimate_history_1st"))
    # Show the accuracy as a figure
    df = snl.summary()
    for phase in df.index:
        snl.estimate_accuracy(
            phase=phase, filename=figpath(f"estimate_accuracy_{phase}")
        )
    # Add future phase to main scenario
    snl.add(name="Main", end_date="31Mar2021")
    snl.add(name="Main", days=100)
    # Add future phase to alternative scenario
    sigma_4th = snl.get("sigma", phase="last")
    sigma_6th = sigma_4th * 2
    snl.clear(name="Medicine", template="Main")
    snl.add(name="Medicine", end_date="31Mar2021")
    snl.add(name="Medicine", days=100, sigma=sigma_6th)
    # Simulation of the number of cases
    sim_df = snl.simulate(name="Main", filename=figpath("simulate"))
    save_df(sim_df, "simulate", output_dir, abbr, use_index=False)
    # Summary
    summary_df = snl.summary()
    save_df(summary_df, "summary", output_dir, abbr, use_index=True)
    # Description of scenarios
    desc_df = snl.describe()
    save_df(desc_df, "describe", output_dir, abbr, use_index=True)
    # Tracking for main scenario
    track_df = snl.track()
    save_df(track_df, "track", output_dir, abbr, use_index=True)
    # Compare scenarios
    for item in ["Rt", "rho", "sigma", "Infected"]:
        snl.history(item, filename=figpath(f"history_{item.lower()}"))
    # Change rate of parameters in main scenario
    snl.history_rate(name="Main", filename=figpath("history_rate"))
    # Score of main scenario
    metrics_list = ["MAE", "MSE", "MSLE", "RMSE", "RMSLE"]
    for metrics in metrics_list:
        metrics_name = metrics.rjust(len(max(metrics_list, key=len)))
        print(f"{metrics_name}: {snl.score(metrics=metrics)}")


def filepath(name, output_dir, country, ext="jpg"):
    """
    Return filepath of a figure.

    Args:
        name (str): name of the figure
        output_dir (pathlib.Path): path of the directory to save the figure
        country (str): country name or abbr
        ext (str, optional): Extension of the output file. Defaults to "jpg".

    Returns:
        pathlib.Path: filepath of the output file
    """
    return output_dir.joinpath(f"{country}_{name}.{ext}")


def save_df(df, name, output_dir, country, use_index=True):
    """
    Save the dataframe as a CSV file.

    Args:
        df (pandas.DataFrame): dataframe
        name (str): name of the dataframe
        output_dir (pathlib.Path): path of the directory to save the figure
        country (str): country name or abbr
        use_index (bool): if True, include index
    """
    df.to_csv(output_dir.joinpath(f"{name}.csv"), index=use_index)


if __name__ == "__main__":
    main()
