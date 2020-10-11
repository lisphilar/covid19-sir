#!/usr/bin/env python
# -*- coding: utf-8 -*-

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
    # Japan dataset
    japan_data = data_loader.japan()
    jhu_data.replace(japan_data)
    # Start scenario analysis
    scenario = cs.Scenario(jhu_data, population_data, "Japan")
    # Show records
    record_df = scenario.records(
        filename=output_dir.joinpath("jpn_records.jpg"))
    record_df.to_csv(output_dir.joinpath("jpn_records.csv"), index=False)
    # Show S-R trend
    scenario.trend(filename=output_dir.joinpath("jpn_trend.jpg"))
    scenario.enable(phases=["0th"])
    print(scenario.summary())
    # Parameter estimation
    scenario.estimate(cs.SIRF)
    # Add future phase to main scenario
    scenario.add(name="Main", end_date="31Dec2020")
    # Simulation of the number of cases
    sim_df = scenario.simulate(
        name="Main",
        filename=output_dir.joinpath("jpn_simulate.jpg")
    )
    sim_df.to_csv(output_dir.joinpath("jpn_simulate.csv"), index=False)
    # Parameter history
    scenario.history(
        target="Rt", filename=output_dir.joinpath("jpn_rt.jpg"))
    scenario.history(
        target="rho", filename=output_dir.joinpath("jpn_rho.jpg"))
    scenario.history(
        target="sigma", filename=output_dir.joinpath("jpn_sigma.jpg"))
    # Change rate of parameters in main scenario (>= 2.8.3-alpha.new.224)
    scenario.history_rate(
        name="Main", filename=output_dir.joinpath("jpn_history_rate.jpg"))
    scenario.history_rate(
        params=["kappa", "sigma", "rho"], name="Main",
        filename=output_dir.joinpath("jpn_history_rate_without_theta.jpg"))
    # Save summary as a CSV file
    summary_df = scenario.summary()
    summary_df.to_csv(output_dir.joinpath("summary.csv"), index=True)


if __name__ == "__main__":
    main()
