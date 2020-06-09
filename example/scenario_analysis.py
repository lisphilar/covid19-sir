#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pathlib import Path
import covsirphy as cs


def main():
    # Create output directory in example directory
    code_path = Path(__file__)
    output_dir = code_path.with_name("output").joinpath(code_path.stem)
    output_dir.mkdir(exist_ok=True, parents=True)
    # Read JHU dataset
    jhu_file = "input/covid_19_data.csv"
    jhu_data = cs.JHUData(jhu_file)
    # Read population dataset
    pop_file = "input/locations_population.csv"
    pop_data = cs.Population(pop_file)
    # Start scenario analysis
    ita_scenario = cs.Scenario(jhu_data, pop_data, "Italy")
    # Show records
    ita_record_df = ita_scenario.records(
        filename=output_dir.joinpath("ita_records.png"))
    ita_record_df.to_csv(output_dir.joinpath("ita_records.csv"), index=False)
    # Show S-R trend
    ita_scenario.trend(filename=output_dir.joinpath("ita_trend.png"))
    # Find change points
    ita_scenario.trend(
        n_points=4,
        filename=output_dir.joinpath("ita_change_points.png")
    )
    print(ita_scenario.summary())
    # Hyperparameter estimation
    ita_scenario.estimate(cs.SIRF)
    # Show the history of optimization
    ita_scenario.estimate_history(
        phase="1st", filename=output_dir.joinpath("ita_estimate_history_1st.png")
    )
    # Show the accuracy as a figure
    ita_scenario.estimate_accuracy(
        phase="1st", filename=output_dir.joinpath("ita_estimate_accuracy_1st.png")
    )
    # Add future phase to main scenario
    ita_scenario.add_phase(name="Main", end_date="01Aug2020")
    ita_scenario.add_phase(name="Main", end_date="31Dec2020")
    ita_scenario.add_phase(name="Main", days=100)
    # Add future phase to alternative scenario
    sigma_4th = ita_scenario.get("sigma", phase="4th")
    sigma_6th = sigma_4th * 2
    ita_scenario.add_phase(name="New medicines", end_date="31Dec2020", sigma=sigma_6th)
    ita_scenario.add_phase(name="New medicines", days=100)
    # Prediction of the number of cases
    sim_df = ita_scenario.simulate(
        name="Main",
        filename=output_dir.joinpath("ita_simulate.png")
    )
    sim_df.to_csv(output_dir.joinpath("ita_simulate.csv"), index=False)
    # Save summary as a CSV file
    summary_df = ita_scenario.summary()
    summary_df.to_csv(output_dir.joinpath("ita_summary.csv"), index=True)
    # Parameter history
    ita_scenario.param_history(
        targets=["Rt"], name="Main", divide_by_first=False, bix_plot=False,
        filename=output_dir.joinpath("ita_param_history_rt.png")
    )
    ita_scenario.param_history(
        targets=["rho", "sigma"], name="New medicines", divide_by_first=True,
        filename=output_dir.joinpath("ita_param_history_rho_sigma.png")
    )
    desc_df = ita_scenario.describe()
    desc_df.to_csv(output_dir.joinpath("ita_description.csv"), index=True)


if __name__ == "__main__":
    main()
