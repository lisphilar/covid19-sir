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
    # Start scenario analysis
    ita_scenario = cs.Scenario(jhu_data, population_data, "Italy", tau=120)
    # Show records
    ita_record_df = ita_scenario.records(
        filename=output_dir.joinpath("ita_records.png"))
    ita_record_df.to_csv(output_dir.joinpath("ita_records.csv"), index=False)
    # Show S-R trend
    ita_scenario.trend(filename=output_dir.joinpath("ita_trend.png"))
    print(ita_scenario.summary())
    # Parameter estimation
    ita_scenario.estimate(cs.SIRF)
    # Show the history of optimization
    ita_scenario.estimate_history(
        phase="1st", filename=output_dir.joinpath("ita_estimate_history_1st.png")
    )
    # Show the accuracy as a figure
    df = ita_scenario.summary()
    for phase in df.index:
        ita_scenario.estimate_accuracy(
            phase=phase, filename=output_dir.joinpath(
                f"ita_estimate_accuracy_{phase}.png"
            )
        )
    # Add future phase to main scenario
    ita_scenario.add(name="Main", end_date="01Oct2020")
    ita_scenario.add(name="Main", end_date="31Dec2020")
    ita_scenario.add(name="Main", days=100)
    # Add future phase to alternative scenario
    sigma_4th = ita_scenario.get("sigma", phase="4th")
    sigma_6th = sigma_4th * 2
    ita_scenario.add(
        name="Medicine", end_date="31Dec2020", sigma=sigma_6th)
    ita_scenario.add(name="Medicine", days=100)
    # Simulation of the number of cases
    sim_df = ita_scenario.simulate(
        name="Main",
        filename=output_dir.joinpath("ita_simulate.png")
    )
    sim_df.to_csv(output_dir.joinpath("ita_simulate.csv"), index=False)
    # Save summary as a CSV file
    summary_df = ita_scenario.summary()
    summary_df.to_csv(output_dir.joinpath("ita_summary.csv"), index=True)
    # Tracking of main scenario
    track_df = ita_scenario.track()
    track_df.to_csv(output_dir.joinpath("ita_track.csv"), index=True)
    # Compare scenarios
    ita_scenario.history(
        "Rt", filename=output_dir.joinpath("ita_history_Rt.png"))
    ita_scenario.history(
        "rho", filename=output_dir.joinpath("ita_history_rho.png"))
    ita_scenario.history(
        "sigma", filename=output_dir.joinpath("ita_history_sigma.png"))
    ita_scenario.history(
        "Infected", filename=output_dir.joinpath("ita_history_infected.png"))
    # Change rate of parameters in main scenario (>= 2.8.3-alpha.new.224)
    scenario.history_rate(
        name="Main", filename=output_dir.joinpath("ita_history_rate.jpg"))


if __name__ == "__main__":
    main()
