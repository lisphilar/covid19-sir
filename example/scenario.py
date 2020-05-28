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
    # Start sceario analysis
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
    # Hyoerparameter estimaition
    ita_scenario.estimate(cs.SIRF)
    # Show the history of optimization
    ita_scenario.estimate_history(
        phase="1st", filename=output_dir.joinpath("ita_estimate_history_1st.png")
    )
    # Show the accuracy as a figure
    ita_scenario.estimate_accuracy(
        phase="1st", filename=output_dir.joinpath("ita_estimate_accuracy_1st.png")
    )
    # Add future phase
    ita_scenario.clear()
    ita_scenario.add_phase(end_date="01Aug2020")
    ita_scenario.add_phase(end_date="31Dec2020")
    # Prediction of the number of cases
    pred_df = ita_scenario.predict(
        filename=output_dir.joinpath("ita_predicted.png")
    )
    pred_df.to_csv(output_dir.joinpath("ita_predicted.csv"), index=False)
    # Save summary as a CSV file
    summary_df = ita_scenario.summary()
    summary_df.to_csv(output_dir.joinpath("ita_summary.csv"), index=False)


if __name__ == "__main__":
    main()
