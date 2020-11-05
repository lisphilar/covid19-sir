#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
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
    oxcgrt_data = data_loader.oxcgrt()
    # Setup analyser
    analyser = cs.PolicyMeasures(jhu_data, population_data, oxcgrt_data)
    # Dictionary of the number of phases in each country
    with output_dir.joinpath("trend.json").open("w") as fh:
        analyser.trend()
        json.dump(analyser.phase_len(), fh, indent=4)
    # Show figure of S-R trend analysis
    for country in analyser.countries:
        snl = analyser.scenario(country)
        name = country.replace(" ", "_")
        snl.trend(filename=output_dir.joinpath(f"trend_{name}.png"))


if __name__ == "__main__":
    main()
