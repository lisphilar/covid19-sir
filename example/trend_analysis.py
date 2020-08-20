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
    oxcgrt_data = data_loader.oxcgrt()
    # S-R trend analysis for all countries
    analyser = cs.PolicyMeasures(jhu_data, population_data, oxcgrt_data)
    for country in analyser.countries:
        snl = analyser.scenario(country)
        name = country.replace(" ", "_")
        snl.trend(filename=output_dir.joinpath(f"trend_{name}.png"))


if __name__ == "__main__":
    main()
