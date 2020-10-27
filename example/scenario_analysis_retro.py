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
    # Japan dataset
    japan_data = data_loader.japan()
    jhu_data.replace(japan_data)
    # Set country name
    country = "Japan"
    abbr = "jpn".lower()
    figpath = functools.partial(
        filepath, output_dir=output_dir, country=abbr, ext="jpg")
    # Start scenario analysis
    snl = cs.Scenario(jhu_data, population_data, country)
    # Retrospective analysis
    snl.retrospective(
        "01Sep2020", model=cs.SIRF, control="Main", target="Retrospective")
    print(snl.summary())
    # Parameter history
    for item in ["Rt", "theta", "rho", "sigma", "kappa", "Infected"]:
        snl.history(item, filename=figpath(f"history_{item.lower()}"))
    # Change rate of parameters in main snl
    snl.history_rate(name="Main", filename=figpath("history_rate"))
    # Save summary as a CSV file
    summary_df = snl.summary()
    summary_df.to_csv(output_dir.joinpath("summary.csv"), index=True)


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


if __name__ == "__main__":
    main()
