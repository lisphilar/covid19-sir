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
    ncov_df = jhu_data.cleaned()
    ncov_df.to_csv(output_dir.joinpath("jhu_cleaned.csv"), index=False)
    # Read Japan dataset
    jpn_file = "input/covid_jpn_total.csv"
    jpn_data = cs.CountryData(jpn_file, country="Japan")
    jpn_data.set_variables(
        date="Date",
        confirmed="Positive",
        fatal="Fatal",
        recovered="Discharged",
        province=None
    )
    jpn_df = jpn_data.cleaned()
    jpn_df.to_csv(output_dir.joinpath("jpn_cleaned.csv"), index=False)
    # Replace data in Japan with Japan-specific dataset
    jhu_data.replace(jpn_data)
    ncov_df = jhu_data.cleaned()
    ncov_df.to_csv(
        output_dir.joinpath("jhu_cleaned_replaced.csv"), index=False
    )
    return ncov_df


if __name__ == "__main__":
    main()
