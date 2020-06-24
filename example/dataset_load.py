#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pathlib import Path
import covsirphy as cs


def main():
    # Create output directory in example directory
    code_path = Path(__file__)
    input_dir = code_path.parent.with_name("input")
    output_dir = code_path.with_name("output").joinpath(code_path.stem)
    output_dir.mkdir(exist_ok=True, parents=True)
    # Create data loader instance
    data_loader = cs.DataLoader(input_dir)
    # Load JHU dataset
    jhu_data = data_loader.jhu()
    print(jhu_data.citation)
    ncov_df = jhu_data.cleaned()
    ncov_df.to_csv(output_dir.joinpath("covid19_cleaned_jhu.csv"), index=False)
    # Load Japan dataset (the number of cases)
    japan_data = data_loader.japan()
    print(japan_data.citation)
    japan_df = japan_data.cleaned()
    japan_df.to_csv(output_dir.joinpath(
        "covid19_cleaned_japan.csv"), index=False)
    # Replace records of Japan with Japan-specific dataset
    jhu_data.replace(japan_data)
    ncov_df = jhu_data.cleaned()
    ncov_df.to_csv(
        output_dir.joinpath("jhu_cleaned_replaced.csv"), index=False
    )
    # Load OxCGRT dataset
    oxcgrt_data = data_loader.oxcgrt()
    print(oxcgrt_data.citation)
    oxcgrt_df = oxcgrt_data.cleaned()
    oxcgrt_df.to_csv(
        output_dir.joinpath("oxcgrt_cleaned.csv"), index=False
    )
    # Create a subset for a country with ISO3 country code
    oxcgrt_data.subset(iso3="JPN").to_csv(
        output_dir.joinpath("oxcgrt_cleaned_jpn.csv"), index=False
    )


if __name__ == "__main__":
    main()
