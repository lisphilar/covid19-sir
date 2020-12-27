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
    # Create data loader instance
    data_loader = cs.DataLoader(input_dir)
    # Load JHU dataset
    print("<The number of cases>")
    jhu_data = data_loader.jhu()
    print(jhu_data.citation)
    ncov_df = jhu_data.cleaned()
    ncov_df.to_csv(output_dir.joinpath("covid19_cleaned_jhu.csv"), index=False)
    # Subset for Japan
    japan_df, _ = jhu_data.records("Japan")
    japan_df.to_csv(
        output_dir.joinpath("jhu_cleaned_japan.csv"), index=False)
    # Load Population dataset
    print("<Population values>")
    population_data = data_loader.population()
    print(population_data.citation)
    population_df = population_data.cleaned()
    population_df.to_csv(
        output_dir.joinpath("population_cleaned.csv"), index=False
    )
    # Load OxCGRT dataset
    print("<Government response tracker>")
    oxcgrt_data = data_loader.oxcgrt()
    print(oxcgrt_data.citation)
    oxcgrt_df = oxcgrt_data.cleaned()
    oxcgrt_df.to_csv(
        output_dir.joinpath("oxcgrt_cleaned.csv"), index=False
    )
    # Load PCR test dataset
    print("<The number of PCR tests>")
    pcr_data = data_loader.pcr()
    print(pcr_data.citation)
    pcr_df = pcr_data.cleaned()
    pcr_df.to_csv(
        output_dir.joinpath("pcr_cleaned.csv"), index=False)
    pcr_data.positive_rate(
        country="Greece",
        filename=output_dir.joinpath("pcr_positive_rate_Greece.jpg"))
    # Load vaccine dataset
    print("<The number of vaccinations>")
    vaccine_data = data_loader.vaccine()
    print(vaccine_data.citation)
    vaccine_df = vaccine_data.cleaned()
    vaccine_df.to_csv(
        output_dir.joinpath("vaccine_cleaned.csv"), index=False)
    subset_df = vaccine_data.subset(country="Canada")
    subset_df.to_csv(
        output_dir.joinpath("vaccine_subset_canada.csv"), index=False)
    # Load population pyramid dataset
    print("<Population pyramid>")
    pyramid_data = data_loader.pyramid()
    print(pyramid_data.citation)
    subset_df = pyramid_data.subset(country="Japan")
    subset_df.to_csv(
        output_dir.joinpath("pyramid_subset_japan.csv"), index=False)


if __name__ == "__main__":
    main()
