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
    # Load JHU-style dataset
    jhu_data = data_loader.jhu()
    # Load Population dataset
    population_data = data_loader.population()
    # Government Response Tracker (OxCGRT)
    oxcgrt_data = data_loader.oxcgrt(verbose=True)
    # Create analyser with tau value 360 [min] (can be changed)
    analyser = cs.PolicyMeasures(
        jhu_data, population_data, oxcgrt_data, tau=360)
    # S-R trend analysis
    analyser.trend()
    analyser.trend(min_len=2)
    # Parameter estimation
    analyser.estimate(cs.SIRF)
    # All results
    track_df = analyser.track()
    track_df.to_csv(output_dir.joinpath("track.csv"), index=False)


if __name__ == "__main__":
    main()
