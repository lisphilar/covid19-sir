#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pathlib import Path
from pprint import pprint
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
    min_len = max(analyser.phase_len().keys())
    analyser.trend(min_len=min_len)
    pprint(analyser.phase_len(), compact=True)
    # Parameter estimation
    analyser.estimate(cs.SIRF)
    # Parameter history of Rt
    rt_df = analyser.param_history(
        "Rt", roll_window=None, filename="history_rt.png")
    rt_df.to_csv(output_dir.joinpath("history_rt.csv"), index=False)
    # Parameter history of rho
    rho_df = analyser.param_history(
        "rho", roll_window=14, filename="history_rho.jpg")
    rho_df.to_csv(output_dir.joinpath("history_rho.csv"), index=False)
    # Parameter history of sigma
    sigma_df = analyser.param_history(
        "sigma", roll_window=14, filename="history_sigma.jpg")
    sigma_df.to_csv(output_dir.joinpath("history_sigma.csv"), index=False)
    # Parameter history of kappa
    kappa_df = analyser.param_history(
        "kappa", roll_window=14, filename="history_kappa.jpg")
    kappa_df.to_csv(output_dir.joinpath("history_kappa.csv"), index=False)
    # Parameter history of theta
    theta_df = analyser.param_history(
        "theta", roll_window=14, filename="history_theta.jpg")
    theta_df.to_csv(output_dir.joinpath("history_theta.csv"), index=False)


if __name__ == "__main__":
    main()
