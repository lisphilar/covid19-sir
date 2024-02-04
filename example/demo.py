#!/usr/bin/env python

"""
When you use this file from the top directory of the repository with poetry, please run
cd example; poetry run ./demo.py; cd ../
"""

from pathlib import Path
import covsirphy as cs


def main(geo="Japan", file_prefix="jpn"):
    """Create figures for demonstration.

    Args:
        geo (str): location identifier
        file_prefix (str): prefix of the filenames
    """
    print("This script works with version >= 3.0.0")
    print(cs.get_version())
    code_path = Path(__file__)
    output_dir = code_path.with_name("output").joinpath(f"{code_path.stem}_{file_prefix}")
    filer = cs.Filer(str(output_dir), numbering="01")
    # Data preparation,time-series segmentation, parameter estimation with SIR-F model
    snr = cs.ODEScenario.auto_build(geo=geo, model=cs.SIRFModel)
    # Check actual records
    snr.simulate(name=None, **filer.png("actual"))
    # Show the result of time-series segmentation
    snr.to_dynamics(name="Baseline").detect(**filer.png("segmentation"))
    # Perform simulation with estimated ODE parameter values
    snr.simulate(name="Baseline", **filer.png("simulate"))
    # Predict ODE parameter values (30 days from the last date of actual records)
    snr.build_with_template(name="Predicted", template="Baseline")
    snr.predict(days=30, name="Predicted")
    # Perform simulation with estimated and predicted ODE parameter values
    snr.simulate(name="Predicted", **filer.png("predicted"))
    # Add a future phase to the baseline (ODE parameters will not be changed)
    snr.append()
    # Show created phases and ODE parameter values
    snr.summary()
    # Compare reproduction number of scenarios (predicted/baseline)
    snr.compare_param("Rt", **filer.png("Rt"))
    # Compare simulated number of cases
    snr.compare_cases("Confirmed", **filer.png("confirmed"))
    # Describe representative values
    snr.describe()


if __name__ == "__main__":
    main()
