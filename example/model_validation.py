#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pathlib import Path
import warnings
import covsirphy as cs


def main():
    warnings.simplefilter("error")
    # Create output directory in example directory
    code_path = Path(__file__)
    output_dir = code_path.with_name("output").joinpath(code_path.stem)
    output_dir.mkdir(exist_ok=True, parents=True)
    # Setting
    validator = cs.ModelValidator(n_trials=8, seed=1)
    # Execute validation
    validator.run(cs.SIR)
    validator.run(cs.SIRD)
    validator.run(cs.SIRF)
    print(validator.summary())


if __name__ == "__main__":
    main()
