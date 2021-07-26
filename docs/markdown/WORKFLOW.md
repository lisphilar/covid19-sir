# Workflow

Here is the workflow of analysis with `covsirphy` library.

1. Prepare datasets
    1. Decide whether to use the recommended datasets
    2. (Optional) Read datasets saved in our local environment
    3. (Auto) Download the recommended datasets
    4. (Optional) Perform database lock
    5. (Auto) Clean data
2. [Perform Exploratory data analysis](https://lisphilar.github.io/covid19-sir/usage_dataset.html)
3. [Learn SIR-derived models](https://lisphilar.github.io/covid19-sir/usage_theoretical.html)
4. [Learn S-R trend analysis](https://lisphilar.github.io/covid19-sir/usage_phases.html)
5. [Perform scenario analysis](https://lisphilar.github.io/covid19-sir/usage_quick.html)
    1. Register cleaned data
    2. Check records of the selected country/province
    3. Perform S-R trend analysis to split time series data to phases
    4. Estimate ODE parameter values in the past phases
    5. (Experimental) Predict ODE parameter values in the future phases
    6. Simulate the number of cases with some scenarios
    7. (To Be Implemented) Find solutions to end the outbreak
