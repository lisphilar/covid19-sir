# Workflow

The easiest way to know the usage of `covsirphy` core functionalities is to read [Usage: quickest tour](https://lisphilar.github.io/covid19-sir/usage_quickest.html) after [instalation of the latest version](https://lisphilar.github.io/covid19-sir/markdown/INSTALLATION.html).

Then, please take a look at the following. Here is the workflow of analysis. They can be done with a few lines of Python codee and find the details with the linked documents.

1. [Prepare datasets](https://lisphilar.github.io/covid19-sir/markdown/LOADING.html)
    1. Decide whether to use the recommended datasets
    1. (Optional) Read local datasets
    1. (Optional) Perform database lock
    1. Download the recommended datasets
    1. Clean data
1. [Perform exploratory data analysis](https://lisphilar.github.io/covid19-sir/usage_dataset.html)
1. [Learn SIR-derived models](https://lisphilar.github.io/covid19-sir/usage_theoretical.html)
1. [Learn S-R trend analysis](https://lisphilar.github.io/covid19-sir/usage_phases.html)
1. [Perform scenario analysis](https://lisphilar.github.io/covid19-sir/usage_quick.html)
    1. Register cleaned data
    1. Check records of the selected country/province
    1. Perform S-R trend analysis to split time series data to phases
    1. Estimate ODE parameter values in the past phases
    1. (Experimental) Predict ODE parameter values in the future phases
    1. Simulate the number of cases with some scenarios
    1. (To Be Implemented) Find solutions to end the outbreak

If you have any questions and any advice, feel free to contact developers and users via [GitHub Issues](https://github.com/lisphilar/covid19-sir/issues). Always welcome!
