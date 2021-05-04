# Glossary

This is a list of term in CovsirPhy project. These terms are used as method names and figure titles.

<dl>

<dt><strong>accuracy of models</strong></dt>
<dd>Whether variables and parameters of a model are represent the currenct situation of the outbreak or not.</dd>

<dt><strong>accuracy of parameter estimation</strong></dt>
<dd>How small the difference of actual number of cases and simulated number of cases with the estimated parameter values in a phase. This will be scored with RMSLE.</dd>

<dt><strong>accuracy of trend analysis</strong></dt>
<dd>Whether the change date of phases are effective or not.</dd>

<dt><strong>delay period</strong></dt>
<dd>The number of days that indicators take for the ODE parameter values to be changed.</dd>

<dt><strong>forecasting</strong></dt>
<dd>Forecasting of the number of cases in the future phases. This needs trend analysis, parameter estimation, simulation and prediction.</dd>

<dt><strong>model</strong></dt>
<dd>ODE models derived from the simple SIR model.</dd>

<dt><strong>parameter estimation</strong></dt>
<dd>Optimization of model parameter values with actual records of a phase.</dd>

<dt><strong>phase</strong></dt>
<dd>A sequential dates in which parameter values of SIR-derived models are fixed. </dd>

<dt><strong>prediction</strong></dt>
<dd>Prediction of parameter values in the future phases using relationship of estimated parameters and indexes regarding measures taken by countries and individuals.</dd>

<dt><strong>recovery period</strong></dt>
<dd>The time period between case confirmation and recovery (as it is subjectively defined per country).</dd>

<dt><strong>simulation</strong></dt>
<dd>Calculation of the number of cases with fixed parameter values and initial values.</dd>

<dt><strong>scenario analysis</strong></dt>
<dd>Analyse the number of cases in the future phases with some sets of ODE parameter values. With this analysis, we can estimate the impact of our activities against the outbreak on the number of cases.</dd>

<dt><strong>tau parameter in models</strong></dt>
<dd>Tau is a parameter used to convert actual time (with unit [min]) to time steps (without units).
This conversion enables us to use actual dataset with ODE models. It is a divisor of 1440 [min] (= 1 day).

Tau is generally considered as a parameter and is not set to a predetermined value.
Its value is determined by the estimator only during the last phase and then uses that value for all the previous phases as well.

Because the refresh rate of the data is per day, the unit of tau value should be equal to or under 1 day.
So when tau is estimated to be for example 360 [min] instead of 1440 [min], that would mean the analysis is more effective if we study the records per 6 hours (since 360/1440 = 1/4 of the day) and not as a whole day.
If for some reason tau would be set to more than 1 day, then many records would be ignored and that is the reason tau is bound to max 1 day = 1440 [min].</dd>

<dt><strong>trend</strong></dt>
<dd>Parameter set of a phase.</dd>

<dt><strong>trend analysis</strong></dt>
<dd>Breaking down the series of records to phases.</dd>

</dl>
