# Adversarial Robustness Toolbox

## PyFair Integrations

Basic Terms (Parameters)

1.  [TEF (Threat Event Frequency)](https://pyfair.readthedocs.io/en/latest/#threat-event-frequency-tef):</br> 
    Number of times model is attacked. In our case n_steps param in LowProFool attack.
2.  [V (Vulnerability)](https://pyfair.readthedocs.io/en/latest/#vulnerability-v):</br>
    Probability of successful attacks. In our case success_score

3.  [LEF (Loss Event Frequency)](https://pyfair.readthedocs.io/en/latest/#loss-event-frequency-lef):</br>
    Frequency of loss during given time period. Calculated by multiplying TEF and V.
4.  [LM (Loss Magnitude)](https://pyfair.readthedocs.io/en/latest/#loss-magnitude-lm):</br>
    Total loss in a single Loss Event. In our case success_score (a successful attack).
5.  [Risk](https://pyfair.readthedocs.io/en/latest/#risk-r):</br>
    Represent the ultimate loss for a given time period. Calculated my multiplying LEF and LM.

## PyFair Report Plots

[Note] The Graphs shows the distribution of 3 varients of model (same model with different eta, eta_decay and lambda values). you can use `--num_models` argument in `generate_graphs.py` to generate results for n number of models with different eta, eta_deacy and lambda values (randomly selected). For each model 100 samples are generated with ramdomly selected `n_steps` and those samples are used to generate 10,000 simulations for PyFair Reports.

1.  Risk Distrubution:</br>
    Represent the ultimate loss for a given time period. 
    
    `x-axis` = n_steps in LowProFool Attack </br>
    `y-axis` = frequency of samples that caused ultimate loss.

2.  Exceedence Probability Curve: </br>
    Shows expected loss at each percentile of samples.

    `x-axis` = percentile </br>
    `y-axis` = loss

3.  Loss Exceedence Curve: </br>
    Probability of loss exceedence on each n_steps (what is the probability of increment in loss by increasing n_steps)

    `x-axis` = n_steps in LowProFool Attack</br>
    `y-axis` = probability of increment in loss

4. The tree (in parameters section) shows the which nodes are provided as input and which nodes are calculated to get the value of final node (Risk). 

5. Distribution plots in paramters section are distribution plots of input data. We used `mean` and `std` of 100 samples to generate distribution for 10,000 simulations of pyfair. 

