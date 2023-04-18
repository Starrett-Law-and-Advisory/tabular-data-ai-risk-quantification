# Adversarial Robustness Toolbox

## Command Line Arguments
These are the arguments we use (see end of `main.py`):
```python
    parser.add_argument('-m', '--mode', default='test', type=str, choices=['test', 'report_line', 'report_hist', 'report_pyfair'])
    parser.add_argument('-cf', '--save_csv_file', default='results.csv')
    parser.add_argument('-nts', '--n_train_samples', type=int)
    parser.add_argument('-nvs', '--n_val_samples', type=int)
    parser.add_argument('-ns', '--n_steps', default=100, type=int)
    parser.add_argument('-e', '--eta', default=0.2, type=float)
    parser.add_argument('-l', '--lambd', default=0.2, type=float)
    parser.add_argument('-ed', '--eta_decay', default=0.9, type=float)
    parser.add_argument('-d', '--dataset', type=str)
    parser.add_argument('-cn', '--column_names', nargs='+', default=None)
    parser.add_argument('-tc', '--target_column', type=str)
    parser.add_argument('-soc', '--success_on_class', type=int)
    # -----------------------------------------------------------------
    # Generating reports
    # -----------------------------------------------------------------
    parser.add_argument('-er', '--eta_range', nargs=2, default=[0.2, 0.5], type=float)
    parser.add_argument('-lr', '--lambd_range', nargs=2, default=[0.2, 0.5], type=float)
    parser.add_argument('-edr', '--eta_decay_range',  nargs=2, default=[0.5, 0.9], type=float)
    parser.add_argument('-nsr', '--n_steps_range', nargs=2, default=[1, 10], type=int)
    parser.add_argument('-nhs', '--n_hist_samples', default=1000, type=int)
    parser.add_argument('-nm', '--num_models', default=1, type=int)
    parser.add_argument('-di', '--display', action='store_true')
```

Explanation of arguments is below:
1. `mode`: Can choose 4 options: `test`, `report_line`, `report_hist`, `report_pyfair`. The first option simply creates a model, and attacks it and prints out the success rate of attack. The other 3 options generate different kinds of reports
2. `save_csv_file`: CSV file to save results in. Only used if `mode=test`
3. `n_train_samples`: Number of training samples to train the model on
4. `n_val_samples`: Number of validation samples to test the model on
5. `n_steps`: Number of times the input sample is perturbed. Note that LowProFool works by repeatedly perturbing the input sample until the model outputs the desired class. `n_steps` is the maximum times we perturb the input sample before giving up
6. `eta`: A parameter of LowProFool attack (see paper for more details)
7. `lambd`: A parameter of LowProFool attack (see paper for more details)
8. `eta_decay`: A parameter of LowProFool attack (see paper for more details)
9. `dataset`: Which dataset to use. This is the path to a CSV file similar to `creditcard.csv`
10. `column_names`: Columns that form the input in the dataset CSV file
11. `target_column`: Column name in the CSV file that contains the target class / groundtruth
12. `success_on_class`: The desired class we want the model to output. The input samples are perturbed until the model outputs this class
13. `eta_range`: Used while generating reports. `eta` is varied in this range to get different values of `success_rate`
14. `lambd_range`: Used while generating reports. `lambd` is varied in this range to get different values of `success_rate`
15. `eta_decay_range`: Used while generating reports. `eta_decay` is varied in this range to get different values of `success_rate`
16. `n_steps_range`: Used while generating reports. `n_steps` is varied in this range to get different values of `success_rate`
17. `n_hist_samples`: Number of different parameters to try out for creating a histogram. Only activated when `mode=report_hist`
18. `num_models`: Number of different models to create [DEPRECATED]
19. `display`: If running on local machine, then pass this to open a popup window to show graphs rather than saving them. Only activated if `mode=report_line` or `mode=report_hist`

Details:
1. `eta`: Determines the magnitude of the changes made to the original input to create the adversarial example, a larger eta value will result in a larger perturbation, which can lead to a higher fooling rate, but may also make the perturbation more noticeable and increase the likelihood of detection and vice versa.
2. `eta_decay`: Used to gradually reduce the eta. This can be useful for improving the convergence of the algorithm and avoiding overshooting. The eta decay  parameter determines how quickly the step size is reduced over time. A higher eta decay value will lead to a slower reduction of the step size, while a lower value will lead to a faster reduction. Typically, eta decay values are set between 0.9 and 1.0, with larger values corresponding to slower decay rates.
3. `lambd`: Regularization parameter used to control the trade-off between the size of the perturbation and the prediction error of the model. It determines the importance of minimizing the distance (change) between the original input and the adversarial input, versus maximizing the prediction error of the model on the adversarial input. A larger lambda value will prioritize minimizing the distance, which can lead to a smaller perturbation but potentially lower fooling rate. On the other hand, a smaller lambda value will prioritize maximizing the prediction error, which can lead to a higher fooling rate but potentially larger perturbations.

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

