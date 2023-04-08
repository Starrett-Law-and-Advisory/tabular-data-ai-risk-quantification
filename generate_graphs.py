import argparse
from main import create_model_and_attack, write_to_csv
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np 
import warnings 
import os
from matplotlib.ticker import PercentFormatter
from tqdm import tqdm
from pyfair import FairModel, FairSimpleReport

def create_report(models_data, n_simulations=10_000):
    models = []
    for name, data in models_data.items():
        model = FairModel(f"{name}", n_simulations=n_simulations)
        model.input_data('Threat Event Frequency', mean=np.mean(data[1]), stdev=np.std(data[1]))
        model.input_data('Vulnerability', mean=np.mean(data[0]), stdev=np.std(data[0]))
        # model.input_data('Loss Magnitude', mean=np.mean(data[0]), stdev=np.std(data[0]))
        model.input_data('Loss Magnitude', constant=1)
        model.calculate_all()
        models.append(model)

    fsr = FairSimpleReport(models, currency_prefix='')
    fsr.to_html('report.html')

def plot_graph(config, data, n_steps_list, x_label='n_steps', y_label='success_rate (param p in Bernoulli)',
               output_dir='visualization', display=True):


    os.makedirs(output_dir, exist_ok=True)
    fig_title = f'nts: {config.n_train_samples}, nvs: {config.n_val_samples}, eta: {config.eta}, lambd: {config.lambd}, eta_decay: {config.eta_decay}'
    mean = np.mean(data)
    std = np.std(data)


    plt.title(fig_title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.ylim((0, 1.1))

    plt.text(1, 1, f'\u03BC={mean:.2f}, \u03C3={std:.2f}')
    # plot data line 
    plt.plot(np.arange(config.n_steps_range[0], config.n_steps_range[1]+1), data)

    # plot mean an std line
    plt.axhline(mean, color='r', linestyle='dashed')

    plt.savefig(f'{output_dir}/{fig_title}.png')

    if display:
        plt.show()


def plot_histogram(data, x_label='success_rate (param p in Bernoulli)', y_label='Percentage samples',
               output_dir='visualization', fig_title='histogram', display=True):


    os.makedirs(output_dir, exist_ok=True)

    plt.title(fig_title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)

    weights=np.ones(len(data)) / len(data)

    bins=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    plt.hist(data, bins=bins, weights=weights, edgecolor="black")

    plt.gca().yaxis.set_major_formatter(PercentFormatter(1))

    plt.savefig(f'{output_dir}/{fig_title}.png')

    if display:
        plt.show()

def generate_histogram(config):

    low_n_steps, high_n_steps = config.n_steps_range
    low_eta, high_eta = config.eta_range
    low_lmbda, high_lmbda = config.lmbda_range
    low_eta_decay, high_eta_decay = config.eta_decay_range

    success_rates = []

    if config.dataset:
        print(f'Loading custom dataset: {config.dataset}')
    else:
        print(f'Loading default dataset: Cancer')


    for i in tqdm(range(config.n_samples)):

        eta = np.random.uniform(low_eta, high_eta, 1)[0]
        lmbda = np.random.uniform(low_lmbda, high_lmbda, 1)[0]
        eta_decay = np.random.uniform(low_eta_decay, high_eta_decay, 1)[0]
        n_steps = np.random.randint(low_n_steps, high_n_steps, 1, int)[0]

        success_rate = create_model_and_attack(
            n_steps=int(n_steps),
            eta=eta,
            lambd=lmbda,
            eta_decay=eta_decay,
            n_train_samples=config.n_train_samples,
            n_val_samples=config.n_val_samples,
            dataset=config.dataset,
            column_names=config.column_names,
            target_column=config.target_column,
            success_on_class=config.success_on_class
        )

        write_to_csv(config.csv_file, config, success_rate)
        # print("Success rate: {:.2f}%".format(100*success_rate))
        success_rates.append(success_rate)

    plot_histogram(success_rates, display=config.display, fig_title=f'fig-class-{config.success_on_class}')


def generate_pyfair_report(config):

    low_eta, high_eta = config.eta_range
    low_lmbda, high_lmbda = config.lmbda_range
    low_eta_decay, high_eta_decay = config.eta_decay_range

    n_models = config.num_models
    model_data = {}

    for _ in range(n_models):
        eta = np.round(np.random.uniform(low_eta, high_eta, 1)[0], 3)
        lmbda = np.round(np.random.uniform(low_lmbda, high_lmbda, 1)[0], 3)
        eta_decay = np.round(np.random.uniform(low_eta_decay, high_eta_decay, 1)[0], 3)

        config.eta = eta
        config.lmbda = lmbda
        config.eta_decay = eta_decay
        success_scores, n_steps_list = generate_linechart(config, plot=False)
        model_name = f"eta: {eta} lmbda: {lmbda}, eta_decay: {eta_decay}"
        model_data[model_name] = (success_scores, n_steps_list)

    create_report(model_data)

def generate_linechart(config, plot=True):

    success_rates = []
    if config.dataset:
        print(f'Loading custom dataset: {config.dataset}')
    else:
        print(f'Loading default dataset: Cancer')

    if not plot:
        n_steps_list = np.random.randint(config.n_steps_range[0], config.n_steps_range[1]+1, 100, np.uint8).tolist()
    else:
        n_steps_list = range(config.n_steps_range[0], config.n_steps_range[1]+1) 

    for n_step in tqdm(n_steps_list):
        success_rate = create_model_and_attack(
            n_steps=n_step,
            eta=config.eta,
            lambd=config.lambd,
            eta_decay=config.eta_decay,
            n_train_samples=config.n_train_samples,
            n_val_samples=config.n_val_samples,
            dataset=config.dataset,
            column_names=config.column_names,
            target_column=config.target_column,
            success_on_class=config.success_on_class
        )

        write_to_csv(config.csv_file, config, success_rate)
        # print("Success rate: {:.2f}%".format(100*success_rate))
        success_rates.append(success_rate)

    if plot:
        plot_graph(config, success_rates, n_steps_list, display=config.display)

    return success_rates, n_steps_list

def generate_graph(config):

    if config.mode == 'pyfair':
        generate_pyfair_report(config)
    elif config.mode == 'line':
        generate_linechart(config)
    elif config.mode == 'hist':
        generate_histogram(config)


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', help = 'Choose hist, line or pyfair')
    parser.add_argument('-cf', '--csv-file', default='results.csv')
    parser.add_argument('-nts', '--n_train_samples', type=int)
    parser.add_argument('-nvs', '--n_val_samples', type=int)
    parser.add_argument('-nsr', '--n_steps_range', nargs=2, default=[1, 10], type=int)
    parser.add_argument('-er', '--eta_range', nargs=2, default=[0.2, 0.5], type=float)
    parser.add_argument('-e', '--eta', default=0.2, type=float)
    parser.add_argument('-l', '--lambd', default=0.2, type=float)
    parser.add_argument('-lr', '--lmbda_range', nargs=2, default=[0.2, 0.5], type=float)
    parser.add_argument('-ed', '--eta_decay', default=0.9, type=float)
    parser.add_argument('-edr', '--eta_decay_range',  nargs=2, default=[0.5, 0.9], type=float)
    parser.add_argument('-ns', '--n_samples', default=100_000, type=int)
    parser.add_argument('-nd', '--no_display', action='store_false', dest='display', default=True)
    parser.add_argument('-d', '--dataset', type=str)
    parser.add_argument('-cn', '--column_names', nargs='+', default=None)
    parser.add_argument('-tc', '--target_column', type=str)
    parser.add_argument('-soc', '--success_on_class', type=int)
    parser.add_argument('-nm', '--num_models', default=1, type=int)
    args = parser.parse_args()
    generate_graph(args)
