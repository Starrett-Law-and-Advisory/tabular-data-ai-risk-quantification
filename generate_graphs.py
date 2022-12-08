import argparse
from main import test_3, write_to_csv
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np 
import warnings 
import os

warnings.filterwarnings(action='ignore')

def plot_graph(config, data, x_label='n_steps', y_label='success_rate (param p in bernoulli)',
               output_dir='visualization', display=True):

    os.makedirs(output_dir, exist_ok=True)
    fig_title = f'nts: {config.n_train_samples}, nvs: {config.n_val_samples}, eta: {config.eta}, lambd: {config.lambd}, eta_decay: {config.eta_decay}'
    mean = np.mean(data)    
    std = np.std(data)
    

    plt.title(fig_title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.ylim((0, 1.2))

    plt.text(1, 1, f'\u03BC={mean:.2f}, \u03C3={std:.2f}')
    # plot data line 
    plt.plot(np.arange(1, config.n_steps_range+1), data)

    # plot mean an std line
    plt.axhline(mean, color='r', linestyle='dashed')    
    plt.axhline(std, color='b', linestyle='dashed')    
    plt.savefig(f'{output_dir}/{fig_title}.png')
    
    if display:
        plt.show()
    

def generate_graph(config):

    success_rates = []
    
    for ns in range(1, config.n_steps_range+1):
        success_rate = test_3(
            n_steps=ns,
            eta=config.eta,
            lambd=config.lambd,
            eta_decay=config.eta_decay,
            n_train_samples=config.n_train_samples,
            n_val_samples=config.n_val_samples
        )

        write_to_csv(config.csv_file, config, success_rate)
        print("Success rate: {:.2f}%".format(100*success_rate))
        success_rates.append(success_rate)


    plot_graph(config, success_rates, display=config.display)
    



if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-cf', '--csv-file', default='results.csv')
    parser.add_argument('-nts', '--n_train_samples', type=int)
    parser.add_argument('-nvs', '--n_val_samples', type=int)
    parser.add_argument('-nsr', '--n_steps_range',  default=100, type=int)
    parser.add_argument('-e', '--eta', default=0.2, type=float)
    parser.add_argument('-l', '--lambd', default=0.2, type=float)
    parser.add_argument('-ed', '--eta_decay', default=0.9, type=float)
    parser.add_argument('-nd', '--no_display', action='store_false', dest='display', default=True)
    args = parser.parse_args()
    generate_graph(args)
