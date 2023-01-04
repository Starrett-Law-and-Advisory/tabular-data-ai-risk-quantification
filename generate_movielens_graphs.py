import argparse
from movie_lens import main as test
import os
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import PercentFormatter
from tqdm import tqdm

def plot_graph(config, data, x_label='epochs', y_label='test_accuracy',
               output_dir='visualization', display=True):

    os.makedirs(output_dir, exist_ok=True)
    fig_title = f'delta={config["delta"]}, lr={config["learning_rate"]}, nm={config["noise_multiplier"]}, lnc={config["l2_norm_clip"]}, max_mu={config["max_mu"]}'

    mean = np.mean(data)    
    std = np.std(data)
    
    plt.title(fig_title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.ylim((0, 1.1))

    plt.text(1, 1, f'\u03BC={mean:.2f}, \u03C3={std:.2f}')

    # plot data line 
    plt.plot(np.arange(len(data)), data)

    # plot mean an std line
    plt.axhline(mean, color='r', linestyle='dashed')    

    plt.savefig(f'{output_dir}/{fig_title}.png')
    
    if display:
        plt.show()

def plot_histogram(config, data, x_label='test_accuracy', y_label='Percentage samples',
               output_dir='visualization', fig_title='histogram', display=True):

    fig_title = f'epochs={config.epochs}, delta={config.delta_range}, lr={config.learning_rate_range}, nm={config.noise_multiplier_range}, lnc={config.l2_norm_clip_range}, max_mu={config.max_mu}'

    
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

def generate_line_chart(config):

    rmse, epsilons, mus = test({
        'l2_norm_clip': config.l2_norm_clip,
        'noise_multiplier': config.noise_multiplier,
        'delta': config.delta,
        'learning_rate': config.learning_rate,
        'epochs': config.epochs,
        'max_mu': config.max_mu,
        'model_dir': config.model_dir,
        'dpsgd': config.dpsgd
    })

    plot_graph(rmse, display=config.display, y_label='rmse')
    plot_graph(epsilons, display=config.display, y_label='eps')
    plot_graph(mus, display=config.display, y_label='mu')
    

def generate_histogram(config):
    
    low_l2_norm_clip, high_l2_norm_clip = config.l2_norm_clip_range
    low_noise_multiplier, high_noise_multiplier = config.noise_multiplier_range
    low_delta, high_delta = config.delta_range
    low_learning_rate, high_learning_rate = config.learning_rate_range

    rmse = []
    epsilons = []
    mus = []

    for i in tqdm(range(config.num_samples)):
        l2_norm_clip = np.random.uniform(low_l2_norm_clip, high_l2_norm_clip, 1)[0]
        noise_multiplier = np.random.uniform(low_noise_multiplier, high_noise_multiplier, 1)[0]
        delta = np.random.uniform(low_delta, high_delta, 1)[0]
        learning_rate = np.random.uniform(low_learning_rate, high_learning_rate, 1)[0]

        outs = test({
            'l2_norm_clip': l2_norm_clip,
            'noise_multiplier': noise_multiplier,
            'delta': delta,
            'learning_rate': learning_rate,
            'epochs': config.epochs,
            'max_mu': config.max_mu,
            'model_dir': config.model_dir,
            'dpsgd': config.dpsgd
        })
        rmse.append(outs[0][-1])
        epsilons.append(outs[1][-1])
        mus.append(outs[2][-1])

    plot_histogram(rmse, display=config.display, x_label='rmse')
    plot_histogram(epsilons, display=config.display, x_label='eps')
    plot_histogram(mus, display=config.display, x_label='mu')
 
def main(args):
    # generate_histogram(args)
    generate_line_chart(args)
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--epochs', type=int, default=25, help='number of training epochs')
    parser.add_argument('-mu', '--max_mu', type=int,  default=2, help='max value of mu (program terminates if mu gets bigger than this)')

    parser.add_argument('-lncr', '--l2_norm_clip_range', nargs=2, type=float, default=[5, 7], help='range for max L2 norm value')
    parser.add_argument('-nmr', '--noise_multiplier_range', nargs=2, type=float, default=[0.3, 1], help='range of noise to be added during training')
    parser.add_argument('-lrr', '--learning_rate_range', nargs=2, type=float, default=[0.001, 0.1], help='learning rate for training')
    parser.add_argument('-dr', '--delta_range', nargs=2, type=float, default=[1e-6, 1e-5], help='range for delta value for eplison calculation')

    parser.add_argument('-lnc', '--l2_norm_clip', type=float, default=5, help='max L2 norm value')
    parser.add_argument('-nm', '--noise_multiplier', type=float, default=0.55, help='amount of noise to be added during training')
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.01, help='learning rate for training')
    parser.add_argument('-d', '--delta', type=float, default=1e-6, help='delta value for eplison calculation')


    parser.add_argument('-ns', '--num_samples', type=int, default=100, help='number of samples to be generated.')

    parser.add_argument('--model_dir', type=str, default=None, help='path top save model')
    parser.add_argument('--use_sgd', action='store_false', dest='dpsgd', 
                        default=True, help='if provided valila sdg will be used for optimization otherwise DP-SQG will be used.')

    parser.add_argument('--no_display', action='store_false', dest='display', 
                        default=True, help='if provided graphs will not be displayed.')


    args = parser.parse_args()
    main(args)

