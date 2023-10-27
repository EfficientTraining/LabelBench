import argparse
import pickle
import os
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from utils import accum_max, moving_avg, none_smooth
import warnings
warnings.filterwarnings("ignore")

if __name__ == "__main__":
    matplotlib.rcParams.update({'font.size': 16})
    matplotlib.rcParams.update({'figure.autolayout': True})

    parser = argparse.ArgumentParser()
    parser.add_argument("--wandb_name", type=str, help="Wandb user name.")
    parser.add_argument("--dataset", type=str, help="Dataset name.")
    parser.add_argument("--batch_size", type=int,
                        help="We train neural network after collecting batch_size number of new labels.")
    parser.add_argument("--embed_model", type=str, help="Substring of the embedding model name.")
    parser.add_argument("--classifier_model", type=str, help="Substring of the classifier model name.")
    parser.add_argument("--trainer_config", type=str, help="Substring of the trainer name.")
    parser.add_argument("--metrics", nargs="+", help="Metric names for plotting.")
    parser.add_argument("--strategies", nargs="+", help="Strategy names for plotting.")
    parser.add_argument("--colors", nargs="+", help="Color for plots.")
    parser.add_argument("--smoothing", choices=["none", "max", "mov_avg"])
    parser.add_argument("--y_lim", type=float, nargs="+", default=None)
    args = parser.parse_args()

    if args.smoothing == "none":
        smooth_fn = none_smooth
    elif args.smoothing == "max":
        smooth_fn = accum_max
    else:
        smooth_fn = moving_avg

    path_name = "./%s/bs_%d/embed_%s/model_%s/trainer_%s" % (
        args.dataset, args.batch_size, args.embed_model, args.classifier_model, args.trainer_config)
    alg_dict = {}
    for name in os.listdir(path_name):
        with open("%s/%s" % (path_name, name), "rb") as file:
            run_dict, _ = pickle.load(file)
        alg_name = name.split("_")[0]
        if alg_name in alg_dict:
            alg_dict[alg_name].append(run_dict)
        else:
            alg_dict[alg_name] = [run_dict]

    num_labeled = alg_dict[args.strategies[0]][0]["Num Labeled"]

    for metric in args.metrics:
        y_lim = args.y_lim
        fig, ax = plt.subplots(1)
        for strategy, color in zip(args.strategies, args.colors):
            runs = alg_dict[strategy]
            num_runs = float(len(runs))
            perfs = np.array([smooth_fn(np.array(run[metric])) for run in runs])
            perf_mean = np.mean(perfs, axis=0)
            perf_ste = np.std(perfs, ddof=1, axis=0) / np.sqrt(num_runs)
            ax.plot(num_labeled, perf_mean, linewidth=3, color=color)
            ax.fill_between(num_labeled, perf_mean - perf_ste, perf_mean + perf_ste, alpha=.3, color=color)
            if y_lim is None and strategy == "random":
                y_lim = [perfs[0][0], None]

        plt.xlabel("Number of Labels")
        plt.ylabel(metric)
        
        legend_list = []
        for strategy in args.strategies:
            legend_list.append(strategy.upper())
            legend_list.append('_Hidden')
        ax.legend(legend_list)
        plt.ylim(y_lim)
        ax.grid(True, linestyle="--")
        plt.savefig("plots/%s_%d_%s_%s_%s_%s.pdf" % (
            args.dataset, args.batch_size, args.embed_model, args.classifier_model, args.trainer_config,
            metric.replace(" ", "_")))
        plt.show()
