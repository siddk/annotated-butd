"""
visualize.py

Plot Metrics for Various Runs (e.g. to compare models against each other on different datasets).
"""
from tap import Tap

import json
import matplotlib.pyplot as plt
import os

METRICS = ['train_epoch_acc', 'val_acc']
DISPATCH = {'train_epoch_acc': 'Train Epoch Accuracy', 'val_acc': 'Validation Epoch Accuracy'}


class ArgumentParser(Tap):
    # Plot Dataset
    dataset: str = 'GQA'                                # Dataset for Plotting Metrics

    # Metrics Path
    metrics: str = 'checkpoints/metrics'                # Path to Metrics


def visualize():
    # Parse Arguments
    args = ArgumentParser().parse_args()

    # Load Metrics for each Model from Given Dataset
    metrics = {m: {} for m in METRICS}
    for mfile in os.listdir(args.metrics):
        if args.dataset in mfile:
            model = mfile.split('-')[1]
            with open(os.path.join(args.metrics, mfile), 'r') as f:
                data = json.load(f)

            # Iterate through Metrics
            for m in METRICS:
                metrics[m][model] = data[m]

    # Setup
    if not os.path.exists('plots'):
        os.makedirs('plots')

    # Plot
    for m in metrics:
        plt.figure(figsize=(10, 10))
        for model in metrics[m]:
            plt.plot(range(len(metrics[m][model])), metrics[m][model], label=model)
        plt.title('%s %s' % (args.dataset, DISPATCH[m]))
        plt.xlabel('Training Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.savefig('plots/%s-%s.png' % (args.dataset, m))
        plt.clf()


if __name__ == "__main__":
    visualize()
