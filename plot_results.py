# SPDX-FileCopyrightText: 2022 Idiap Research Institute
#
# SPDX-License-Identifier: MIT

""" Plots the ROC and precision-recall curves of intrinsic/extrinsic/all hallucinations. """

import json
import matplotlib
import matplotlib.pyplot as plt
import os
from sklearn import metrics

from compute_convex_hull import compute_stepwise_convex_hull

# Colors from: https://flatuicolors.com/palette/us
# Light colors
LIGHT_RED = '#ff7675'
LIGHT_PURPLE = '#a29bfe'
LIGHT_BLUE = '#74b9ff'
LIGHT_GREEN = '#55efc4'
LIGHT_YELLOW = '#ffeaa7'

# Dark colors
DARK_RED = '#d63031'
DARK_PURPLE = '#6c5ce7'
DARK_BLUE = '#0984e3'
DARK_GREEN = '#00b894'
DARK_YELLOW = '#fdcb6e'
DARK_GREY = '#636e72'
DARK_ORANGE = '#e17055'
COLORS = [DARK_RED, DARK_BLUE, DARK_GREEN, DARK_PURPLE, DARK_GREY, DARK_ORANGE, DARK_YELLOW]

matplotlib.use('Agg')
plt.rcParams.update({'font.size': 13})


def plot_curve(plot_dir, filename, points, models, linewidth=2, dataset='frank', hal_type='all', mode='pr'):
    assert dataset in ['frank', 'tlhd-cnndm'], f"Unknown dataset: {dataset}"
    assert hal_type in ['intrinsic', 'extrinsic', 'all'], f"Unknown hallucination type: {hal_type}"
    assert mode in ['pr', 'roc'], f"Unknown mode: {mode}"
    plt.figure()
    handles = []
    for i, (cur_points, model) in list(enumerate(zip(points, models)))[::-1]:
        x, y = zip(*cur_points)
        x, y = compute_stepwise_convex_hull(x, y, mode=mode)
        auc = f'{metrics.auc(x, y):.3f}'.lstrip('0')
        handle, = plt.plot(x, y, color=COLORS[i], lw=linewidth, label=f"{model} (AUC = {auc})")
        handles.append(handle)
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    if hal_type == 'intrinsic' and mode == 'pr':
        if dataset == 'frank':
            plt.ylim([-0.01, 0.1])
        else:
            plt.ylim([-0.01, 0.55])
    plt.title(f"{'FRANK' if dataset == 'frank' else 'TLHD-CNNDM'} ({hal_type} hallucinations)")
    plt.xlabel("False Positive Rate" if mode == 'roc' else "Recall")
    plt.ylabel("True Positive Rate" if mode == 'roc' else "Precision")
    plt.legend(handles=handles[::-1], loc="lower right" if mode == 'roc' else None)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, filename))
    plt.close()


def main(args):
    if not os.path.exists(args.plot_dir):
        os.makedirs(args.plot_dir)

    models = ['BART-GBP', 'FEQA', 'DAE', 'Fairseq', 'Lexical-1', 'Lexical-2', 'Lexical-3']
    models_in_path = ['bart_gbp', 'feqa', 'dae', 'fairseq_use_ref_0', 'lexical_1', 'lexical_2', 'lexical_3']
    for dataset in ['frank', 'tlhd-cnndm']:
        data = {}
        for model, model_in_path in zip(models, models_in_path):
            with open(os.path.join(args.results_dir, dataset, f'{model_in_path}_results.json'), 'r') as f:
                data[model] = json.load(f)
        for hal in ['intrinsic_hallucinations', 'extrinsic_hallucinations', 'all_hallucinations']:
            # ROC curve
            points = [zip(data[model][hal]['fpr'], data[model][hal]['tpr']) for model in models]
            plot_curve(args.plot_dir, f'{dataset}_roc_{hal}.pdf', points, models, mode='roc', dataset=dataset,
                       hal_type=hal.split('_')[0])

            # precision-recall curve
            points = [zip(data[model][hal]['recall'], data[model][hal]['precision']) for model in models]
            plot_curve(args.plot_dir, f'{dataset}_pr_{hal}.pdf', points, models, mode='pr', dataset=dataset,
                       hal_type=hal.split('_')[0])


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--plot_dir', default='plots')
    parser.add_argument('--results_dir', default='results')
    main(parser.parse_args())
