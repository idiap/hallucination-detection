# SPDX-FileCopyrightText: 2022 Idiap Research Institute
#
# SPDX-License-Identifier: MIT

import json
from sklearn import metrics

from compute_convex_hull import compute_stepwise_convex_hull
from evaluation_utils import flatten, serialize


def main(args):
    with open(args.annotations_path, 'r') as f:
        annotations = [json.loads(l) for l in f.readlines()]
    annotations = {a['id']: a for a in annotations}
    with open(args.predictions_path, 'r') as f:
        predictions = [json.loads(l) for l in f.readlines()]
    predictions = {p['id']: p for p in predictions}
    assert all(aid in predictions for aid in annotations), "Missing predictions for some annotations"

    # compute auc
    results = {}
    for hallucinations in ['intrinsic_hallucinations', 'extrinsic_hallucinations', 'all_hallucinations']:
        y_true = []
        y_scores = []
        for pid, prediction in predictions.items():
            indices_to_remove = set()
            if hallucinations == 'all_hallucinations':
                y_true.extend([max(l1, l2) for l1, l2 in zip(
                    flatten(annotations[pid]['intrinsic_hallucinations']),
                    flatten(annotations[pid]['extrinsic_hallucinations'])
                )])
            else:
                other_hal = f"{'intrinsic' if hallucinations.startswith('extrinsic') else 'extrinsic'}_hallucinations"
                indices_to_remove = set(i for i, label in enumerate(flatten(annotations[pid][other_hal])) if label)
                trues = flatten(annotations[pid][hallucinations])
                y_true.extend([t for i, t in enumerate(trues) if i not in indices_to_remove])

            y_scores.extend([p for i, p in enumerate(prediction['hallucination_probs']) if i not in indices_to_remove])

        fpr, tpr, roc_thresholds = metrics.roc_curve(y_true, y_scores)
        fpr, tpr = compute_stepwise_convex_hull(fpr, tpr, mode='roc')
        roc_auc = metrics.auc(fpr, tpr)
        precision, recall, pr_thresholds = metrics.precision_recall_curve(y_true, y_scores)
        recall, precision = compute_stepwise_convex_hull(recall, precision, mode='pr')
        pr_auc = metrics.auc(recall, precision)
        best_f1_score = max(2 * p * r / (p + r) if p + r > 0 else 0 for p, r in zip(precision, recall))

        results[hallucinations] = {
            'fpr': serialize(fpr),
            'tpr': serialize(tpr),
            'roc_auc': roc_auc,
            'precision': serialize(precision),
            'recall': serialize(recall),
            'pr_auc': pr_auc,
            'f1_score': best_f1_score,
        }

    with open(args.output_path, 'w') as f:
        json.dump(results, f)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--annotations_path', default='data/frank_annotations.jsonl')
    parser.add_argument('--predictions_path', default='results/frank/fairseq_use_ref_0_predictions.jsonl')
    parser.add_argument('--output_path', default='results/frank/fairseq_use_ref_0_results.json')
    main(parser.parse_args())
