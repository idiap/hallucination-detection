# SPDX-FileCopyrightText: 2022 Idiap Research Institute
#
# SPDX-License-Identifier: MIT

""" Predict hallucinations from textual overlap. """

from collections import defaultdict

import json
import os
import string
from sklearn import metrics

from compute_convex_hull import compute_stepwise_convex_hull
from evaluation_utils import flatten, serialize
from textual_overlap import TextualOverlap


def remove_punctuation(text):
    return ''.join(char for char in text if char not in string.punctuation)


def predict(article, sentences, ngram=1):
    # create article sentence segments
    article_sentence_segments = []
    cur_sentence = 0
    for w in article.split():
        article_sentence_segments.append(cur_sentence)
        if w.endswith('.'):
            cur_sentence += 1

    # align article and summary
    ngram_overlap = TextualOverlap(ngram)
    article_tokens = [remove_punctuation(w).lower() for w in article.split()]
    all_candidate_words = []
    all_intrinsic_sentences_scores = []
    all_intrinsic_token_probs = []
    all_extrinsic_hallucinations = []
    for sentence in sentences:
        all_candidate_words.append(sentence.split())
        summary_tokens = [remove_punctuation(w).lower() for w in sentence.split()]
        alignment = ngram_overlap.get_alignment(article_tokens, summary_tokens)

        # compute for each word whether it is aligned to source
        is_aligned = [0] * len(summary_tokens)
        for a in alignment:
            is_aligned[a[0]:a[0] + a[2]] = [1] * a[2]

        # compute number of summary tokens aligned to the same source sentence
        source_sentence_aligned_tokens = defaultdict(int)
        for a in alignment:
            source_sentence = article_sentence_segments[a[1]]
            num_tokens = a[2]
            source_sentence_aligned_tokens[source_sentence] += num_tokens
        num_source_sentences = len(source_sentence_aligned_tokens.keys())

        # compute token hallucination prob as 1 - fraction of tokens aligned to same source sentence
        intrinsic_token_probs = [0] * len(summary_tokens)
        for a in alignment:
            source_sentence = article_sentence_segments[a[1]]
            aligned_to_same_source_sentence = source_sentence_aligned_tokens[source_sentence]
            token_prob = 1 - aligned_to_same_source_sentence / sum(is_aligned)
            intrinsic_token_probs[a[0]:a[0] + a[2]] = [token_prob] * a[2]

        all_intrinsic_token_probs.append(intrinsic_token_probs)
        all_intrinsic_sentences_scores.append([num_source_sentences if a else 0 for a in is_aligned])
        all_extrinsic_hallucinations.append([1 - a for a in is_aligned])

    return {
        'candidate': sentences,
        'candidate_words': all_candidate_words,
        'intrinsic_num_aligned_sentences_scores': all_intrinsic_sentences_scores,
        'intrinsic_num_aligned_tokens_probs': all_intrinsic_token_probs,
        'extrinsic_hallucinations': all_extrinsic_hallucinations,
    }


def main(args):
    with open(args.annotations_path, 'r') as f:
        annotations = [json.loads(l) for l in f.readlines()]
    annotations = {a['id']: a for a in annotations}

    predictions_path = args.predictions_path.replace('{ngram}', str(args.ngram))
    if os.path.exists(predictions_path):
        with open(predictions_path, 'r') as f:
            predictions = [json.loads(l) for l in f.readlines()]
        predictions = {p['id']: p for p in predictions}
    else:
        predictions = {}
        for i, example in annotations.items():
            sentences = [example['sentence']] if 'sentence' in example else example['candidate_sentences']
            prediction = predict(example['article'], sentences, args.ngram)
            prediction['id'] = i
            predictions[i] = prediction
            with open(predictions_path, 'a') as f:
                json.dump(prediction, f)
                f.write('\n')
    assert all(aid in predictions for aid in annotations), "Missing predictions for some annotations"

    # compute auc and thresholds for probs
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

            # probs are assigned to intrinsic/extrinsic based on n-gram's occurrence in article, so they don't overlap
            # take the max to get the union
            preds = [max(l1, l2) for l1, l2 in zip(
                flatten(prediction['intrinsic_num_aligned_tokens_probs']),
                flatten(prediction['extrinsic_hallucinations'])
            )]
            y_scores.extend([p for i, p in enumerate(preds) if i not in indices_to_remove])

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

    with open(args.output_path.replace('{ngram}', str(args.ngram)), 'w') as f:
        json.dump(results, f)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--annotations_path', default='data/frank_annotations.jsonl')
    parser.add_argument('--predictions_path', default='results/frank/lexical_{ngram}_predictions.jsonl')
    parser.add_argument('--output_path', default='results/frank/lexical_{ngram}_results.json')
    parser.add_argument('--ngram', type=int, default=1)
    main(parser.parse_args())
