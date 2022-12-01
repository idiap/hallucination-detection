# SPDX-FileCopyrightText: 2022 Idiap Research Institute
#
# SPDX-License-Identifier: MIT

""" Converts token scores to word probabilities. """

import json
from transformers import BartTokenizer

from bart_gbp_scores import WHITESPACE_CHAR, ALIGNMENT_STATISTICS
from evaluation_utils import PUNCTUATION, align_tokens_to_words, find_sentence_boundaries_in_tokens


def convert_scores_to_probs(scores):
    """ Scale scores to [0, 1], take 1 - scores to get probabilities of hallucination. """
    for statistic in ALIGNMENT_STATISTICS:
        data = scores.values()

        # collect all values of `statistic` to get the min/max used for scaling to probs
        values = [v for d in data for v in d[statistic] if v > -1]
        min_v, max_v = min(values), max(values)

        for d in data:
            # higher scores are less likely to be hallucinations
            # get probs of hallucination by scaling to [0, 1] and inverting
            # for non-participating positions (v == -1) set prob to 0
            d[statistic] = [1 - (v - min_v) / (max_v - min_v) if v > -1 else 0 for v in d[statistic]]


def align_token_annotations(tokens, alignment, token_annotations):
    """ Convert token annotations into word format according to `alignment` between `words` and `tokens`. """
    word_annotations = []
    for word_idx, token_indices in alignment.items():
        if (
                len(token_indices) > 1
                and not len(set(token_annotations[i] for i in token_indices if tokens[i] not in PUNCTUATION)) == 1
        ):
            label = max(set(token_annotations[i] for i in token_indices if tokens[i] not in PUNCTUATION))
        else:
            label = token_annotations[token_indices[0]]
        word_annotations.append(label)
    return word_annotations


def convert_tokens(tokenizer, tokens, token_probs, candidate_sentences, sentence_in_question=None):
    """ Converts token annotations to format. """
    assert sentence_in_question in candidate_sentences if sentence_in_question else True, (
        f"Sentence in question not found in candidate sentences.\n"
        f"Candidate sentences: {candidate_sentences}\nSentence:{sentence_in_question}"
    )
    tokens = [t.replace(WHITESPACE_CHAR, '') for t in tokens]  # remove BART whitespace character
    offset = 0
    candidate_words = []
    word_probs = {name: [] for name in token_probs}
    for candidate_sentence in candidate_sentences:
        # align tokens to candidate sentence
        if tokens[offset] == '':
            offset += 1  # hack to get around one weird case of a double space between to sentences
        start_idx, end_idx = find_sentence_boundaries_in_tokens(tokenizer, candidate_sentence, tokens[offset:])
        start, end = start_idx + offset, end_idx + offset
        offset += end_idx
        if sentence_in_question and candidate_sentence != sentence_in_question:
            continue
        words = candidate_sentence.split()
        candidate_words.append(words)
        alignment = align_tokens_to_words(tokenizer, tokens[start:end], words)
        for name, probs in token_probs.items():
            word_probs[name].append(align_token_annotations(tokens[start:end], alignment, probs[start:end]))
    return {name: probs for name, probs in word_probs.items()}


def main(args):
    with open(args.annotations_path, 'r') as f:
        annotations = [json.loads(l) for l in f.readlines()]
    annotations = {a['id']: a for a in annotations}
    with open(args.token_scores_path, 'r') as f:
        token_scores = [json.loads(l) for l in f.readlines()]
    token_scores = {t['id']: t for t in token_scores}
    assert all(aid in token_scores for aid in annotations), "Missing predictions for some annotations"

    # convert token scores to probabilities
    convert_scores_to_probs(token_scores)
    token_probs = token_scores  # rename

    # convert token-level probs to desired format
    test_key = list(annotations.keys())[0]
    assert 'summary_tokens' in token_probs[test_key].keys()
    tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')
    word_probs = {tid: convert_tokens(
        tokenizer,
        t['summary_tokens'],
        {name: t[name] for name in ALIGNMENT_STATISTICS},
        annotations[tid]['candidate_sentences'],
        annotations[tid]['sentence'] if 'sentence' in annotations[tid] else None,
    ) for tid, t in token_probs.items()}

    with open(args.output_path, 'w') as f:
        for wid, probs in word_probs.items():
            output = {
                'id': wid,
                'article': annotations[wid]['article'],
                'candidate_sentences': annotations[wid]['candidate_sentences'],
            }
            if 'candidate_words' in annotations[wid]:
                output['candidate_words'] = annotations[wid]['candidate_words']
            elif 'sentence_words' in annotations[wid]:
                output['sentence_words'] = annotations[wid]['sentence_words']
            output.update(probs)
            json.dump(output, f)
            f.write('\n')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--annotations_path', default='data/frank_annotations.jsonl')
    parser.add_argument('--token_scores_path', default='results/frank/bart_gbp_scores.jsonl')
    parser.add_argument('--output_path', default='results/frank/bart_gbp_predictions.jsonl')
    main(parser.parse_args())
