# SPDX-FileCopyrightText: 2022 Idiap Research Institute
#
# SPDX-License-Identifier: MIT

""" Computes BART-GBP scores: association strength, fraction of tokens aligned and inverse decoding entropy. """

from collections import Counter, defaultdict

import json
import numpy as np
import os
import pickle
import scipy.special
import scipy.stats
import torch

from alignment_utils import align_by_attention_rank, rindex, standardize_shape

WHITESPACE_CHAR = u'\u0120'  # BART's whitespace character
BOS_TOKEN = '<s>'
SENT_SEP_TOKEN = '.'
SPECIAL_TOKENS = [BOS_TOKEN, SENT_SEP_TOKEN]

# names of alignment statistics
ALIGNMENT_STATISTICS = [
    'intrinsic_association_strength',
    'extrinsic_sentence_aligned',
    'extrinsic_inverse_entropy',
]


def fix_alignments(alignments, fix_ngram):
    """ Fixes the alignments of `fix_ngram`-grams that are consecutive in both the source and summary. """
    is_fixed = [0] * len(alignments)
    if fix_ngram <= 0:
        return is_fixed
    for i in range(len(alignments) - (fix_ngram - 1)):
        if alignments[i] == -1:
            continue

        # fix alignment if generated n-gram is the same in source
        if all([alignments[i + n] == alignments[i] + n for n in range(fix_ngram)]):
            is_fixed[i:i + fix_ngram] = [1] * fix_ngram
    return is_fixed


def context_voting_step(
        source_tokens,
        candidate_tokens,
        alignments,
        is_fixed,
        window_size,
        context_agreement,
        sentence_separator_token=SENT_SEP_TOKEN,
):
    """ Let neighbors in a window around the current token vote for its alignment. """
    context_alignments = []
    for position in range(len(alignments)):
        if is_fixed and is_fixed[position]:
            context_alignments.append(alignments[position])
            continue

        left_context = [i for i in range(position - window_size, position) if 0 <= i]
        right_context = [i for i in range(position + 1, position + window_size + 1) if i < len(candidate_tokens)]

        # stop context at sentence boundaries
        left_context_tokens = [candidate_tokens[i] for i in left_context]
        if sentence_separator_token in left_context_tokens:
            sent_sep_index = rindex(left_context_tokens, sentence_separator_token)
            left_context = left_context[sent_sep_index + 1:]
        right_context_tokens = [candidate_tokens[i] for i in right_context]
        if sentence_separator_token in right_context_tokens:
            sent_sep_index = right_context_tokens.index(sentence_separator_token)
            right_context = right_context[:sent_sep_index]

        context_indices = left_context + right_context

        # in a diagonal pattern, neighbor i would predict the current alignment to be alignments[i] + relative distance,
        # where relative distance of position to i is: position - i
        # only aligned neighbors predict
        context_predictions = [alignments[i] + position - i for i in context_indices if alignments[i] > -1]

        # find matching alignment with most votes from context, select if votes exceed `context_agreement`
        alignment = alignments[position]
        for prediction, count in Counter(context_predictions).most_common():
            if -1 < prediction < len(source_tokens) and source_tokens[prediction] == candidate_tokens[position]:
                if count / len(context_predictions) >= context_agreement:
                    alignment = prediction
                    break

        context_alignments.append(alignment)
    return context_alignments


def align_by_context_voting(
        source_tokens,
        candidate_tokens,
        alignments,
        is_fixed,
        max_rounds,
        window_size,
        context_agreement,
        sentence_separator_token=SENT_SEP_TOKEN,
):
    """
    Iteratively let neighbors vote on alignment until converged or `max_rounds` is reached.
    Don't change alignments that are fixed.
    """
    seen_alignments = set(tuple(alignments))
    for r in range(max_rounds):
        alignments = context_voting_step(
            source_tokens, candidate_tokens, alignments, is_fixed, window_size, context_agreement,
            sentence_separator_token
        )
        if tuple(alignments) not in seen_alignments:
            seen_alignments.add(tuple(alignments))
        else:
            break
    return alignments


def segment_by_alignment(tokens, alignments, sentence_separator_token=SENT_SEP_TOKEN):
    """ Creates segments from alignments. Unaligned tokens and sentence separators are individual segments. """
    segments = []
    is_aligned = []
    cur_segment = []
    prev_aligned_pos = -10
    for token, aligned_pos in zip(tokens, alignments):
        if aligned_pos == -1 or token == sentence_separator_token:
            if cur_segment:
                segments.append(cur_segment)
                cur_segment = []
            segments.append([token])
            is_aligned.append(0)
            prev_aligned_pos = -1
        elif aligned_pos == prev_aligned_pos + 1:
            # continue current segment
            cur_segment.append(token)
            prev_aligned_pos = aligned_pos
        else:
            # end current segment and start a new one
            if cur_segment:
                segments.append(cur_segment)
            cur_segment = [token]
            prev_aligned_pos = aligned_pos
            is_aligned.append(1)

    if cur_segment:
        segments.append(cur_segment)
    return segments, is_aligned


def create_alignment_segments(tokens, alignments):
    """
    Creates segments for aligned tokens.
    Tokens belong to the same segment if they are consecutive and aligned with a consecutive span in the source.
    """
    segments = []
    cur_segment = -1
    for i, (token, aligned_pos) in enumerate(zip(tokens, alignments)):
        if aligned_pos == -1 or token in SPECIAL_TOKENS:
            segments.append(-1)
        elif i > 1 and aligned_pos - 1 == alignments[i - 1]:
            segments.append(cur_segment)
        else:
            cur_segment += 1
            segments.append(cur_segment)
    return segments


def create_decoding_entropy_segments(tokens, alignments, entropies):
    """
    Creates segments for unaligned tokens.
    Tokens belong to a segment as long as their decoding entropy doesn't surpass the segment's average.
    """
    segments = []
    cur_segment = -1
    cur_entropies = []
    for i, (token, aligned_pos, entropy) in enumerate(zip(tokens, alignments, entropies)):
        if i == 0 or aligned_pos > -1 or token in SPECIAL_TOKENS:
            segments.append(-1)
            cur_entropies = []
        elif not cur_entropies:
            # new segment after aligned token
            cur_segment += 1
            segments.append(cur_segment)
            cur_entropies.append(entropy)
        elif entropy > np.mean(cur_entropies):
            # new segment b/c entropy is larger than mean of current
            cur_segment += 1
            segments.append(cur_segment)
            cur_entropies = [entropy]
        else:
            # continue segment
            segments.append(cur_segment)
            cur_entropies.append(entropy)
    return segments


def get_sentence_segments(tokens, sentence_separator_token=SENT_SEP_TOKEN):
    sentence_segments = []
    cur_sent = 0
    for i, token in enumerate(tokens):
        sentence_segments.append(cur_sent)
        if (
                token == sentence_separator_token
                and i + 1 < len(tokens)
                and tokens[i + 1].startswith(WHITESPACE_CHAR)
        ):
            cur_sent += 1
    return sentence_segments


def get_probability_mass_between_alignments(
        alignments, alignment_segments, sentence_segments, self_attention, aggregate='mean',
):
    """ Returns the maximum probability mass to another alignment (in the same summary sentence) for each alignment. """
    assert aggregate in ['max', 'mean'], f'Unknown probability mass aggregation function: {aggregate}'
    probability_mass_to_other_alignments = [-1] * len(alignments)
    for sent_i in range(max(sentence_segments) + 1):
        alignment_segment_token_indices = defaultdict(list)
        for i in range(len(alignments)):
            if sentence_segments[i] != sent_i:
                continue
            alignment_segment_token_indices[alignment_segments[i]].append(i)
        if -1 in alignment_segment_token_indices:
            del alignment_segment_token_indices[-1]
        if len(alignment_segment_token_indices) < 2:
            continue

        # compute probability mass between alignments
        probability_mass_between_alignments = {}
        for seg_i in sorted(alignment_segment_token_indices):
            token_indices = alignment_segment_token_indices[seg_i]
            source_indices = [alignments[i] for i in token_indices]
            s1, e1 = source_indices[0], source_indices[-1] + 1
            for seg_j in sorted(alignment_segment_token_indices):
                if seg_j <= seg_i:
                    continue
                token_indices = alignment_segment_token_indices[seg_j]
                source_indices = [alignments[i] for i in token_indices]
                s2, e2 = source_indices[0], source_indices[-1] + 1
                probability_mass = np.sum(self_attention[s1:e1, s2:e2]) + np.sum(self_attention[s2:e2, s1:e1])
                normalized_prob_mass = probability_mass / (2 * (e1 - s1) * (e2 - s2))
                probability_mass_between_alignments[(seg_i, seg_j)] = normalized_prob_mass

        # aggregate the probability mass to other segments for each segment (max or mean)
        aggregated_probability_mass_to_other_segments = {}
        for seg_i in alignment_segment_token_indices:
            probability_mass_list = []
            for seg_j in alignment_segment_token_indices:
                if seg_j == seg_i:
                    continue
                segments = tuple(sorted((seg_i, seg_j)))
                probability_mass_list.append(probability_mass_between_alignments[segments])
            if aggregate == 'max':
                aggregated_probability_mass = max(probability_mass_list)
            else:
                aggregated_probability_mass = np.mean(probability_mass_list)
            aggregated_probability_mass_to_other_segments[seg_i] = aggregated_probability_mass

        # save each alignments max probability mass to other segments
        for alignment_segment, probability_mass in aggregated_probability_mass_to_other_segments.items():
            for i in alignment_segment_token_indices[alignment_segment]:
                probability_mass_to_other_alignments[i] = probability_mass
    return probability_mass_to_other_alignments


def get_segment_inverse_decoding_entropy(segments, decoding_entropies):
    """ Returns the smoothed inverse decoding entropy of the first token of each segment. """
    segment_decoding_entropies = [-1] * len(segments)
    segment_token_indices = defaultdict(list)
    for i, segment in enumerate(segments):
        segment_token_indices[segment].append(i)
    if -1 in segment_token_indices:
        del segment_token_indices[-1]  # delete unaligned
    first_token_entropies = {seg: decoding_entropies[indices[0]] for seg, indices in segment_token_indices.items()}
    for segment, indices in segment_token_indices.items():
        for i in indices:
            segment_decoding_entropies[i] = 1 / (first_token_entropies[segment] + 1)  # smoothed inverse entropy
    return segment_decoding_entropies


def get_attention_tensor(attentions, tokens, layers):
    attentions = standardize_shape(attentions)
    assert len(tokens) == attentions.size(-2), "Number of tokens doesn't match attentions"
    attentions = attentions.mean(dim=1)  # average over heads
    if layers:
        attentions = torch.stack([attentions[l] for l in layers]).mean(dim=0)
    else:
        attentions = attentions.mean(dim=0)
    return attentions.numpy()


def main(args):
    # load data
    dataset = {
        f[:-4]: os.path.join(args.bart_outputs_dir, f)
        for f in os.listdir(args.bart_outputs_dir)
        if f.endswith('.pkl')
    }

    # load examples already processed
    processed_ids = {}
    if os.path.exists(args.output_path):
        with open(args.output_path, 'r') as f:
            processed_ids = {json.loads(l)['id'] for l in f.readlines()}
    for example_id in dataset:
        if example_id in processed_ids:
            continue

        with open(dataset[example_id], 'rb') as f:
            outputs = pickle.load(f)
        article_tokens = outputs['article_tokens']
        summary_tokens = outputs['summary_tokens']
        source_tokens = outputs['source_tokens']
        candidate_tokens = outputs['candidate_tokens']
        scores = outputs['scores']
        cross_attentions = outputs['cross_attentions']
        encoder_attentions = outputs['encoder_attentions']

        # get decoding entropies
        probs = scipy.special.softmax(torch.stack(scores).numpy(), axis=-1)
        decoding_entropies = [scipy.stats.entropy(p) for p in probs]
        assert all([not np.isnan(e) for e in decoding_entropies]), "NaN in decoding entropies"

        # align source and summary first by attention ranks <= `max_rank`, then by context voting
        alignments = align_by_attention_rank(source_tokens, candidate_tokens, cross_attentions, args.max_rank)
        is_fixed = fix_alignments(alignments, args.fix_ngram)
        alignments = align_by_context_voting(
            source_tokens, candidate_tokens, alignments, is_fixed, args.max_rounds, args.window_size,
            args.context_agreement
        )

        # postprocess alignments: special tokens are unaligned
        alignments = [a if t not in SPECIAL_TOKENS else -1 for t, a in zip(candidate_tokens, alignments)]

        # split tokens into sentences when a sentence separator token is succeeded by a whitespace
        summary_sentence_segments = get_sentence_segments(summary_tokens)

        # get the fraction of each sentence's tokens that are aligned
        extrinsic_sentence_aligned = []
        cur_sent = 0
        cur_sent_tokens = 0
        cur_sent_aligned = 0
        for sent_i, token, aligned_pos in zip(summary_sentence_segments, candidate_tokens, alignments):
            if sent_i != cur_sent:
                extrinsic_sentence_aligned.extend(
                    [cur_sent_aligned / cur_sent_tokens] * len([1 for s in summary_sentence_segments if s == cur_sent])
                )
                cur_sent = sent_i
                cur_sent_tokens = 0
                cur_sent_aligned = 0
            if token not in SPECIAL_TOKENS:
                cur_sent_tokens += 1
                cur_sent_aligned += 1 if aligned_pos > -1 else 0
        extrinsic_sentence_aligned.extend(
            [cur_sent_aligned / cur_sent_tokens] * len([1 for s in summary_sentence_segments if s == cur_sent])
        )
        assert len(extrinsic_sentence_aligned) == len(candidate_tokens), "Not all summary sentence tokens processed"

        # intrinsic hallucinations: segment aligned tokens according to alignments,
        # compare encoder attentions among segments in the same sentence
        alignment_segments = create_alignment_segments(candidate_tokens, alignments)
        if all([als == -1 for als in alignment_segments]):
            intrinsic_association_strength = [-1] * len(candidate_tokens)
        else:
            intrinsic_association_strength = get_probability_mass_between_alignments(
                alignments, alignment_segments, summary_sentence_segments, encoder_attentions, aggregate='mean',
            )

        # extrinsic hallucinations: segment unaligned tokens according to moving average of decoding entropy,
        # classify based on the decoding entropy of the segment's first token (compared to max of unaligned tokens)
        decoding_entropy_segments = create_decoding_entropy_segments(candidate_tokens, alignments, decoding_entropies)
        if all([des == -1 for des in decoding_entropy_segments]):
            extrinsic_inverse_entropy = [-1] * len(candidate_tokens)
        else:
            extrinsic_inverse_entropy = get_segment_inverse_decoding_entropy(
                decoding_entropy_segments, decoding_entropies,
            )

        # write result
        result = {
            'id': example_id,
            'article_tokens': article_tokens,
            'summary_tokens': summary_tokens,
            'intrinsic_association_strength': [float(v) for v in intrinsic_association_strength],
            'extrinsic_sentence_aligned': extrinsic_sentence_aligned,
            'extrinsic_inverse_entropy': [float(v) for v in extrinsic_inverse_entropy],
        }
        with open(args.output_path, 'a') as f:
            json.dump(result, f)
            f.write('\n')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Computes BART-GBP scores.')
    parser.add_argument('--bart_outputs_dir', default='outputs/bart_outputs_frank')
    parser.add_argument('--output_path', default='results/frank/bart_gbp_scores.jsonl')

    # algorithm params
    parser.add_argument('--max_rank', type=int, default=3, help='Maximum attention rank that is used for alignment')
    parser.add_argument('--fix_ngram', type=int, default=0,
                        help='Fix alignments if they span the same n-gram in source and summary')
    parser.add_argument('--window_size', type=int, default=3, help='Context window size')
    parser.add_argument('--context_agreement', type=float, default=0.5,
                        help='Fraction of context votes that need to agree on the position of the current token')
    parser.add_argument('--max_rounds', type=int, default=10,
                        help='Maximum number of rounds when aligning by context voting')
    main(parser.parse_args())
