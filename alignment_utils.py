# SPDX-FileCopyrightText: 2022 Idiap Research Institute
#
# SPDX-License-Identifier: MIT

""" Utilities for attention alignment. """

import numpy as np

WHITESPACE_CHAR = u'\u0120'  # BART's whitespace character


def compute_attention_ranks(source_tokens, generated_tokens, attention_weights, output_index=False):
    """
    Returns a list of the rank of each generated token's attention weight to its corresponding source token,
    or -1 if there is no corresponding source token.
    """
    assert attention_weights.ndim == 2, "Unexpected number of attention dimensions"
    assert len(generated_tokens) == attention_weights.shape[0], "Generated tokens and attentions don't match"
    assert len(source_tokens) == attention_weights.shape[1], "Source tokens and attentions don't match"

    # replace BART's whitespace character
    source_tokens = [token.replace(WHITESPACE_CHAR, '').lower() for token in source_tokens]
    generated_tokens = [token.replace(WHITESPACE_CHAR, '').lower() for token in generated_tokens]

    attention_ranks = []
    for generated_token, attentions in zip(generated_tokens, attention_weights):
        sorted_indices = reversed(np.argsort(attentions))
        found = False
        for i, idx in enumerate(sorted_indices):
            if generated_token == source_tokens[idx]:
                attention_ranks.append(i if not output_index else (i, idx))
                found = True
                break
        if not found:
            attention_ranks.append(-1 if not output_index else (-1, None))
    return attention_ranks


def align_by_attention_rank(source_tokens, generated_tokens, attention_weights, max_rank):
    """ Aligns generated to source tokens if its attention rank is smaller than `max_rank`. """
    attention_ranks = compute_attention_ranks(source_tokens, generated_tokens, attention_weights, output_index=True)
    alignments = []
    for i, (generated_token, (rank, index)) in enumerate(zip(generated_tokens, attention_ranks)):
        if max_rank >= rank > -1:
            alignments.append(index)
        else:
            alignments.append(-1)
    return alignments


def rindex(a_list, element):
    """ Returns the right-most index of `element` in `a_list`. """
    for i in range(len(a_list) - 1, -1, -1):
        if a_list[i] == element:
            return i
    raise ValueError(f'{element} is not in list')


def standardize_shape(attentions):
    """ Brings attentions into a standardized shape of [layers, heads, from_tokens, to_tokens]. """
    assert attentions.dim() in [5, 6], f"Unexpected number of dimension: {attentions.dim()}"
    if attentions.dim() == 5:
        # encoder attentions have shape: [num_layers, batch_size, num_heads, from_tokens, to_tokens]
        return attentions.squeeze()
    else:
        # decoder/cross attentions have shape: [from_tokens, num_layers, batch_size, num_heads, 1, to_tokens]
        return attentions.squeeze().permute(1, 2, 0, 3)
