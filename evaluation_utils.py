# SPDX-FileCopyrightText: 2022 Idiap Research Institute
#
# SPDX-License-Identifier: MIT

""" Utilities to evaluate outputs of hallucination detection models. """

from collections import defaultdict

import json
import numpy as np
import string
from transformers import BartTokenizer

from bart_gbp_scores import BOS_TOKEN, WHITESPACE_CHAR

CUSTOM_PUNCTUATION = ['’', 'âĢ', 'Ļ', 'Âł']  # punctuation encountered in our examples
PUNCTUATION = [char for char in string.punctuation] + CUSTOM_PUNCTUATION


def flatten(a_list):
    """ Flatten a potentially once nested list. """
    if type(a_list[0]) == list:
        return [item for b_list in a_list for item in b_list]
    return a_list


def serialize(ndarray):
    """ Converts numpy ndarray to python float list. """
    return [float(v) for v in ndarray]


def find_sentence_boundaries_in_tokens(tokenizer, sentence, tokens):
    """ Finds the first and last index of `sentence` in `tokens`. """
    # tokenize first and last word in sentence
    first_word = sentence.split()[0]
    last_n_words = 1
    last_word = sentence.split()[-1]
    while sentence.count(' ' + last_word) > 1:
        last_n_words += 1
        last_word = ' '.join(sentence.split()[-last_n_words:])
    first_tokens = tokenizer.tokenize(first_word)
    first_tokens_space = tokenizer.tokenize(' ' + first_word)
    last_tokens = tokenizer.tokenize(' ' + last_word)
    first_tokens_space[0] = first_tokens_space[0].replace(WHITESPACE_CHAR, '')
    last_tokens = [t.replace(WHITESPACE_CHAR, '') for t in last_tokens]

    # check if start index is correct
    # first sentence starts with BOS token
    start_idx = 1 if tokens[0] == BOS_TOKEN else 0
    if not (
            all(t1 == t2 for t1, t2 in zip(tokens[start_idx:], first_tokens)) or
            all(t1 == t2 for t1, t2 in zip(tokens[start_idx:], first_tokens_space))
    ):
        raise RuntimeError(f"Failed to automatically detect first token of sentence.\nSentence: {sentence}\n"
                           f"Tokens: {tokens}\nFirst tokens: {first_tokens, first_tokens_space}")

    # find end index
    end_idx = -1
    for i, token in enumerate(tokens[start_idx:]):
        if all(t1 == t2 for t1, t2 in zip(tokens[i:], last_tokens)):
            end_idx = i + len(last_tokens)
            break
    if end_idx == -1:
        raise RuntimeError(f"Can't find end index of sentence.\nSentence: {sentence}\nTokens: {tokens}\n"
                           f"Last tokens: {last_tokens}")

    return start_idx, end_idx


def align_tokens_to_words(tokenizer, tokens, words):
    """ Aligns tokens and words. Returns alignment in the form: {word_idx: [token_indices]}. """
    alignment = {}
    tokens_start_idx = 0
    for word_idx, word in enumerate(words):
        for token_idx in range(tokens_start_idx, len(tokens)):
            if tokenizer.convert_tokens_to_string(tokens[tokens_start_idx:token_idx + 1]) == word:
                alignment[word_idx] = list(range(tokens_start_idx, token_idx + 1))
                tokens_start_idx = token_idx + 1
                break
        while word_idx not in alignment and tokens[tokens_start_idx] in PUNCTUATION:
            tokens_start_idx += 1
            for token_idx in range(tokens_start_idx, len(tokens)):
                if tokenizer.convert_tokens_to_string(tokens[tokens_start_idx:token_idx + 1]) == word:
                    alignment[word_idx] = list(range(tokens_start_idx, token_idx + 1))
                    tokens_start_idx = token_idx + 1
                    break
        if word_idx not in alignment:
            raise RuntimeError(f"Failed to align tokens to words: {word} not found in tokens.\nWords: {words}\n"
                               f"Tokens: {tokens}\nCurrent tokens: {tokens[tokens_start_idx:]}")
    return alignment
