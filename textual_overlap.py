# SPDX-FileCopyrightText: 2022 Idiap Research Institute
#
# SPDX-License-Identifier: MIT

""" Aligns the source and summary based on n-gram matching. """

from collections import defaultdict


class TextualOverlap:

    def __init__(self, shortest_ngram: int):
        assert shortest_ngram > 0, "Shortest ngram should be at least 1"
        self.shortest_ngram = shortest_ngram

    def get_alignment(self, source_tokens, summary_tokens):
        """ Returns alignment as a list of triples of (summary_pos, source_pos, length). """
        alignment = []
        source_starts = defaultdict(list)
        for i, token in enumerate(source_tokens):
            source_starts[token].append(i)

        # compute initial matching in the form {summary_pos: [(src_pos1, length1), (src_pos2, length2)]}
        matching = defaultdict(list)
        for i, token in enumerate(summary_tokens):
            for src_pos in source_starts[token]:
                length = 1
                max_possible_length = min(len(summary_tokens) - i, len(source_tokens) - src_pos)
                for j in range(1, max_possible_length):
                    if source_tokens[src_pos + j] == summary_tokens[i + j]:
                        length += 1
                    else:
                        break
                if length >= self.shortest_ngram:
                    matching[i].append((src_pos, length))

        # successively add the longest span to the alignment, and remove all participating summary positions
        while matching:
            # find longest span
            longest = 0
            summary_pos = -1
            source_pos = -1
            for sum_pos, matching_list in matching.items():
                for src_pos, length in matching_list:
                    if length > longest:
                        longest = length
                        summary_pos = sum_pos
                        source_pos = src_pos

            # add span to alignment
            alignment.append((summary_pos, source_pos, longest))

            # remove all participating summary positions
            remove_indices = set([i for i in range(summary_pos, summary_pos + longest)])
            new_matching = {}
            for sum_pos, matching_list in matching.items():
                if sum_pos not in remove_indices:
                    new_matching_list = []
                    for src_pos, length in matching_list:
                        while sum_pos + length - 1 in remove_indices:
                            length -= 1
                        if length >= self.shortest_ngram:
                            new_matching_list.append((src_pos, length))
                    if new_matching_list:
                        new_matching[sum_pos] = new_matching_list
            matching = new_matching

        return sorted(alignment)


if __name__ == '__main__':
    # tests
    unigram = TextualOverlap(1)
    source_tokens = 'the quick brown fox jumps over the lazy dog'.split()
    summary_tokens = 'the brown fox jumps over the dog'.split()
    target_alignment = [(0, 0, 1), (1, 2, 5), (6, 8, 1)]
    generated_alignment = unigram.get_alignment(source_tokens, summary_tokens)
    assert target_alignment == generated_alignment

    bigram = TextualOverlap(2)
    source_tokens = 'The quick brown fox jumps over the lazy dog'.split()
    summary_tokens = 'The brown fox jumps over the dog'.split()
    target_alignment = [(1, 2, 5)]
    generated_alignment = bigram.get_alignment(source_tokens, summary_tokens)
    assert target_alignment == generated_alignment

    trigram = TextualOverlap(3)
    source_tokens = 'The quick brown fox jumps over the lazy dog'.split()
    summary_tokens = 'The brown fox jumps over the dog'.split()
    target_alignment = [(1, 2, 5)]
    generated_alignment = trigram.get_alignment(source_tokens, summary_tokens)
    assert target_alignment == generated_alignment

    quintgram = TextualOverlap(5)
    source_tokens = 'The quick brown fox jumps over the lazy dog'.split()
    summary_tokens = 'The brown fox jumps over the dog'.split()
    target_alignment = [(1, 2, 5)]
    generated_alignment = quintgram.get_alignment(source_tokens, summary_tokens)
    assert target_alignment == generated_alignment

    six_gram = TextualOverlap(6)
    source_tokens = 'The quick brown fox jumps over the lazy dog'.split()
    summary_tokens = 'The brown fox jumps over the dog'.split()
    target_alignment = []
    generated_alignment = six_gram.get_alignment(source_tokens, summary_tokens)
    assert target_alignment == generated_alignment

    # multiple alignment to same source tokens accepted
    trigram = TextualOverlap(3)
    source_tokens = 'The quick brown fox jumps over the lazy dog'.split()
    summary_tokens = 'The brown fox jumps over the dog The quick brown fox'.split()
    target_alignment = [(1, 2, 5), (7, 0, 4)]
    generated_alignment = trigram.get_alignment(source_tokens, summary_tokens)
    assert target_alignment == generated_alignment

    trigram = TextualOverlap(3)
    source_tokens = 'The quick brown fox jumps over the lazy dog . The brown fox is quick'.split()
    summary_tokens = 'The brown fox jumps quickly over the dog . The quick brown fox'.split()
    target_alignment = [(0, 10, 3), (9, 0, 4)]
    generated_alignment = trigram.get_alignment(source_tokens, summary_tokens)
    assert target_alignment == generated_alignment

    trigram = TextualOverlap(3)
    source_tokens = 'The quick brown fox jumps over the lazy dog . The brown fox jumps'.split()
    summary_tokens = 'The brown fox jumps quickly over the dog . The quick brown fox'.split()
    target_alignment = [(0, 10, 4), (9, 0, 4)]
    generated_alignment = trigram.get_alignment(source_tokens, summary_tokens)
    assert target_alignment == generated_alignment

    unigram = TextualOverlap(1)
    source_tokens = ("It's not the first time the married mother-of-two has appeared in the spotlight. "
                     "She has been placed on administrative leave.").split()
    summary_tokens = 'The married mother-of-two has been placed on administrative leave.'.split()
    target_alignment = [(1, 6, 2), (3, 14, 6)]
    generated_alignment = unigram.get_alignment(source_tokens, summary_tokens)
    assert target_alignment == generated_alignment

