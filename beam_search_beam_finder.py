# SPDX-FileCopyrightText: 2022 Idiap Research Institute
#
# SPDX-License-Identifier: MIT

""" Finds the index of the eventually selected beam during decoding with beam search. """


class BeamFinder:

    @staticmethod
    def find_selected_indices_forward(sequence, beam_tokens, beam_indices):
        """
        Finds the selected beam at each generation step.
        Selected beam at first step is always 0.
        Every next step has a unique beam j where beam_tokens[i][j] == sequence[i]
        and beam_indices[i][j] == previously selected token index.
        """
        selected_beams = []
        cur_token_idx = 0
        for i in range(len(sequence)):
            cur_beam = cur_token_idx
            selected_beams.append(cur_beam)
            possible_beam_indices = set([j for j, bi in enumerate(beam_indices[i]) if bi == cur_beam])
            possible_token_indices = set([j for j, bt in enumerate(beam_tokens[i]) if bt == sequence[i]])
            possible_next_token_idx = possible_beam_indices & possible_token_indices
            assert len(possible_next_token_idx) == 1, f"Possible next token index not unique: {possible_next_token_idx}"
            cur_token_idx = possible_next_token_idx.pop()
        return selected_beams

    @staticmethod
    def find_selected_indices_forward1(sequence, beam_tokens, beam_indices):
        """
        Finds the selected token at each step given the final sequence.
        Observation: Currently selected token's index is next beam.
        So start with beam 0 (always in first step) and add index of selected token until seq_len - 1.
        """
        selected_indices = [0]
        for i, (selected_token, possible_tokens, indices) in enumerate(zip(sequence[:-1], beam_tokens, beam_indices)):
            # find the beam index of the current selected token
            if selected_token in possible_tokens:
                if possible_tokens.count(selected_token) == 1:
                    # if there's only one possible token matching the selected token, add its index
                    selected_indices.append(possible_tokens.index(selected_token))
                else:
                    # if there are multiple possible tokens matching the selected token, we have to look at the prefix
                    possible_token_indices = [j for j, token in enumerate(possible_tokens) if token == selected_token]
                    previous_selected_idx = selected_indices[-1]
                    found = False
                    for candidate_idx in possible_token_indices:
                        candidate_previous_idx = indices[candidate_idx]
                        if candidate_previous_idx == previous_selected_idx:
                            selected_indices.append(candidate_idx)
                            found = True
                            break
                    if not found:
                        # DEBUG
                        print(sequence)
                        print(beam_tokens)
                        print(beam_indices)
                        raise ValueError(f"No prefix matches one of the possible next tokens at position {i}.")
            else:
                # DEBUG
                print(sequence)
                print(beam_tokens)
                print(beam_indices)
                # error: no tokens match the selected one
                raise ValueError(f"Selected token at position {i} does not appear in possible tokens.")
        return selected_indices

    @staticmethod
    def find_selected_indices_forward2(sequence, beam_tokens, beam_indices):
        """ Finds the selected token at each step given the final sequence: Selected beam index at current position. """
        selected_indices = []
        for i, (selected_token, possible_tokens, indices) in enumerate(zip(sequence, beam_tokens, beam_indices)):
            # find the beam index of the current selected token
            if selected_token in possible_tokens:
                if possible_tokens.count(selected_token) == 1:
                    # if there's only one possible token matching the selected token, add its index
                    selected_indices.append(indices[possible_tokens.index(selected_token)])
                else:
                    # if there are multiple possible tokens matching the selected token, we have to look at the prefix
                    possible_beam_indices = [indices[j] for j, token in enumerate(possible_tokens) if token == selected_token]
                    previous_selected_token = sequence[i - 1]
                    possible_previous_tokens = [beam_tokens[i - 1][pbi] for pbi in possible_beam_indices]
                    if possible_previous_tokens.count(previous_selected_token) == 1:
                        # only one possible previous token matches, return its beam index
                        selected_indices.append(possible_beam_indices[possible_previous_tokens.index(previous_selected_token)])
                    else:
                        # multiple previous tokens match
                        # only keep those with matching previous tokens
                        possible_beam_indices = [pbi for pbi in possible_beam_indices if beam_tokens[i - 1][pbi] == previous_selected_token]

                        # among those, only 1 can match the previously selected beam index
                        previous_selected_idx = selected_indices[-1]
                        possible_previous_indices = [beam_indices[i - 1][pbi] for pbi in possible_beam_indices]
                        if possible_previous_indices.count(previous_selected_idx) == 1:
                            selected_indices.append(possible_beam_indices[possible_previous_indices.index(previous_selected_idx)])
                        else:
                            # DEBUG
                            print(sequence)
                            print(beam_tokens)
                            print(beam_indices)
                            raise ValueError(f"No prefix matches one of the possible next beams at position {i}.")
            else:
                # DEBUG
                print(sequence)
                print(beam_tokens)
                print(beam_indices)
                # error: no tokens match the selected one
                raise ValueError(f"Selected token at position {i} does not appear in possible tokens.")
        return selected_indices

    @staticmethod
    def find_last_beam(sequence, beam_tokens, beam_indices):
        """ Finds the beam index of the token at the current position. """
        selected_token = sequence[-1]
        possible_tokens = beam_tokens[-1]
        possible_indices = beam_indices[-1]
        if selected_token in possible_tokens:
            if possible_tokens.count(selected_token) == 1:
                # if there's only one possible token matching the selected token, return its index
                return possible_indices[possible_tokens.index(selected_token)]
            else:
                # if there are multiple possible tokens matching the selected token, we have to look at the prefix
                # save initial and current beam index, then step back with current index
                possible_beam_idx = [(possible_indices[j], j) for j, token in enumerate(possible_tokens) if token == selected_token]
                for pos in range(len(sequence) - 1, 0, -1):
                    previous_possible_beam_idx = []
                    for initial_idx, cur_idx in possible_beam_idx:
                        previous_idx = beam_indices[pos][cur_idx]
                        previous_token = beam_tokens[pos - 1][previous_idx]
                        if sequence[pos - 1] == previous_token:
                            previous_possible_beam_idx.append((initial_idx, previous_idx))
                    if len(previous_possible_beam_idx) == 1:
                        return previous_possible_beam_idx[0][0]
                    elif len(previous_possible_beam_idx) == 0:
                        # DEBUG
                        print(sequence)
                        print(selected_token)
                        print(len(sequence) - 1)
                        print(beam_tokens)
                        print(beam_indices)
                        raise ValueError(f"No prefix matches one of the possible next tokens at position {pos}.")
                    else:
                        possible_beam_idx = previous_possible_beam_idx
        else:
            # DEBUG
            print(sequence)
            print(selected_token)
            print(len(sequence) - 1)
            print(beam_tokens)
            print(beam_indices)
            # error: no tokens match the selected one
            raise ValueError(f"Selected token does not appear in possible tokens at position {len(sequence) - 1}.")

    @staticmethod
    def find_selected_indices_backward(beam_indices, end_idx):
        """ Reconstructs the selected index of the best beam from the beam indices (reordering the beams). """
        selected_indices = [end_idx]
        for i in reversed(range(len(beam_indices) - 1)):  # i: len-2 -> 0 (end_idx already selected)
            prev_idx = selected_indices[-1]
            selected_indices.append(beam_indices[i][prev_idx])
        selected_indices.reverse()
        return selected_indices

    @staticmethod
    def do_selected_indices_match(sequence, selected_indices, beam_tokens):
        """ Checks whether the tokens of the selected beam match the sequence at each step. """
        if len(sequence) != len(selected_indices):
            return False
        if selected_indices[0] != 0:
            return False
        for i in range(1, len(sequence)):
            selected_beam = selected_indices[i]
            if beam_tokens[i - 1][selected_beam] != sequence[i - 1]:
                return False
        return True

    @staticmethod
    def find_selected_indices(sequence, beam_search_tokens, beam_search_indices):
        selected_indices = BeamFinder.find_selected_indices_forward(sequence, beam_search_tokens, beam_search_indices)
        selected_indices0 = BeamFinder.find_selected_indices_forward1(sequence, beam_search_tokens, beam_search_indices)
        selected_indices1 = BeamFinder.find_selected_indices_forward2(sequence, beam_search_tokens, beam_search_indices)
        last_beam_idx = BeamFinder.find_last_beam(sequence, beam_search_tokens, beam_search_indices)
        selected_indices2 = BeamFinder.find_selected_indices_backward(beam_search_indices, last_beam_idx)
        # DEBUG
        if not (selected_indices == selected_indices0 == selected_indices1 == selected_indices2):
            print(f'Do indices match: {BeamFinder.do_selected_indices_match(sequence, selected_indices, beam_search_tokens)}')
            print(f'Do indices0 match: {BeamFinder.do_selected_indices_match(sequence, selected_indices0, beam_search_tokens)}')
            print(f'Do indices1 match: {BeamFinder.do_selected_indices_match(sequence, selected_indices1, beam_search_tokens)}')
            print(f'Do indices2 match: {BeamFinder.do_selected_indices_match(sequence, selected_indices2, beam_search_tokens)}')
            print()
            print(sequence)
            print(beam_search_tokens)
            print(beam_search_indices)
        assert selected_indices == selected_indices0
        assert selected_indices == selected_indices1
        assert selected_indices == selected_indices2
        assert BeamFinder.do_selected_indices_match(sequence, selected_indices, beam_search_tokens)
        return selected_indices
