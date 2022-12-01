# SPDX-FileCopyrightText: 2022 Idiap Research Institute
#
# SPDX-License-Identifier: MIT

""" Saves BART outputs used by bart_gbp_scores.py. """

import json
import os
import pickle
import torch
from transformers import BartTokenizer, BartForConditionalGeneration

from bart_gbp_scores import WHITESPACE_CHAR, get_attention_tensor
from beam_search_bart import GenerationMixinEncoderDecoder


class BartSummarizer(BartForConditionalGeneration, GenerationMixinEncoderDecoder):
    """ BART summarization model wrapper that selects only the used attentions from beam search decoding. """


def main(args):
    # load model
    tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')
    model = BartSummarizer.from_pretrained('facebook/bart-large-cnn')
    model.eval()

    # load data
    with open(args.data_path, 'r') as f:
        if args.data_path.endswith('.json'):
            dataset = json.load(f)
        elif args.data_path.endswith('.jsonl'):
            dataset = [json.loads(l) for l in f.readlines()]
        else:
            raise ValueError(f'Unknown data file format: {args.data_path}')

    # create output dir
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    for example in dataset:
        # skip existing outputs
        output_path = os.path.join(args.output_dir, f"{example['id']}.pkl")
        if os.path.exists(output_path):
            continue

        # get model outputs
        inputs = tokenizer(example['article'], max_length=1024, truncation=True, return_tensors='pt')
        with torch.no_grad():
            outputs = model.generate(**inputs, output_scores=True, output_attentions=True, return_dict_in_generate=True)
        assert len(outputs.sequences) == 1, "Decoded batch size is not 1"

        # get article and candidate tokens
        article_tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
        summary_tokens = tokenizer.convert_ids_to_tokens(outputs.sequences[0])[1:-1]  # remove BOS, EOS tokens

        # remove whitespace and lowercase for lexical overlap during alignment
        source_tokens = [t.replace(WHITESPACE_CHAR, '').lower() for t in article_tokens]
        candidate_tokens = [t.replace(WHITESPACE_CHAR, '').lower() for t in summary_tokens]

        # create attention tensors from the selected layers
        cross_attentions = torch.stack([torch.stack(attentions) for attentions in outputs.cross_attentions])
        cross_attentions = get_attention_tensor(cross_attentions, candidate_tokens, args.cross_attention_layers)
        encoder_attentions = torch.stack(outputs.encoder_attentions)
        encoder_attentions = get_attention_tensor(encoder_attentions, source_tokens, args.encoder_layers)

        # write outputs
        outputs = {
            'article_tokens': article_tokens,
            'summary_tokens': summary_tokens,
            'source_tokens': source_tokens,
            'candidate_tokens': candidate_tokens,
            'scores': outputs.scores,
            'cross_attentions': cross_attentions,
            'encoder_attentions': encoder_attentions,
        }
        with open(output_path, 'wb') as f:
            pickle.dump(outputs, f)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Saves BART alignment outputs.')
    parser.add_argument('--data_path', default='data/frank_annotations.jsonl')
    parser.add_argument('--output_dir', default='outputs/bart_outputs_frank')
    parser.add_argument('--cross_attention_layers', type=int, nargs='+', default=None,
                        help='Use cross-attention from these layers (default: all)')
    parser.add_argument('--encoder_layers', type=int, nargs='+', default=None,
                        help='Use encoder self-attention from these layers (default: all)')
    main(parser.parse_args())
