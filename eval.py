import sys 
import os
import argparse
import time
import datetime
import logging
import numpy as np 
import json

import torch
import torch.nn as nn

from model import Encoder, Decoder
from utils import set_logger,read_vocab,write_vocab,build_vocab,Tokenizer,padding_idx
from dataloader import create_split_loaders
from train import setup
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu, SmoothingFunction

cc = SmoothingFunction()

class Arguments():
    def __init__(self, config):
        for key in config:
            setattr(self, key, config[key])

def main(args):
    model_prefix = '{}_{}'.format(args.model_type, args.train_id)

    checkpoint_path = args.CHK_DIR + model_prefix + '/'
    result_path = args.RESULT_DIR + model_prefix + '/'
    cp_file = checkpoint_path + "best_model.pth.tar"

    if not os.path.exists(checkpoint_path):
        sys.exit('No checkpoint_path found {}'.format(checkpoint_path))

    print('Loading model: {}'.format(model_prefix))
    
    # set up vocab txt
    setup(args, clear=False)
    print(args.__dict__)

    # indicate src and tgt language
    src, tgt = 'en', 'zh'

    maps = {'en':args.TRAIN_VOCAB_EN, 'zh':args.TRAIN_VOCAB_ZH}
    vocab_src = read_vocab(maps[src])
    tok_src = Tokenizer(language=src, vocab=vocab_src, encoding_length=args.MAX_INPUT_LENGTH)
    vocab_tgt = read_vocab(maps[tgt])
    tok_tgt = Tokenizer(language=tgt, vocab=vocab_tgt, encoding_length=args.MAX_INPUT_LENGTH)
    print ('Vocab size src/tgt:{}/{}'.format( len(vocab_src), len(vocab_tgt)) )

    # Setup the training, validation, and testing dataloaders
    train_loader, val_loader, test_loader = create_split_loaders(args.DATA_DIR,(tok_src, tok_tgt), args.batch_size, args.MAX_VID_LENGTH, (src, tgt), num_workers=4, pin_memory=True)
    print ('train/val/test size: {}/{}/{}'.format( len(train_loader), len(val_loader), len(test_loader) ))

    ## init model
    if args.model_type == 's2s':
        encoder = Encoder(vocab_size=len(vocab_src), embed_size=args.wordembed_dim, hidden_size=args.enc_hid_size).cuda()
        decoder = Decoder(embed_size=args.wordembed_dim, hidden_size=args.dec_hid_size, vocab_size=len(vocab_tgt)).cuda()

    ### load best model and eval
    print ('************ Start eval... ************')
    eval(test_loader, encoder, decoder, cp_file, tok_tgt, result_path)


def eval(test_loader, encoder, decoder, cp_file, tok_tgt, result_path):
    '''
    Testing the model
    '''
    ### the best model is the last model saved in our implementation
    epoch = torch.load(cp_file)['epoch']
    print ('Use epoch {0} as the best model for testing'.format(epoch))
    encoder.load_state_dict(torch.load(cp_file)['enc_state_dict'])
    decoder.load_state_dict(torch.load(cp_file)['dec_state_dict'])
    
    decoder.eval()  # eval mode (no dropout or batchnorm)
    if encoder is not None:
        encoder.eval()

    ids = list() 
    hypotheses = list()  # hypotheses (predictions)

    with torch.no_grad():
        # Batches
        for cnt, (srccap, video, caplen_src, sent_id) in enumerate(test_loader, 1):
            srccap, video, caplen_src = srccap.cuda(), video.cuda(), caplen_src.cuda()

            # Forward prop.
            src_out, init_hidden, vid_out = encoder(srccap, video) # fea: decoder input from encoder, should be of size (mb, encout_dim) = (mb, decoder_dim)
            preds, pred_lengths = decoder.beam_decoding(srccap, init_hidden, src_out, vid_out, args.MAX_INPUT_LENGTH, beam_size=5)

            # Hypotheses
            preds = preds.tolist()
            temp_preds = list()
            for j, p in enumerate(preds):
                temp_preds.append(preds[j][:pred_lengths[j]])  # remove pads and idx-0

            preds = [tok_tgt.decode_sentence(t) for t in temp_preds]

            hypotheses.extend(preds) # preds= [[1,2,3], ... ]

            ids.extend(sent_id)


    ## save to json
    dc = dict(zip(ids, hypotheses))

    if not os.path.exists(result_path):
        os.makedirs(result_path)
    with open(result_path+'submission.json', 'w') as fp:
        json.dump(dc, fp)

    return dc


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='VMT')
    parser.add_argument('--config', type=str, default='./configs.yaml')
    args = parser.parse_args()
    with open(args.config, 'r') as fin:
        import yaml
        args = Arguments(yaml.load(fin))
    main(args)
