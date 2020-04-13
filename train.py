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
from utils import set_logger,read_vocab,write_vocab,build_vocab,Tokenizer,padding_idx,clip_gradient,adjust_learning_rate
from dataloader import create_split_loaders
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu, SmoothingFunction
cc = SmoothingFunction()

class Arguments():
    def __init__(self, config):
        for key in config:
            setattr(self, key, config[key])

def save_checkpoint(state, cp_file):
    torch.save(state, cp_file)


def count_paras(encoder, decoder, logging=None):
    '''
    Count model parameters.
    '''
    nparas_enc = sum(p.numel() for p in encoder.parameters())
    nparas_dec = sum(p.numel() for p in decoder.parameters())
    nparas_sum = nparas_enc + nparas_dec
    if logging is None: 
        print ('#paras of my model: enc {}M  dec {}M total {}M'.format(nparas_enc/1e6, nparas_dec/1e6, nparas_sum/1e6))
    else:
        logging.info('#paras of my model: enc {}M  dec {}M total {}M'.format(nparas_enc/1e6, nparas_dec/1e6, nparas_sum/1e6))

def setup(args, clear=False):
    '''
    Build vocabs from train or train/val set.
    '''
    TRAIN_VOCAB_EN, TRAIN_VOCAB_ZH = args.TRAIN_VOCAB_EN, args.TRAIN_VOCAB_ZH
    if clear: ## delete previous vocab
        for file in [TRAIN_VOCAB_EN, TRAIN_VOCAB_ZH]:
            if os.path.exists(file):
                os.remove(file)
    # Build English vocabs
    if not os.path.exists(TRAIN_VOCAB_EN):
        write_vocab(build_vocab(args.DATA_DIR, language='en'),  TRAIN_VOCAB_EN)
    #build Chinese vocabs
    if not os.path.exists(TRAIN_VOCAB_ZH):
        write_vocab(build_vocab(args.DATA_DIR, language='zh'), TRAIN_VOCAB_ZH)

    # set up seed
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)


def main(args):
    model_prefix = '{}_{}'.format(args.model_type, args.train_id)
    
    log_path = args.LOG_DIR + model_prefix + '/'
    checkpoint_path = args.CHK_DIR + model_prefix + '/'
    result_path = args.RESULT_DIR + model_prefix + '/'
    cp_file = checkpoint_path + "best_model.pth.tar"
    init_epoch = 0

    if not os.path.exists(log_path):
        os.makedirs(log_path)
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)

    ## set up the logger
    set_logger(os.path.join(log_path, 'train.log'))

    ## save argparse parameters
    with open(log_path+'args.yaml', 'w') as f:
        for k, v in args.__dict__.items():
            f.write('{}: {}\n'.format(k, v))

    logging.info('Training model: {}'.format(model_prefix))

    ## set up vocab txt
    setup(args, clear=True)
    print(args.__dict__)

    # indicate src and tgt language
    src, tgt = 'en', 'zh'

    maps = {'en':args.TRAIN_VOCAB_EN, 'zh':args.TRAIN_VOCAB_ZH}
    vocab_src = read_vocab(maps[src])
    tok_src = Tokenizer(language=src, vocab=vocab_src, encoding_length=args.MAX_INPUT_LENGTH)
    vocab_tgt = read_vocab(maps[tgt])
    tok_tgt = Tokenizer(language=tgt, vocab=vocab_tgt, encoding_length=args.MAX_INPUT_LENGTH)
    logging.info('Vocab size src/tgt:{}/{}'.format( len(vocab_src), len(vocab_tgt)) )

    ## Setup the training, validation, and testing dataloaders
    train_loader, val_loader, test_loader = create_split_loaders(args.DATA_DIR, (tok_src, tok_tgt), args.batch_size, args.MAX_VID_LENGTH, (src, tgt), num_workers=4, pin_memory=True)
    logging.info('train/val/test size: {}/{}/{}'.format( len(train_loader), len(val_loader), len(test_loader) ))

    ## init model
    if args.model_type == 's2s':
        encoder = Encoder(vocab_size=len(vocab_src), embed_size=args.wordembed_dim, hidden_size=args.enc_hid_size).cuda()
        decoder = Decoder(embed_size=args.wordembed_dim, hidden_size=args.dec_hid_size, vocab_size=len(vocab_tgt)).cuda()

    encoder.train()
    decoder.train()

    ## define loss
    criterion = nn.CrossEntropyLoss(ignore_index=padding_idx).cuda()
    ## init optimizer 
    dec_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, decoder.parameters()),
                                             lr=args.decoder_lr, weight_decay=args.weight_decay)
    enc_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, encoder.parameters()),
                                             lr=args.encoder_lr, weight_decay=args.weight_decay)

    count_paras(encoder, decoder, logging)

    ## track loss during training
    total_train_loss, total_val_loss = [], []
    best_val_bleu, best_epoch = 0, 0

    ## init time
    zero_time = time.time()

    # Begin training procedure
    earlystop_flag = False
    rising_count = 0

    for epoch in range(init_epoch, args.epochs):
        ## train for one epoch
        start_time = time.time()
        train_loss = train(train_loader, encoder, decoder, criterion, enc_optimizer, dec_optimizer, epoch)

        val_loss, sentbleu, corpbleu = validate(val_loader, encoder, decoder, criterion)
        end_time = time.time()

        epoch_time = end_time - start_time
        total_time = end_time - zero_time
        
        logging.info('Total time used: %s Epoch %d time uesd: %s train loss: %.4f val loss: %.4f sentbleu: %.4f corpbleu: %.4f' % (
                str(datetime.timedelta(seconds=int(total_time))),
                epoch, str(datetime.timedelta(seconds=int(epoch_time))), train_loss, val_loss, sentbleu, corpbleu))

        if corpbleu > best_val_bleu:
            best_val_bleu = corpbleu
            save_checkpoint({ 'epoch': epoch, 
                'enc_state_dict': encoder.state_dict(), 'dec_state_dict': decoder.state_dict(),
                'enc_optimizer': enc_optimizer.state_dict(), 'dec_optimizer': dec_optimizer.state_dict(),
                }, cp_file)
            best_epoch = epoch

        logging.info("Finished {0} epochs of training".format(epoch+1))

        total_train_loss.append(train_loss)
        total_val_loss.append(val_loss)

    logging.info('Best corpus bleu score {:.4f} at epoch {}'.format(best_val_bleu, best_epoch))

    ### the best model is the last model saved in our implementation
    logging.info ('************ Start eval... ************')
    eval(test_loader, encoder, decoder, cp_file, tok_tgt, result_path)

def train(train_loader, encoder, decoder, criterion, enc_optimizer, dec_optimizer, epoch):
    '''
    Performs one epoch's training.
    '''
    encoder.train()
    decoder.train()

    avg_loss = 0
    for cnt, (srccap, tgtcap, video, caplen_src, caplen_tgt, srcrefs, tgtrefs) in enumerate(train_loader, 1):

        srccap, tgtcap, video, caplen_src, caplen_tgt = srccap.cuda(), tgtcap.cuda(), video.cuda(), caplen_src.cuda(), caplen_tgt.cuda()

        src_out, init_hidden, vid_out = encoder(srccap, video) # fea: decoder input from encoder, should be of size (mb, encout_dim) = (mb, decoder_dim)
        scores = decoder(srccap, tgtcap, init_hidden, src_out, vid_out, args.MAX_INPUT_LENGTH, teacher_forcing_ratio=args.teacher_ratio)

        targets = tgtcap[:, 1:] # Since we decoded starting with <start>, the targets are all words after <start>, up to <end>
        loss = criterion(scores[:, 1:].contiguous().view(-1, decoder.vocab_size), targets.contiguous().view(-1))
        # Back prop.
        dec_optimizer.zero_grad()
        if enc_optimizer is not None:
            enc_optimizer.zero_grad()
        loss.backward()

        # Clip gradients
        if args.grad_clip is not None:
            clip_gradient(dec_optimizer, args.grad_clip)
            clip_gradient(enc_optimizer, args.grad_clip)

        # Update weights
        dec_optimizer.step()
        enc_optimizer.step()

        # Keep track of metrics
        avg_loss += loss.item()

    return avg_loss/cnt

def validate(val_loader, encoder, decoder, criterion):
    '''
    Performs one epoch's validation.
    '''
    decoder.eval()  # eval mode (no dropout or batchnorm)
    if encoder is not None:
        encoder.eval()

    references = list()  # references (true captions) for calculating corpus BLEU-4 score
    hypotheses = list()  # hypotheses (predictions)

    avg_loss = 0

    with torch.no_grad():
        # Batches
        for cnt, (srccap, tgtcap, video, caplen_src, caplen_tgt, srcrefs, tgtrefs) in enumerate(val_loader, 1):
            srccap, tgtcap, video, caplen_src, caplen_tgt = srccap.cuda(), tgtcap.cuda(), video.cuda(), caplen_src.cuda(), caplen_tgt.cuda()

            # Forward prop.
            src_out, init_hidden, vid_out = encoder(srccap, video) # fea: decoder input from encoder, should be of size (mb, encout_dim) = (mb, decoder_dim)
            scores, pred_lengths = decoder.inference(srccap, tgtcap, init_hidden, src_out, vid_out, args.MAX_INPUT_LENGTH)

            # Since we decoded starting with <start>, the targets are all words after <start>, up to <end>
            targets = tgtcap[:, 1:]
            scores_copy = scores.clone()

            # Calculate loss
            loss = criterion(scores[:, 1:].contiguous().view(-1, decoder.vocab_size), targets.contiguous().view(-1))

            # Hypotheses
            _, preds = torch.max(scores_copy, dim=2)
            preds = preds.tolist()
            temp_preds = list()
            for j, p in enumerate(preds):
                temp_preds.append(preds[j][1:pred_lengths[j]])  # remove pads and idx-0

            preds = temp_preds
            hypotheses.extend(preds) # preds= [1,2,3]

            tgtrefs = [ list(map(int, i.split())) for i in tgtrefs] # tgtrefs = [[1,2,3], [2,4,3], [1,4,5,]]
            
            for r in tgtrefs:
                references.append([r]) 

            assert len(references) == len(hypotheses)

            avg_loss += loss.item()

        # Calculate metrics
        avg_loss = avg_loss/cnt
        corpbleu = corpus_bleu(references, hypotheses)
        sentbleu = 0
        for i, (r, h) in enumerate(zip(references, hypotheses), 1):
            sentbleu += sentence_bleu(r, h, smoothing_function=cc.method7)
        sentbleu /= i

    return avg_loss, sentbleu, corpbleu

def eval(test_loader, encoder, decoder, cp_file, tok_tgt, result_path):
    '''
    Testing the model
    '''
    ### the best model is the last model saved in our implementation
    epoch = torch.load(cp_file)['epoch']
    logging.info ('Use epoch {0} as the best model for testing'.format(epoch))
    encoder.load_state_dict(torch.load(cp_file)['enc_state_dict'])
    decoder.load_state_dict(torch.load(cp_file)['dec_state_dict'])
    
    decoder.eval()  # eval mode (no dropout or batchnorm)
    if encoder is not None:
        encoder.eval()

    ids = list() # sentence ids
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

    ## save to json for submission
    dc = dict(zip(ids, hypotheses))
    print (len(dc))

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
