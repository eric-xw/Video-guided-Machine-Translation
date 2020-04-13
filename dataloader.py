import json
import numpy as np
import os 

import torch 
from torch.utils.data import Dataset, DataLoader 

def load_video_features(fpath, max_length):
    feats = np.load(fpath, encoding='latin1')[0]  # encoding='latin1' to handle the inconsistency between python 2 and 3
    if feats.shape[0] < max_length:
        dis = max_length - feats.shape[0]
        feats = np.lib.pad(feats, ((0, dis), (0, 0)), 'constant', constant_values=0)
    elif feats.shape[0] > max_length:
        inds = sorted(random.sample(range(feats.shape[0]), max_length))
        feats = feats[inds]
    assert feats.shape[0] == max_length
    return np.float32(feats)

class vatex_dataset(Dataset):
    def __init__(self, data_dir, file_path, img_dir, split_type, tokenizers, max_vid_len, pair):
        src, tgt = pair
        maps = {'en':'enCap', 'zh':'chCap'}
        self.data_dir = data_dir
        self.img_dir = img_dir
        # load tokenizer
        self.tok_src, self.tok_tgt = tokenizers
        self.max_vid_len = max_vid_len
        self.split_type = split_type

        with open(self.data_dir+file_path, 'r') as file:
            data = json.load(file)
        self.srccaps, self.tgtcaps = [], []
        self.sent_ids = []
        for d in data: 
            srccap = d[maps[src]][5:]
            self.srccaps.extend(srccap)
            sent_id = [''.join((d['videoID'],'&',str(i))) for i in range(len(srccap))]
            self.sent_ids.extend(sent_id)
            if split_type != 'test':
                tgtcap = d[maps[tgt]][5:]
                self.tgtcaps.extend(tgtcap)

    def __len__(self):
        return len(self.srccaps)

    def __getitem__(self, idx):
        str_srccap,  sent_id = self.srccaps[idx], self.sent_ids[idx]
        vid = sent_id[:-2]
        srccap, caplen_src = self.tok_src.encode_sentence(str_srccap)
        srcref = self.tok_src.encode_sentence_nopad_2str(str_srccap)
        img = load_video_features(os.path.join(self.data_dir,'vatex_features/',self.img_dir,vid+'.npy'), self.max_vid_len)
        if self.split_type != 'test':
            str_tgtcap = self.tgtcaps[idx]
            tgtcap, caplen_tgt = self.tok_tgt.encode_sentence(str_tgtcap)
            tgtref = self.tok_tgt.encode_sentence_nopad_2str(str_tgtcap)
            return srccap, tgtcap, img, caplen_src, caplen_tgt, srcref, tgtref
        else:
            return srccap, img, caplen_src, sent_id

def get_loader(data_dir, tokenizers, split_type, batch_size, max_vid_len, pair, num_workers, pin_memory):
    maps = {'train':['vatex_training_v1.0.json', 'trainval'], 'val': ['vatex_validation_v1.0.json', 'trainval'], 
        'test': ['vatex_public_test_english_v1.1.json', 'public_test']}
    file_path, img_dir = maps[split_type]
    mydata = vatex_dataset(data_dir, file_path, img_dir, split_type, tokenizers, max_vid_len, pair)
    if split_type in ['train']:
        shuffle = True
    elif split_type in ['val', 'test']:
        shuffle = False
    myloader = DataLoader(dataset=mydata, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory)
    return myloader

def create_split_loaders(data_dir, tokenizers, batch_size, max_vid_len, pair, num_workers=0, pin_memory=False):
    train_loader = get_loader(data_dir, tokenizers, 'train', batch_size, max_vid_len, pair, num_workers, pin_memory)
    val_loader = get_loader(data_dir, tokenizers, 'val', batch_size, max_vid_len, pair, num_workers, pin_memory)
    test_loader = get_loader(data_dir, tokenizers, 'test', batch_size, max_vid_len, pair, num_workers, pin_memory)
    # test_loader = [0]

    return train_loader, val_loader, test_loader

