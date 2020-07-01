"""
dataset.py

Core script defining NLVR2 Dataset Class --> as well as utilities for loading image features from HDF5 files and
tensorizing data.

Reference: https://github.com/hengyuan-hu/bottom-up-attention-vqa/blob/master/dataset.py
"""
from torch.utils.data import Dataset
from tqdm import tqdm

import h5py
import json
import numpy as np
import os
import torch


def create_entry(example, img2idx):
    # Get Valid Image IDs
    img_id = "-".join(example['identifier'].split('-')[:-1])
    left_img, right_img = img_id + '-img0', img_id + '-img1'
    assert left_img in img2idx and right_img in img2idx, 'Image ID not in Index!'

    entry = {
        # ID
        'question_id': example['identifier'],

        # URLs
        'left_url': example['left_url'],
        'right_url': example['right_url'],

        # Image Idx
        'left_image': img2idx[left_img],
        'right_image': img2idx[right_img],

        # Question & Answer
        'question': example['sentence'],
        'answer': 1 if example['label'] == 'True' else 0
    }

    return entry


def load_dataset(img2idx, nlvr2_q='data/NLVR2-Questions', mode='train'):
    """ Load Dataset Entries """
    question_path = os.path.join(nlvr2_q, '%s.json' % mode)
    with open(question_path, 'r') as f:
        examples = [json.loads(x) for x in f.readlines()]

    print('\t[*] Creating NLVR2 %s Entries...' % mode)
    entries = []
    for ex in examples:
        entry = create_entry(ex, img2idx)
        entries.append(entry)

    return entries


class NLVR2FeatureDataset(Dataset):
    def __init__(self, dictionary, img2idx, nlvr2_q='data/NLVR2-Questions', cache='data/NLVR2-Cache', mode='train'):
        super(NLVR2FeatureDataset, self).__init__()
        self.dictionary, self.img2idx = dictionary, img2idx

        # Load HDF5 Image Features
        print('\t[*] Loading HDF5 Features...')
        self.v_dim, self.s_dim = 2048, 6
        self.hf = h5py.File(os.path.join(cache, '%s36.hdf5' % mode), 'r')
        self.features = self.hf.get('image_features')
        self.spatials = self.hf.get('spatial_features')

        # Crete the Dataset Entries by Iterating through the Data
        self.entries = load_dataset(self.img2idx, nlvr2_q=nlvr2_q, mode=mode)

        self.tokenize()
        self.tensorize()

    def tokenize(self, max_length=40):
        """ Tokenize and Front-Pad the Questions in the Dataset """
        for entry in self.entries:
            tokens = self.dictionary.tokenize(entry['question'], False)
            tokens = tokens[:max_length]
            if len(tokens) < max_length:
                # Note here we pad in front of the sentence
                padding = [self.dictionary.padding_idx] * (max_length - len(tokens))
                tokens = padding + tokens
            assert len(tokens) == max_length, "Tokenized & Padded Question != Max Length!"
            entry['q_token'] = tokens

    def tensorize(self):
        for entry in self.entries:
            question = torch.from_numpy(np.array(entry['q_token']))
            entry['q_token'] = question

    def __getitem__(self, index):
        entry = self.entries[index]

        # Combine Left Image and Right Image Features
        left_features = torch.from_numpy(np.array(self.features[entry['left_image']]))
        left_spatials = torch.from_numpy(np.array(self.spatials[entry['left_image']]))

        right_features = torch.from_numpy(np.array(self.features[entry['right_image']]))
        right_spatials = torch.from_numpy(np.array(self.spatials[entry['right_image']]))

        # Important - Add Indicator Variables for Differentiating Left/Right in Image Pairs!
        left_indicator = torch.zeros((36, 1))
        right_indicator = torch.ones((36, 1))

        features = torch.cat([left_features, right_features], dim=0)
        spatials = torch.cat([left_spatials, right_spatials], dim=0)
        indicators = torch.cat([left_indicator, right_indicator], dim=0)

        question = entry['q_token']
        target = entry['answer']

        return features, spatials, indicators, question, target

    def __len__(self):
        return len(self.entries)
