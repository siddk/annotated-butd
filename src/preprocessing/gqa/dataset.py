"""
dataset.py

Core script defining GQA Dataset Class --> as well as utilities for loading image features from HDF5 files and
tensorizing data.

Reference: https://github.com/hengyuan-hu/bottom-up-attention-vqa/blob/master/dataset.py
"""
from torch.utils.data import Dataset

import h5py
import json
import numpy as np
import os
import torch


def create_entry(example, qid, img2idx, ans2label):
    # Get Valid Image ID
    img_id = example['imageId']
    assert img_id in img2idx, 'Image ID not in Index!'

    entry = {
        'question_id': qid,

        # ID
        'image_id': img_id,

        # Image Idx
        'image': img2idx[img_id],

        # Question & Answer
        'question': example['question'],
        'answer': ans2label[example['answer'].lower()]
    }

    return entry


def load_dataset(img2idx, ans2label, gqa_q='data/GQA-Questions', mode='train'):
    """ Load Dataset Entries """
    question_path = os.path.join(gqa_q, '%s_balanced_questions.json' % mode)
    with open(question_path, 'r') as f:
        examples = json.load(f)

    print('\t[*] Creating GQA %s Entries...' % mode)
    entries = []
    for ex_key in sorted(examples):
        entry = create_entry(examples[ex_key], ex_key, img2idx, ans2label)
        entries.append(entry)

    return entries


class GQAFeatureDataset(Dataset):
    def __init__(self, dictionary, ans2label, label2ans, img2idx, gqa_q='data/GQA-Questions', cache='data/GQA-Cache',
                 mode='train'):
        super(GQAFeatureDataset, self).__init__()
        self.dictionary, self.ans2label, self.label2ans, self.img2idx = dictionary, ans2label, label2ans, img2idx

        # Load HDF5 Image Features
        print('\t[*] Loading HDF5 Features...')
        if mode in ['train', 'val']:
            prefix = 'trainval'
        else:
            prefix = 'testdev'

        self.v_dim, self.s_dim = 2048, 6
        self.hf = h5py.File(os.path.join(cache, '%s36.hdf5' % prefix), 'r')
        self.features = self.hf.get('image_features')
        self.spatials = self.hf.get('spatial_features')

        # Create the Dataset Entries by Iterating through the Data
        self.entries = load_dataset(self.img2idx, ans2label, gqa_q=gqa_q, mode=mode)

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

        # Get Features
        features = torch.from_numpy(np.array(self.features[entry['image']]))
        spatials = torch.from_numpy(np.array(self.spatials[entry['image']]))
        question = entry['q_token']
        target = entry['answer']

        return features, spatials, question, target

    def __len__(self):
        return len(self.entries)
