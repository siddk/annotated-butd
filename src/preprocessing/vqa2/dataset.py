"""
dataset.py

Core script defining VQA-2 Dataset Class --> as well as utilities for loading image features from HDF5 files and
tensorizing data.

Reference: https://github.com/hengyuan-hu/bottom-up-attention-vqa/blob/master/dataset.py
"""
from torch.utils.data import Dataset

import h5py
import json
import numpy as np
import os
import torch


def create_entry(img, question, answer):
    answer.pop('image_id')
    answer.pop('question_id')

    entry = {
        'question_id': question['question_id'],
        'image_id':    question['image_id'],
        'image':       img,
        'question':    question['question'],
        'answer':      answer,
    }

    return entry


def load_dataset(img2idx, answer_targets, vqa2_q='data/VQA2-Questions', mode='train'):
    """ Load Dataset Entries """
    question_path = os.path.join(vqa2_q, 'v2_OpenEnded_mscoco_%s2014_questions.json' % mode)
    with open(question_path, 'r') as f:
        questions = sorted(json.load(f)['questions'], key=lambda x: x['question_id'])
    answers = sorted(answer_targets, key=lambda x: x['question_id'])
    assert len(questions) == len(answers), "Number of Questions != Number of Answers!"

    print('\t[*] Creating VQA-2 Entries...')
    entries = []
    for question, answer in zip(questions, answers):
        assert question['question_id'] == answer['question_id'], "Question ID != Answer ID!"
        assert question['image_id'] == answer['image_id'], "Question Image ID != Answer Image ID"

        entry = create_entry(img2idx['COCO_%s2014_%012d' % (mode, question["image_id"])], question, answer)
        entries.append(entry)

    return entries


class VQAFeatureDataset(Dataset):
    def __init__(self, dictionary, ans2label, label2ans, img2idx, ans_targets, vqa2_q='data/VQA2-Questions',
                 cache='data/VQA2-Cache', mode='train'):
        super(VQAFeatureDataset, self).__init__()
        self.ans2label, self.label2ans, self.num_ans_candidates = ans2label, label2ans, len(ans2label)
        self.dictionary, self.img2idx = dictionary, img2idx
        self.ans_targets = ans_targets

        # Load HDF5 Image Features
        print('\t[*] Loading HDF5 Features...')
        with h5py.File(os.path.join(cache, '%s36.hdf5' % mode), 'r') as hf:
            self.features = np.array(hf.get('image_features'))
            self.spatials = np.array(hf.get('spatial_features'))
        self.v_dim, self.s_dim = self.features.shape[2], self.spatials.shape[2]

        # Create the Dataset Entries by Iterating through the Data
        self.entries = load_dataset(self.img2idx, self.ans_targets, vqa2_q=vqa2_q, mode=mode)

        self.tokenize()
        self.tensorize()

    def tokenize(self, max_length=14):
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
        self.features = torch.from_numpy(self.features)
        self.spatials = torch.from_numpy(self.spatials)

        for entry in self.entries:
            question = torch.from_numpy(np.array(entry['q_token']))
            entry['q_token'] = question

            answer = entry['answer']
            labels = np.array(answer['labels'])
            scores = np.array(answer['scores'], dtype=np.float32)
            if len(labels):
                labels = torch.from_numpy(labels)
                scores = torch.from_numpy(scores)
                entry['answer']['labels'] = labels
                entry['answer']['scores'] = scores
            else:
                entry['answer']['labels'] = None
                entry['answer']['scores'] = None

    def __getitem__(self, index):
        entry = self.entries[index]
        features = self.features[entry['image']]
        spatials = self.spatials[entry['image']]

        question = entry['q_token']
        answer = entry['answer']
        labels = answer['labels']
        scores = answer['scores']
        target = torch.zeros(self.num_ans_candidates)

        if labels is not None:
            target.scatter_(0, labels, scores)

        return features, spatials, question, target

    def __len__(self):
        return len(self.entries)
