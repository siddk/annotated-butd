"""
**GQA.py** is a streamlined port of the [Annotated BUTD](https://github.com/siddk/annotated-butd) repository,
to exist in a single-file (for those that prefer to see the whole story all at once).

It steps through each of the stages of training a Bottom-Up Top-Down (BUTD) model
on the GQA Dataset, including:

 - **Preprocessing**
 - **Architecture Definition**
 - **Training** (facilitated by [Pytorch-Lightning](https://pytorch-lightning.readthedocs.io/en/latest/))

Note that this file only includes streamlined code for the original Bottom-Up Top-Down Model, with the simple
product-based fusion operation.

For the BUTD-FiLM Model, consult the
[Modular branch](https://github.com/siddk/annotated-butd/blob/modular/src/models/film.py) of the
[Annotated BUTD](https://github.com/siddk/annotated-butd) repository -- the code is quite similar.


To run this executable file, run the following (append `--gpus 1` if running on GPU):

```
python gqa.py --run_name GQA
```

---
"""
from argparse import Namespace
from datetime import datetime
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import LightningLoggerBase
from pytorch_lightning.utilities import rank_zero_only
from tap import Tap
from torch.nn.utils.weight_norm import weight_norm
from torch.utils.data import Dataset, DataLoader

import base64
import csv
import numpy as np
import h5py
import json
import os
import pickle
import pytorch_lightning as pl
import random
import sys
import torch
import torch.nn as nn

# == Argument Parser ==

"""
Defines an argument parser with paths to the appropriate arguments -- we use
[Tap](https://github.com/swansonk14/typed-argument-parser) for readability.

Note the arguments `gqa_questions` and `gqa_features`. These are paths to the GQA Questions and extracted 
Bottom-Up Features, and should be pre-downloaded using 
[this script](https://github.com/siddk/annotated-butd/blob/modular/scripts/gqa.sh). 

We also define an argument 
`gqa_cache` that we use to store serialized/formatted data created during the preprocessing step (e.g. HDF5 files).
Feel free to change this to a path that is convenient for you (and has enough storage -- this directory can grow 
large!).

Similarly note the argument `glove` which contains a path to pre-trained 
[GloVe Embeddings](https://nlp.stanford.edu/projects/glove/). These can be downloaded via 
[this script](https://github.com/siddk/annotated-butd/blob/modular/scripts/glove.sh).

Other important arguments include `gpus` (the number of gpus to run with :: default = 0), and the random seed `seed`.

The argument `checkpoint` is a path to a checkpoint directory, to store model metrics and checkpoints 
(saved based on best validation accuracy). Feel free to change this as you see fit.

All other arguments contain sane defaults for initializing different parts of the BUTD model -- these are not optimized,
but seem to work well.
"""
class ArgumentParser(Tap):
    run_name: str                                   # Run Name -- for informative logging

    data: str = "data/"                             # Where downloaded data is located
    checkpoint: str = "checkpoints/"                # Where to save model checkpoints and serialized statistics

    gqa_questions: str = 'data/GQA-Questions'       # Path to GQA Balanced Training Set of Questions
    gqa_features: str = 'data/GQA-Features'         # Path to GQA Features
    gqa_cache: str = 'data/GQA-Cache'               # Path to GQA Cache Directory for storing serialized data

    glove: str = 'data/GloVe/glove.6B.300d.txt'     # Path to GloVe Embeddings File (300-dim)

    gpus: int = 0                                   # Number of GPUs to run with (default :: 0)

    model: str = 'butd'                             # Model Architecture to run with -- < butd | film >
    dataset: str = 'gqa'                            # Dataset to run BUTD Model with -- < vqa2 | gqa | nlvr2 >

    emb_dim: int = 300                              # Word Embedding Dimension --> Should Match GloVe (300)
    emb_dropout: float = 0.0                        # Dropout to Apply to Word Embeddings

    rnn: str = 'GRU'                                # RNN Type for Question Encoder --> one of < 'GRU' | 'LSTM' >
    rnn_layers: int = 1                             # Number of RNN Stacked Layers (for Statement Encoder)
    bidirectional: bool = False                     # Whether or not RNN is Bidirectional
    q_dropout: float = 0.0                          # RNN Dropout (for Question Encoder)

    attention_dropout: float = 0.2                  # Dropout for Attention Operation (fusing Image + Question)

    answer_dropout: float = 0.5                     # Dropout to Apply to Answer Classifier

    hidden: int = 1024                              # Dimensionality of Hidden Layer (Question Encoder & Object Encoder)

    weight_norm: bool = True                        # Boolean whether or not to use Weight Normalization

    bsz: int = 256                                  # Batch Size --> the Bigger the Better
    epochs: int = 15                                # Number of Training Epochs

    opt: str = 'adamax'                             # Optimizer for Performing Gradient Updates
    gradient_clip: float = 0.25                     # Value for Gradient Clipping

    seed: int = 7                                   # Random Seed (for Reproducibility)

# ---

# **Parse arguments** -- Convert to and from Namespace because of weird PyTorch Lightning Bug, and set the name of the Run
# for meaningful logging.

args = Namespace(**ArgumentParser().parse_args().as_dict())
run_name = args.run_name + '-%s' % args.model + '-x%d' % args.seed + '+' + datetime.now().strftime('%m-%d-[%H:%M]')
print('[*] Starting Train Job in Mode %s with Run Name: %s' % (args.dataset.upper(), run_name))

# **Book-Keeping** -- Set the Random Seed for all relevant libraries.
print('[*] Setting Random Seed to %d!' % args.seed)
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)

# ---

# == Preprocessing ==

"""
There are 4 steps to the preprocessing pipeline:

  1. **Preprocessing Question Data**: Involves creating dictionaries of question tokens, for later vectorization.
  2. **Preprocessing Answer Data**: Creating Mappings between string answers and their corresponding indices (for 
    softmax prediction in model's final layer)
  3. **Preprocessing Image Features**: Creating an HDF5 file for easy/efficient access to Bottom-Up Object Features for
    each image, to serve as input to model.
  4. **Dataset Assembly**: Create an official `torch.Dataset` wrapping the VQA Data in an easy-to-batch format.

--- 
"""

# === 1. Preprocessing Question Data ===

"""
*Assemble a Dictionary mapping question tokens to integer indices. Additionally, use the created dictionaries to index 
and load in GloVe vectors.*

---
"""


class Dictionary(object):
    def __init__(self, word2idx=None, idx2word=None):
        if word2idx is None:
            word2idx = {}
        if idx2word is None:
            idx2word = []
        self.word2idx, self.idx2word = word2idx, idx2word

    @property
    def ntoken(self):
        return len(self.word2idx)

    @property
    def padding_idx(self):
        return len(self.word2idx)

    def tokenize(self, sentence, add_word):
        sentence = sentence.lower().replace(',', '').replace('.', '').replace('?', '').replace('\'s', ' \'s')
        words, tokens = sentence.split(), []

        if add_word:
            for w in words:
                tokens.append(self.add_word(w))
        else:
            for w in words:
                tokens.append(self.word2idx.get(w, self.padding_idx))

        return tokens

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)


# Create Dictionary from GQA Question Files, and Initialize GloVe Embeddings from File
def gqa_create_dictionary_glove(gqa_q='data/GQA-Questions', glove='data/GloVe/glove.6B.300d.txt',
                                cache='data/GQA-Cache'):
    """
    ---
    **Note:** It's worth talking about a common design pattern that you'll see throughout this codebase, around utilizing
    the `gqa_cache` directory to its fullest potential.

    As we compute serialized/formatted versions of data (token dictionaries, embedding matrices, HDF5 files), we cache them
    for future runs to speed up the iteration time.

    For a research codebase (where speedy iteration is the name of the game),
    we find this to be a useful practice.
    ---
    """
    dfile, gfile = os.path.join(cache, 'dictionary.pkl'), os.path.join(cache, 'glove.npy')
    if os.path.exists(dfile) and os.path.exists(gfile):
        with open(dfile, 'rb') as f:
            dictionary = pickle.load(f)

        weights = np.load(gfile)
        return dictionary, weights

    elif not os.path.exists(cache):
        os.makedirs(cache)

    dictionary = Dictionary()
    questions = ['train_balanced_questions.json', 'val_balanced_questions.json', 'testdev_balanced_questions.json',
                 'test_balanced_questions.json']

    # Iterate through Question in Question Files and update Vocabulary
    print('\t[*] Creating Dictionary from GQA Questions...')
    for qfile in questions:
        qpath = os.path.join(gqa_q, qfile)
        with open(qpath, 'r') as f:
            examples = json.load(f)

        for ex_key in examples:
            ex = examples[ex_key]
            dictionary.tokenize(ex['question'], add_word=True)

    # Load GloVe Embeddings
    print('\t[*] Loading GloVe Embeddings...')
    with open(glove, 'r') as f:
        entries = f.readlines()

    # Assert that we're using the 300-Dimensional GloVe Embeddings
    assert len(entries[0].split()) - 1 == 300, 'ERROR - Not using 300-dimensional GloVe Embeddings!'

    # Create Embedding Weights
    weights = np.zeros((len(dictionary.idx2word), 300), dtype=np.float32)

    # Populate Embedding Weights
    for entry in entries:
        word_vec = entry.split()
        word, vec = word_vec[0], list(map(float, word_vec[1:]))
        if word in dictionary.word2idx:
            weights[dictionary.word2idx[word]] = vec

    # Dump Dictionary and Weights to file
    with open(dfile, 'wb') as f:
        pickle.dump(dictionary, f)
    np.save(gfile, weights)

    # Return Dictionary and Weights
    return dictionary, weights

# ---

# === 2. Preprocessing Answer Data ===

"""
*Assemble dictionaries mapping answer strings to indices and vice-versa, for priming the Softmax in the final layer of 
the BUTD model.*

---
"""

# Create mapping from answers to labels
def gqa_create_answers(gqa_q='data/GQA-Questions', cache='data/GQA-Cache'):

    # Create File Paths and Load from Disk (if cached)
    dfile = os.path.join(cache, 'answers.pkl')
    if os.path.exists(dfile):
        with open(dfile, 'rb') as f:
            ans2label, label2ans = pickle.load(f)

        return ans2label, label2ans

    ans2label, label2ans = {}, []
    questions = ['train_balanced_questions.json', 'val_balanced_questions.json', 'testdev_balanced_questions.json']

    # Iterate through Answer in Question Files and update Mapping
    print('\t[*] Creating Answer Labels from GQA Question/Answers...')
    for qfile in questions:
        qpath = os.path.join(gqa_q, qfile)
        with open(qpath, 'r') as f:
            examples = json.load(f)

        for ex_key in examples:
            ex = examples[ex_key]
            if not ex['answer'].lower() in ans2label:
                ans2label[ex['answer'].lower()] = len(ans2label)
                label2ans.append(ex['answer'])

    # Dump Dictionaries to File
    with open(dfile, 'wb') as f:
        pickle.dump((ans2label, label2ans), f)

    return ans2label, label2ans


# ---

# === 3. Preprocessing Image Features ===

"""
*Reads in a tsv file with pre-trained bottom up attention features and writes them to hdf5 file. Additionally builds
image ID --> Feature IDX Mapping.*

*Hierarchy of HDF5 file:*

    { 
     'image_features': num_images x num_boxes x 2048 
     'image_spatials': num_images x num_boxes x 6 
     'image_bb': num_images x num_boxes x 4 
    }

---
"""

# Set CSV Field Size Limit (Big TSV Files...)
csv.field_size_limit(sys.maxsize)

FIELDNAMES = ["img_id", "img_h", "img_w", "objects_id", "objects_conf", "attrs_id", "attrs_conf", "num_boxes", "boxes",
              "features"]
NUM_FIXED_BOXES = 36
FEATURE_LENGTH = 2048


def gqa_create_image_features(gqa_f='data/GQA-Features', cache='data/GQA-Cache'):
    """ Iterate through BUTD TSV and Build HDF5 Files with Bounding Box Features, Image ID --> IDX Mappings """
    print('\t[*] Setting up HDF5 Files for Image/Object Features...')

    # Create Trackers for Image IDX --> Index
    trainval_indices, testdev_indices = {}, {}
    tv_file = os.path.join(cache, 'trainval36.hdf5')
    td_file = os.path.join(cache, 'testdev36.hdf5')

    tv_idxfile = os.path.join(cache, 'trainval36_img2idx.pkl')
    td_idxfile = os.path.join(cache, 'testdev36_img2idx.pkl')

    if os.path.exists(tv_file) and os.path.exists(td_file) and os.path.exists(tv_idxfile) and \
            os.path.exists(td_idxfile):

        with open(tv_idxfile, 'rb') as f:
            trainval_indices = pickle.load(f)

        with open(td_idxfile, 'rb') as f:
            testdev_indices = pickle.load(f)

        return trainval_indices, testdev_indices

    with h5py.File(tv_file, 'w') as h_trainval, h5py.File(td_file, 'w') as h_testdev:
        # Get Number of Images in each Split
        with open(os.path.join(gqa_f, 'vg_gqa_obj36.tsv'), 'r') as f:
            ntrainval = len(f.readlines())

        with open(os.path.join(gqa_f, 'gqa_testdev_obj36.tsv'), 'r') as f:
            ntestdev = len(f.readlines())

        # Setup HDF5 Files
        trainval_img_features = h_trainval.create_dataset('image_features', (ntrainval, NUM_FIXED_BOXES,
                                                                             FEATURE_LENGTH), 'f')
        trainval_img_bb = h_trainval.create_dataset('image_bb', (ntrainval, NUM_FIXED_BOXES, 4), 'f')
        trainval_spatial_features = h_trainval.create_dataset('spatial_features', (ntrainval, NUM_FIXED_BOXES, 6), 'f')

        testdev_img_features = h_testdev.create_dataset('image_features', (ntestdev, NUM_FIXED_BOXES, FEATURE_LENGTH),
                                                        'f')
        testdev_img_bb = h_testdev.create_dataset('image_bb', (ntestdev, NUM_FIXED_BOXES, 4), 'f')
        testdev_spatial_features = h_testdev.create_dataset('spatial_features', (ntestdev, NUM_FIXED_BOXES, 6), 'f')

        # Start Iterating through TSV
        print('\t[*] Reading Train-Val TSV File and Populating HDF5 File...')
        trainval_counter, testdev_counter = 0, 0
        with open(os.path.join(gqa_f, 'vg_gqa_obj36.tsv'), 'r') as tsv:
            reader = csv.DictReader(tsv, delimiter='\t', fieldnames=FIELDNAMES)
            for item in reader:
                item['num_boxes'] = int(item['num_boxes'])
                image_id = item['img_id']
                image_w = float(item['img_w'])
                image_h = float(item['img_h'])
                bb = np.frombuffer(base64.b64decode(item['boxes']), dtype=np.float32).reshape((item['num_boxes'], -1))

                box_width = bb[:, 2] - bb[:, 0]
                box_height = bb[:, 3] - bb[:, 1]
                scaled_width = box_width / image_w
                scaled_height = box_height / image_h
                scaled_x = bb[:, 0] / image_w
                scaled_y = bb[:, 1] / image_h

                scaled_width = scaled_width[..., np.newaxis]
                scaled_height = scaled_height[..., np.newaxis]
                scaled_x = scaled_x[..., np.newaxis]
                scaled_y = scaled_y[..., np.newaxis]

                spatial_features = np.concatenate(
                    (scaled_x,
                     scaled_y,
                     scaled_x + scaled_width,
                     scaled_y + scaled_height,
                     scaled_width,
                     scaled_height),
                    axis=1)

                trainval_indices[image_id] = trainval_counter
                trainval_img_bb[trainval_counter, :, :] = bb
                trainval_img_features[trainval_counter, :, :] = \
                    np.frombuffer(base64.b64decode(item['features']), dtype=np.float32).reshape((item['num_boxes'], -1))
                trainval_spatial_features[trainval_counter, :, :] = spatial_features
                trainval_counter += 1

        print('\t[*] Reading Test-Dev TSV File and Populating HDF5 File...')
        with open(os.path.join(gqa_f, 'gqa_testdev_obj36.tsv'), 'r') as tsv:
            reader = csv.DictReader(tsv, delimiter='\t', fieldnames=FIELDNAMES)
            for item in reader:
                item['num_boxes'] = int(item['num_boxes'])
                image_id = item['img_id']
                image_w = float(item['img_w'])
                image_h = float(item['img_h'])
                bb = np.frombuffer(base64.b64decode(item['boxes']), dtype=np.float32).reshape((item['num_boxes'], -1))

                box_width = bb[:, 2] - bb[:, 0]
                box_height = bb[:, 3] - bb[:, 1]
                scaled_width = box_width / image_w
                scaled_height = box_height / image_h
                scaled_x = bb[:, 0] / image_w
                scaled_y = bb[:, 1] / image_h

                scaled_width = scaled_width[..., np.newaxis]
                scaled_height = scaled_height[..., np.newaxis]
                scaled_x = scaled_x[..., np.newaxis]
                scaled_y = scaled_y[..., np.newaxis]

                spatial_features = np.concatenate(
                    (scaled_x,
                     scaled_y,
                     scaled_x + scaled_width,
                     scaled_y + scaled_height,
                     scaled_width,
                     scaled_height),
                    axis=1)

                testdev_indices[image_id] = testdev_counter
                testdev_img_bb[testdev_counter, :, :] = bb
                testdev_img_features[testdev_counter, :, :] = \
                    np.frombuffer(base64.b64decode(item['features']), dtype=np.float32).reshape((item['num_boxes'], -1))
                testdev_spatial_features[testdev_counter, :, :] = spatial_features
                testdev_counter += 1

    # Dump TrainVal and TestDev Indices to File
    with open(tv_idxfile, 'wb') as f:
        pickle.dump(trainval_indices, f)

    with open(td_idxfile, 'wb') as f:
        pickle.dump(testdev_indices, f)

    return trainval_indices, testdev_indices

# ---

# === 4. Dataset Assembly ===

""" 
*Define GQA Feature Dataset `torch.Dataset`, with utilities for loading image features from HDF5 files, and tensorizing
data.*

---
"""

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
                # Note that we pad in front of the sentence (GRU reads left-to-right)
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


def create_entry(example, qid, img2idx, ans2label):
    img_id = example['imageId']
    assert img_id in img2idx, 'Image ID not in Index!'

    entry = {
        'question_id': qid,
        'image_id': img_id,
        'image': img2idx[img_id],
        'question': example['question'],
        'answer': ans2label[example['answer'].lower()]
    }
    return entry

# ---


# == Model Definition ==

"""
In this section, we formally define the Bottom-Up Top-Down Model with product-based multi-modal fusion. 

This model is moderately different than that [originally proposed](https://arxiv.org/abs/1707.07998) and is instead 
inspired by the implementation by [Hengyuan Hu et. al.](https://github.com/hengyuan-hu/bottom-up-attention-vqa) with 
some minor tweaks around the handling of spatial features. 

It's also worth noting that this Model is built using the 
[PyTorch-Lightning](https://github.com/PyTorchLightning/pytorch-lightning) library -- an excellent resource for quickly
prototyping research-based models.

---
"""

# === Sub-Module Definitions ===

# Simple utility class defining a fully connected network (multi-layer perceptron)
class MLP(nn.Module):
    def __init__(self, dims, use_weight_norm=True):
        super(MLP, self).__init__()

        layers = []
        for i in range(len(dims) - 1):
            in_dim, out_dim = dims[i], dims[i + 1]
            if use_weight_norm:
                layers.append(weight_norm(nn.Linear(in_dim, out_dim), dim=None))
            else:
                layers.append(nn.Linear(in_dim, out_dim))
            layers.append(nn.ReLU())

        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        # output: [bsz, \*, dims[0]] --> [bsz, \*, dims[-1]]
        return self.mlp(x)


# Initialize an Embedding Matrix with the appropriate dimensions --> Defines padding as last token in dict
class WordEmbedding(nn.Module):
    def __init__(self, ntoken, dim, dropout=0.0):
        super(WordEmbedding, self).__init__()
        self.ntoken, self.dim = ntoken, dim

        self.emb = nn.Embedding(ntoken + 1, dim, padding_idx=ntoken)
        self.dropout = nn.Dropout(dropout)

    def load_embeddings(self, weights):
        """ Set Embedding Weights from Numpy Array """
        assert weights.shape == (self.ntoken, self.dim)
        self.emb.weight.data[:self.ntoken] = torch.from_numpy(weights)

    def forward(self, x):
        # x : [bsz, seq_len]
        # output: [bsz, seq_len, emb_dim]
        return self.dropout(self.emb(x))


# Initialize the RNN Question Encoder with the appropriate configuration
class QuestionEncoder(nn.Module):
    def __init__(self, in_dim, hidden_dim, nlayers=1, bidirectional=False, dropout=0.0, rnn='GRU'):
        super(QuestionEncoder, self).__init__()
        self.in_dim, self.hidden, self.nlayers, self.bidirectional = in_dim, hidden_dim, nlayers, bidirectional
        self.rnn_type, self.rnn_cls = rnn, nn.GRU if rnn == 'GRU' else nn.LSTM

        # Initialize RNN
        self.rnn = self.rnn_cls(self.in_dim, self.hidden, self.nlayers, bidirectional=self.bidirectional,
                                dropout=dropout, batch_first=True)

    def forward(self, x):
        """
        x: [bsz, seq_len, emb_dim]

        output[0]: [bsz, seq_len, ndirections * hidden]

        output[1]: [bsz, nlayers * ndirections, hidden]
        """
        output, hidden = self.rnn(x)  # Note that Hidden Defaults to 0

        # If not Bidirectional --> Just return last output state
        if not self.bidirectional:
            # output: [bsz, hidden]
            return output[:, -1]

        # Otherwise, concat forward state for last element and backward state for first element
        else:
            # output: [bsz, 2 * hidden]
            f, b = output[:, -1, :self.hidden], output[:, 0, self.hidden:]
            return torch.cat([f, b], dim=1)


# Initialize the Attention Mechanism with the appropriate fusion operation
class Attention(nn.Module):
    def __init__(self, image_dim, question_dim, hidden, dropout=0.2, use_weight_norm=True):
        super(Attention, self).__init__()

        # Attention w/ Product Fusion
        self.image_proj = MLP([image_dim, hidden], use_weight_norm=use_weight_norm)
        self.question_proj = MLP([question_dim, hidden], use_weight_norm=use_weight_norm)
        self.dropout = nn.Dropout(dropout)
        self.linear = weight_norm(nn.Linear(hidden, 1), dim=None) if use_weight_norm else nn.Linear(hidden, 1)

    def forward(self, image_features, question_emb):
        # image_features: [bsz, k, image_dim = 2048]

        # question_emb: [bsz, question_dim]

        # Project both image and question embedding to hidden and repeat question_emb
        num_objs = image_features.size(1)
        image_proj = self.image_proj(image_features)
        question_proj = self.question_proj(question_emb).unsqueeze(1).repeat(1, num_objs, 1)

        # **Key**: Fuse w/ Product
        image_question = image_proj * question_proj

        # Dropout Joint Representation
        joint_representation = self.dropout(image_question)

        # Compute Logits -- Softmax
        logits = self.linear(joint_representation)
        return nn.functional.softmax(logits, dim=1)

# ---


# === Key Model Definition ===
class BUTD(pl.LightningModule):
    def __init__(self, hparams, train_dataset, val_dataset, ans2label=None, label2ans=None):
        super(BUTD, self).__init__()

        # Save Hyper-Parameters and Dataset
        self.hparams = hparams
        self.train_dataset, self.val_dataset = train_dataset, val_dataset
        self.ans2label, self.label2ans = ans2label, label2ans

        # Build Model
        self.build_model()

    def build_model(self):
        # Build Word Embeddings (for Questions)
        self.w_emb = WordEmbedding(ntoken=self.train_dataset.dictionary.ntoken, dim=self.hparams.emb_dim,
                                   dropout=self.hparams.emb_dropout)

        # Build Question Encoder
        self.q_enc = QuestionEncoder(in_dim=self.hparams.emb_dim, hidden_dim=self.hparams.hidden,
                                     nlayers=self.hparams.rnn_layers, bidirectional=self.hparams.bidirectional,
                                     dropout=self.hparams.q_dropout, rnn=self.hparams.rnn)

        # Build Attention Mechanism
        self.att = Attention(image_dim=self.train_dataset.v_dim + 6, question_dim=self.q_enc.hidden,
                             hidden=self.hparams.hidden, dropout=self.hparams.attention_dropout,
                             use_weight_norm=self.hparams.weight_norm)

        # Build Projection Networks
        self.q_project = MLP([self.q_enc.hidden, self.hparams.hidden], use_weight_norm=self.hparams.weight_norm)
        self.img_project = MLP([self.train_dataset.v_dim + 6, self.hparams.hidden],
                               use_weight_norm=self.hparams.weight_norm)

        # Build Answer Classifier
        self.ans_classifier = nn.Sequential(*[
            weight_norm(nn.Linear(self.hparams.hidden, 2 * self.hparams.hidden), dim=None)
            if self.hparams.weight_norm else nn.Linear(self.hparams.hidden, 2 * self.hparams.hidden),

            nn.ReLU(),
            nn.Dropout(self.hparams.answer_dropout),

            weight_norm(nn.Linear(2 * self.hparams.hidden, len(self.ans2label)), dim=None)
            if self.hparams.weight_norm else nn.Linear(2 * self.hparams.hidden, len(self.ans2label))
        ])

    def forward(self, image_features, spatial_features, question_features, indicator_features=None):
        # image_features: [bsz, K, image_dim]
        # question_features: [bsz, seq_len]

        # Embed and Encode Question -- [bsz, q_hidden]
        w_emb = self.w_emb(question_features)
        q_enc = self.q_enc(w_emb)

        # Create new Image Features

        # **Key**: Concatenate Spatial Features!
        if indicator_features is not None:
            image_features = torch.cat([image_features, spatial_features, indicator_features], dim=2)
        else:
            image_features = torch.cat([image_features, spatial_features], dim=2)

        # Attend over Image Features and Create Image Encoding

        # img_enc: [bsz, img_hidden]
        att = self.att(image_features, q_enc)
        img_enc = (image_features * att).sum(dim=1)

        # Project Image and Question Features --> [bsz, hidden]
        q_repr = self.q_project(q_enc)
        img_repr = self.img_project(img_enc)

        # Merge
        joint_repr = q_repr * img_repr

        # Compute and Return Logits
        return self.ans_classifier(joint_repr)

    def configure_optimizers(self):
        if self.hparams.opt == 'adamax':
            return torch.optim.Adamax(self.parameters())

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.hparams.bsz, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.hparams.bsz)

    def training_step(self, train_batch, batch_idx):
        img, spatials, question, answer = train_batch

        # Run Forward Pass
        logits = self.forward(img, spatials, question)

        # Compute Loss (Cross-Entropy)
        loss = nn.functional.cross_entropy(logits, answer)

        # Compute Answer Accuracy
        accuracy = torch.mean((logits.argmax(dim=1) == answer).float())

        # Set up Data to be Logged
        log = {'train_loss': loss, 'train_acc': accuracy}

        return {'loss': loss, 'train_loss': loss, 'train_acc': accuracy, 'progress_bar': log, 'log': log}

    def training_epoch_end(self, outputs):
        # Outputs --> List of Individual Step Outputs
        avg_loss = torch.stack([x['callback_metrics']['train_loss'] for x in outputs]).mean()
        avg_acc = torch.stack([x['callback_metrics']['train_acc'] for x in outputs]).mean()

        log = {'train_epoch_loss': avg_loss, 'train_epoch_acc': avg_acc}

        return {'progress_bar': log, 'log': log}

    def validation_step(self, val_batch, batch_idx):
        img, spatials, question, answer = val_batch

        # Run Forward Pass
        logits = self.forward(img, spatials, question)

        # Compute Loss (Cross-Entropy)
        loss = nn.functional.cross_entropy(logits, answer)

        # Compute Answer Accuracy
        accuracy = torch.mean((logits.argmax(dim=1) == answer).float())

        return {'val_loss': loss, 'val_acc': accuracy}

    def validation_epoch_end(self, outputs):
        # Outputs --> List of Individual Step Outputs
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        avg_acc = torch.stack([x['val_acc'] for x in outputs]).mean()

        log = {'val_loss': avg_loss, 'val_acc': avg_acc}
        return {'progress_bar': log, 'log': log}

# ---

# == Logging ==

"""
*We tap into PyTorch-Lightning's extensive Logging Capabilities and define our own simple logger to log metrics like
training loss, training accuracy, validation loss, and validation accuracy to straightforward JSON files.*

---
"""

class MetricLogger(LightningLoggerBase):
    def __init__(self, name, save_dir):
        super(MetricLogger, self).__init__()
        self._name, self._save_dir = name, os.path.join(save_dir, 'metrics')

        # Create Massive Dictionary to JSONify
        self.events = {}

    @property
    def name(self):
        return self._name

    @property
    def experiment(self):
        return None

    @property
    def version(self):
        return 1.0

    @rank_zero_only
    def log_hyperparams(self, params):
        # Params is an argparse.Namespace
        self.events['hyperparams'] = vars(params)

    @rank_zero_only
    def log_metrics(self, metrics, step):
        # Metrics is a dictionary of metric names and values
        for metric in metrics:
            if metric in self.events:
                self.events[metric].append(metrics[metric])
                self.events["%s_step" % metric].append(step)
            else:
                self.events[metric] = [metrics[metric]]
                self.events["%s_step" % metric] = [step]

    @rank_zero_only
    def finalize(self, status):
        # Optional. Any code that needs to be run after training
        self.events['status'] = status

        if not os.path.exists(self._save_dir):
            os.makedirs(self._save_dir)

        with open(os.path.join(self._save_dir, '%s-metrics.json' % self._name), 'w') as f:
            json.dump(self.events, f, indent=4)

# ---

# == Bringing the Pieces Together ==

"""
Here, we bring all the pieces together, calling each of the 4 preprocessing steps, assembling the training and
development datasets, and initializing and training the BUTD model. 
---
"""

# Preprocess Question Data -- Return Dictionary and GloVe-initialized Embeddings
print('\n[*] Pre-processing GQA Questions...')
dictionary, emb = gqa_create_dictionary_glove(gqa_q=args.gqa_questions, glove=args.glove, cache=args.gqa_cache)

# Preprocess Answer Data
print('\n[*] Pre-processing GQA Answers...')
ans2label, label2ans = gqa_create_answers(gqa_q=args.gqa_questions, cache=args.gqa_cache)

# Create Image Features
print('\n[*] Pre-processing GQA BUTD Image Features')
trainval_img2idx, testdev_img2idx = gqa_create_image_features(gqa_f=args.gqa_features, cache=args.gqa_cache)

# Build Train and TestDev Datasets -- Note here that we use the TestDev split of GQA instead of Val (as is common
# practice) because of Visual Genome data leakage in the Validation Set
print('\n[*] Building GQA Train and TestDev Datasets...')
train_dataset = GQAFeatureDataset(dictionary, ans2label, label2ans, trainval_img2idx, gqa_q=args.gqa_questions,
                                  cache=args.gqa_cache, mode='train')

dev_dataset = GQAFeatureDataset(dictionary, ans2label, label2ans, testdev_img2idx, gqa_q=args.gqa_questions,
                                cache=args.gqa_cache, mode='testdev')

# Create BUTD Module (and load Embeddings!)
print('\n[*] Initializing Bottom-Up Top-Down Model...')
nn = BUTD(args, train_dataset, dev_dataset, ans2label, label2ans)
nn.w_emb.load_embeddings(emb)


# Setup Logger for PyTorch-Lightning
mt_logger = MetricLogger(name=run_name, save_dir=args.checkpoint)

# Saves the top-3 Checkpoints based on Validation Accuracy -- feel free to change this metric to suit your needs
checkpoint_callback = ModelCheckpoint(filepath=os.path.join(args.checkpoint, 'runs', run_name,
                                                            'butd-{epoch:02d}-{val_loss:.2f}-{val_acc:.2f}'),
                                      monitor='val_acc', mode='max', save_top_k=3)

# Create Pytorch-Lightning Trainer -- run for the given number of epochs, with gradient clipping!
trainer = pl.Trainer(default_root_dir=args.checkpoint, max_epochs=args.epochs, gradient_clip_val=args.gradient_clip,
                     gpus=args.gpus, benchmark=True, logger=mt_logger, checkpoint_callback=checkpoint_callback)

# Fit and Profit!
print('\n[*] Training...\n')
trainer.fit(nn)
