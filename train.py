"""
train.py

Core script for training Bottom-Up Top-Down Attention Models on various VQA datasets, as well as saving model
checkpoints and logging training statistics.
"""
from argparse import Namespace
from datetime import datetime
from pytorch_lightning.callbacks import ModelCheckpoint
from tap import Tap

from src.logging import MetricLogger
from src.preprocessing.gqa import gqa_create_dictionary_glove, gqa_create_answers, gqa_create_image_features, \
    GQAFeatureDataset
from src.preprocessing.nlvr2 import nlvr2_create_dictionary_glove, nlvr2_create_image_features, \
    NLVR2FeatureDataset
from src.preprocessing.vqa2 import vqa2_create_dictionary_glove, vqa2_create_soft_answers, vqa2_create_image_features, \
    VQAFeatureDataset
from src.models import BUTD, FiLM

import numpy as np
import os
import pytorch_lightning as pl
import random
import torch


class ArgumentParser(Tap):
    # Run Name
    run_name: str                                   # Run Name -- for informative logging

    # Data and Checkpoint Parameters
    data: str = "data/"                             # Where downloaded data is located
    checkpoint: str = "checkpoints/"                # Where to save model checkpoints and serialized statistics

    # GQA-Specific Parameters
    gqa_questions: str = 'data/GQA-Questions'       # Path to GQA Balanced Training Set of Questions
    gqa_features: str = 'data/GQA-Features'         # Path to GQA Features
    gqa_cache: str = 'data/GQA-Cache'               # Path to GQA Cache Directory for storing serialized data

    # NLVR2-Specific Parameters
    nlvr2_questions: str = 'data/NLVR2-Questions'   # Path to NLVR-2 Questions
    nlvr2_features: str = 'data/NLVR2-Features'     # Path to NLVR-2 Features
    nlvr2_cache: str = 'data/NLVR2-Cache'           # Path to NLVR-2 Cache Directory for storing serialized data

    # VQA2-Specific Parameters
    vqa2_questions: str = 'data/VQA2-Questions'     # Path to VQA-2 Questions
    vqa2_features: str = 'data/VQA2-Features'       # Path to VQA-2 Features
    vqa2_cache: str = 'data/VQA2-Cache'             # Path to VQA-2 Cache Directory for storing serialized data

    # GloVe Vectors
    glove: str = 'data/GloVe/glove.6B.300d.txt'     # Path to GloVe Embeddings File (300-dim)

    # GPUs
    gpus: int = 0                                   # Number of GPUs to run with (default :: 0)

    # Mode
    model: str = 'butd'                             # Model Architecture to run with -- < butd | film >
    dataset: str = 'gqa'                            # Dataset to run BUTD Model with -- < vqa2 | gqa | nlvr2 >

    # Bottom-Up Top-Down (BUTD) Model Parameters
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

    # Training Parameters
    bsz: int = 256                                  # Batch Size --> the Bigger the Better
    epochs: int = 15                                # Number of Training Epochs

    opt: str = 'adamax'                             # Optimizer for Performing Gradient Updates
    gradient_clip: float = 0.25                     # Value for Gradient Clipping

    # Random Seed
    seed: int = 7                                   # Random Seed (for Reproducibility)


def train():
    # Parse Arguments --> Convert to and from Namespace because of weird PyTorch Lightning Bug
    args = Namespace(**ArgumentParser().parse_args().as_dict())

    # Set Run Name
    run_name = args.run_name + '-%s' % args.model + '-x%d' % args.seed + '+' + datetime.now().strftime('%m-%d-[%H:%M]')
    print('[*] Starting Train Job in Mode %s with Run Name: %s' % (args.dataset.upper(), run_name))

    # Set Randomness
    print('[*] Setting Random Seed to %d!' % args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Setup Logger for PyTorch-Lightning
    mt_logger = MetricLogger(name=run_name, save_dir=args.checkpoint)

    # Dataset-Specific Pre-Processing
    if args.dataset == 'gqa':
        # Preprocess Question Data --> Return Dictionary and GloVe-initialized Embeddings
        print('\n[*] Pre-processing GQA Questions...')
        dictionary, emb = gqa_create_dictionary_glove(gqa_q=args.gqa_questions, glove=args.glove, cache=args.gqa_cache)

        # Preprocess Answer Data
        print('\n[*] Pre-processing GQA Answers...')
        ans2label, label2ans = gqa_create_answers(gqa_q=args.gqa_questions, cache=args.gqa_cache)

        # Create Image Features
        print('\n[*] Pre-processing GQA BUTD Image Features')
        trainval_img2idx, testdev_img2idx = gqa_create_image_features(gqa_f=args.gqa_features, cache=args.gqa_cache)

        # Build Train and TestDev Datasets --> TestDev instead of Val because of Visual Genome leakage in Val
        print('\n[*] Building GQA Train and TestDev Datasets...')
        train_dataset = GQAFeatureDataset(dictionary, ans2label, label2ans, trainval_img2idx, gqa_q=args.gqa_questions,
                                          cache=args.gqa_cache, mode='train')

        dev_dataset = testdev_dataset = GQAFeatureDataset(dictionary, ans2label, label2ans, testdev_img2idx,
                                                          gqa_q=args.gqa_questions, cache=args.gqa_cache,
                                                          mode='testdev')

    elif args.dataset == 'nlvr2':
        # Preprocess Question Data --> Return Dictionary and GloVe-initialized Embeddings
        print('\n[*] Pre-processing NLVR-2 Questions...')
        dictionary, emb = nlvr2_create_dictionary_glove(nlvr2_q=args.nlvr2_questions, glove=args.glove,
                                                        cache=args.nlvr2_cache)

        # No Answer Data Pre-Processing (True/False Questions)
        ans2label, label2ans = None, None

        # Create Image Features
        print('\n[*] Pre-processing NLVR-2 BUTD Image Features...')
        train_img2idx, dev_img2idx, test_img2idx = nlvr2_create_image_features(nlvr2_f=args.nlvr2_features,
                                                                               cache=args.nlvr2_cache)

        # Build Train and Dev Datasets (no Test for good Experimental Practice)
        print('\n[*] Building NLVR-2 Train and Dev Datasets...')
        train_dataset = NLVR2FeatureDataset(dictionary, train_img2idx, nlvr2_q=args.nlvr2_questions,
                                            cache=args.nlvr2_cache, mode='train')

        dev_dataset = NLVR2FeatureDataset(dictionary, dev_img2idx, nlvr2_q=args.nlvr2_questions, mode='dev')

    elif args.dataset == 'vqa2':
        # Preprocess Question Data --> Return Dictionary and GloVe-initialized Embeddings
        print('\n[*] Pre-processing VQA-2 Questions...')
        dictionary, emb = vqa2_create_dictionary_glove(vqa2_q=args.vqa2_questions, glove=args.glove,
                                                       cache=args.vqa2_cache)

        # Preprocess Answer Data --> Filter Answers based-on Occurrences, Compute Soft Labels
        print('\n[*] Pre-processing VQA-2 Answers...')
        ans2label, label2ans, train_targets, val_targets = vqa2_create_soft_answers(vqa2_q=args.vqa2_questions,
                                                                                    cache=args.vqa2_cache)

        # Create Image Features
        print('\n[*] Pre-processing VQA-2 BUTD Image Features...')
        train_img2idx, val_img2idx = vqa2_create_image_features(vqa2_f=args.vqa2_features, cache=args.vqa2_cache)

        # Build Train and Val/Dev Datasets (no Test for good Experimental Practice)
        print('\n[*] Building VQA-2 Train and Validation Datasets...')
        train_dataset = VQAFeatureDataset(dictionary, ans2label, label2ans, train_img2idx, train_targets,
                                          vqa2_q=args.vqa2_questions, cache=args.vqa2_cache, mode='train')

        dev_dataset = val_dataset = VQAFeatureDataset(dictionary, ans2label, label2ans, val_img2idx, val_targets,
                                                      vqa2_q=args.vqa2_questions, cache=args.vqa2_cache, mode='val')

    # Create BUTD Module (and load Embeddings!)
    if args.model == 'butd':
        print('\n[*] Initializing Bottom-Up Top-Down Model...')
        nn = BUTD(args, train_dataset, dev_dataset, ans2label, label2ans)
        nn.w_emb.load_embeddings(emb)

    elif args.model == 'film':
        print('\n[*] Initializing Bottom-Up Top-Down FiLM Model...')
        nn = FiLM(args, train_dataset, dev_dataset, ans2label, label2ans)
        nn.w_emb.load_embeddings(emb)

    # Create Trainer
    print('\n[*] Training...\n')
    checkpoint_callback = ModelCheckpoint(filepath=os.path.join(args.checkpoint, 'runs', run_name,
                                                                'butd-{epoch:02d}-{val_loss:.2f}-{val_acc:.2f}'),
                                          monitor='val_acc', mode='max', save_top_k=3)

    trainer = pl.Trainer(default_root_dir=args.checkpoint, max_epochs=args.epochs, gradient_clip_val=args.gradient_clip,
                         gpus=args.gpus, benchmark=True, logger=mt_logger, checkpoint_callback=checkpoint_callback)

    # Fit
    trainer.fit(nn)


if __name__ == "__main__":
    train()
