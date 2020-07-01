"""
film.py

Implementation of the FiLM Model, as applied to various VQA Tasks.
"""
from torch.nn.utils.weight_norm import weight_norm
from torch.utils.data import DataLoader

import pytorch_lightning as pl
import torch.nn as nn
import torch


class MLP(nn.Module):
    def __init__(self, dims, use_weight_norm=True):
        """ Simple utility class defining a fully connected network (multi-layer perceptron) """
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
        # [bsz, *, dims[0]] --> [bsz, *, dims[-1]]
        return self.mlp(x)


class WordEmbedding(nn.Module):
    def __init__(self, ntoken, dim, dropout=0.0):
        """ Initialize an Embedding Matrix with the appropriate dimensions --> Defines padding as last token in dict """
        super(WordEmbedding, self).__init__()
        self.ntoken, self.dim = ntoken, dim

        self.emb = nn.Embedding(ntoken + 1, dim, padding_idx=ntoken)
        self.dropout = nn.Dropout(dropout)

    def load_embeddings(self, weights):
        """ Set Embedding Weights from Numpy Array """
        assert weights.shape == (self.ntoken, self.dim)
        self.emb.weight.data[:self.ntoken] = torch.from_numpy(weights)

    def forward(self, x):
        # x : [bsz, seq_len] --> [bsz, seq_len, emb_dim]
        return self.dropout(self.emb(x))


class QuestionEncoder(nn.Module):
    def __init__(self, in_dim, hidden_dim, nlayers=1, bidirectional=False, dropout=0.0, rnn='GRU'):
        """ Initialize the RNN Question Encoder with the appropriate configuration """
        super(QuestionEncoder, self).__init__()
        self.in_dim, self.hidden, self.nlayers, self.bidirectional = in_dim, hidden_dim, nlayers, bidirectional
        self.rnn_type, self.rnn_cls = rnn, nn.GRU if rnn == 'GRU' else nn.LSTM

        # Initialize RNN
        self.rnn = self.rnn_cls(self.in_dim, self.hidden, self.nlayers, bidirectional=self.bidirectional,
                                dropout=dropout, batch_first=True)

    def forward(self, x):
        # x: [bsz, seq_len, emb_dim] --> ([bsz, seq_len, ndirections * hidden], [bsz, nlayers * ndirections, hidden])
        output, hidden = self.rnn(x)  # Note that Hidden Defaults to 0

        # If not Bidirectional --> Just Return last Output State
        if not self.bidirectional:
            # [bsz, hidden]
            return output[:, -1]

        # Otherwise, concat forward state for last element and backward state for first element
        else:
            # [bsz, 2 * hidden]
            f, b = output[:, -1, :self.hidden], output[:, 0, self.hidden:]
            return torch.cat([f, b], dim=1)


class FiLM(pl.LightningModule):
    def __init__(self, hparams, train_dataset, val_dataset, ans2label=None, label2ans=None):
        super(FiLM, self).__init__()

        # Save Hyper-Parameters and Dataset
        self.hparams = hparams
        self.train_dataset, self.val_dataset = train_dataset, val_dataset
        self.ans2label, self.label2ans = ans2label, label2ans

        # Build Model
        self.build_model()

    def build_model(self):
        # GQA Parameters
        if self.hparams.dataset == 'gqa':
            # Build Word Embeddings (for Questions)
            self.w_emb = WordEmbedding(ntoken=self.train_dataset.dictionary.ntoken, dim=self.hparams.emb_dim,
                                       dropout=self.hparams.emb_dropout)

            # Build Question Encoder
            self.q_enc = QuestionEncoder(in_dim=self.hparams.emb_dim, hidden_dim=self.hparams.hidden,
                                         nlayers=self.hparams.rnn_layers, bidirectional=self.hparams.bidirectional,
                                         dropout=self.hparams.q_dropout, rnn=self.hparams.rnn)

            # Get FiLM Parameters
            self.film_gamma = nn.Linear(self.hparams.hidden, self.train_dataset.v_dim + 6)
            self.film_beta = nn.Linear(self.hparams.hidden, self.train_dataset.v_dim + 6)

            # Build Projection Networks
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

        elif self.hparams.dataset == 'nlvr2':
            # Build Word Embeddings (for Questions)
            self.w_emb = WordEmbedding(ntoken=self.train_dataset.dictionary.ntoken, dim=self.hparams.emb_dim,
                                       dropout=self.hparams.emb_dropout)

            # Build Question Encoder
            self.q_enc = QuestionEncoder(in_dim=self.hparams.emb_dim, hidden_dim=self.hparams.hidden,
                                         nlayers=self.hparams.rnn_layers, bidirectional=self.hparams.bidirectional,
                                         dropout=self.hparams.q_dropout, rnn=self.hparams.rnn)

            # Get FiLM Parameters
            self.film_gamma = nn.Linear(self.hparams.hidden, self.train_dataset.v_dim + 6 + 1)
            self.film_beta = nn.Linear(self.hparams.hidden, self.train_dataset.v_dim + 6 + 1)

            # Build Projection Networks
            self.img_project = MLP([self.train_dataset.v_dim + 6 + 1, self.hparams.hidden],
                                   use_weight_norm=self.hparams.weight_norm)

            # Build Answer Classifier
            self.ans_classifier = nn.Sequential(*[
                weight_norm(nn.Linear(self.hparams.hidden, 2 * self.hparams.hidden), dim=None)
                if self.hparams.weight_norm else nn.Linear(self.hparams.hidden, 2 * self.hparams.hidden),

                nn.ReLU(),
                nn.Dropout(self.hparams.answer_dropout),

                weight_norm(nn.Linear(2 * self.hparams.hidden, 2), dim=None)
                if self.hparams.weight_norm else nn.Linear(2 * self.hparams.hidden, 2)
            ])

        elif self.hparams.dataset == 'vqa2':
            # Build Word Embeddings (for Questions)
            self.w_emb = WordEmbedding(ntoken=self.train_dataset.dictionary.ntoken, dim=self.hparams.emb_dim,
                                       dropout=self.hparams.emb_dropout)

            # Build Question Encoder
            self.q_enc = QuestionEncoder(in_dim=self.hparams.emb_dim, hidden_dim=self.hparams.hidden,
                                         nlayers=self.hparams.rnn_layers, bidirectional=self.hparams.bidirectional,
                                         dropout=self.hparams.q_dropout, rnn=self.hparams.rnn)

            # Get FiLM Parameters
            self.film_gamma = nn.Linear(self.hparams.hidden, self.train_dataset.v_dim + 6)
            self.film_beta = nn.Linear(self.hparams.hidden, self.train_dataset.v_dim + 6)

            # Build Projection Networks
            self.img_project = MLP([self.train_dataset.v_dim + 6, self.hparams.hidden],
                                   use_weight_norm=self.hparams.weight_norm)

            # Build Answer Classifier
            self.ans_classifier = nn.Sequential(*[
                weight_norm(nn.Linear(self.hparams.hidden, 2 * self.hparams.hidden), dim=None)
                if self.hparams.weight_norm else nn.Linear(self.hparams.hidden, 2 * self.hparams.hidden),

                nn.ReLU(),
                nn.Dropout(self.hparams.answer_dropout),

                weight_norm(nn.Linear(2 * self.hparams.hidden, self.train_dataset.num_ans_candidates), dim=None)
                if self.hparams.weight_norm else nn.Linear(2 * self.hparams.hidden,
                                                           self.train_dataset.num_ans_candidates)
            ])

    def forward(self, image_features, spatial_features, question_features, indicator_features=None):
        # image_features: [bsz, K, image_dim]
        # question_features: [bsz, seq_len]

        # Embed and Encode Question --> [bsz, q_hidden]
        w_emb = self.w_emb(question_features)
        q_enc = self.q_enc(w_emb)

        # Create new Image Features
        if indicator_features is not None:
            image_features = torch.cat([image_features, spatial_features, indicator_features], dim=2)
        else:
            image_features = torch.cat([image_features, spatial_features], dim=2)

        # Generate FiLM Parameters
        gamma = self.film_gamma(q_enc)
        beta = self.film_beta(q_enc)

        # Unsqueeze FiLM Parameters
        num_objs = image_features.size(1)
        gamma_proj = gamma.unsqueeze(1).repeat(1, num_objs, 1)
        beta_proj = beta.unsqueeze(1).repeat(1, num_objs, 1)

        # FiLMed Representation
        img_enc = ((image_features * gamma_proj) + beta_proj).mean(dim=1)

        # Project up and Classify
        img_repr = self.img_project(img_enc)

        # Compute and Return Logits
        return self.ans_classifier(img_repr)

    def configure_optimizers(self):
        if self.hparams.opt == 'adamax':
            return torch.optim.Adamax(self.parameters())

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.hparams.bsz, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.hparams.bsz)

    @staticmethod
    def per_example_bce(logits, labels):
        loss = nn.functional.binary_cross_entropy_with_logits(logits, labels)
        loss *= labels.size(1)  # Don't take Mean across final dimension (because multiple answers)
        return loss

    @staticmethod
    def compute_accuracy_with_logits(logits, labels):
        logits = torch.max(logits, 1)[1].data
        one_hots = torch.zeros(*labels.size()).type_as(labels)
        one_hots.scatter_(1, logits.view(-1, 1), 1)
        return one_hots * labels

    def training_step(self, train_batch, batch_idx):
        if self.hparams.dataset == 'gqa':
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

        elif self.hparams.dataset == 'nlvr2':
            img, spatials, indicator, question, answer = train_batch

            # Run Forward Pass
            logits = self.forward(img, spatials, question, indicator_features=indicator)

            # Compute Loss (Cross-Entropy)
            loss = nn.functional.cross_entropy(logits, answer)

            # Compute Answer Accuracy
            accuracy = torch.mean((logits.argmax(dim=1) == answer).float())

            # Set up Data to be Logged
            log = {'train_loss': loss, 'train_acc': accuracy}

            return {'loss': loss, 'train_loss': loss, 'train_acc': accuracy, 'progress_bar': log, 'log': log}

        elif self.hparams.dataset == 'vqa2':
            img, spatials, question, answer = train_batch

            # Run Forward Pass
            logits = self.forward(img, spatials, question)

            # Compute Loss (Per-Example Binary Cross-Entropy)
            loss = self.per_example_bce(logits, answer)

            # Compute Answer Accuracy
            accuracy = self.compute_accuracy_with_logits(logits, answer).sum() / self.hparams.bsz

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
        if self.hparams.dataset == 'gqa':
            img, spatials, question, answer = val_batch

            # Run Forward Pass
            logits = self.forward(img, spatials, question)

            # Compute Loss (Cross-Entropy)
            loss = nn.functional.cross_entropy(logits, answer)

            # Compute Answer Accuracy
            accuracy = torch.mean((logits.argmax(dim=1) == answer).float())

            return {'val_loss': loss, 'val_acc': accuracy}

        elif self.hparams.dataset == 'nlvr2':
            img, spatials, indicator, question, answer = val_batch

            # Run Forward Pass
            logits = self.forward(img, spatials, question, indicator_features=indicator)

            # Compute Loss (Cross-Entropy)
            loss = nn.functional.cross_entropy(logits, answer)

            # Compute Answer Accuracy
            accuracy = torch.mean((logits.argmax(dim=1) == answer).float())

            return {'val_loss': loss, 'val_acc': accuracy}

        elif self.hparams.dataset == 'vqa2':
            img, spatials, question, answer = val_batch

            # Run Forward Pass
            logits = self.forward(img, spatials, question)

            # Compute Loss (Per-Example Binary Cross-Entropy)
            loss = self.per_example_bce(logits, answer)

            # Compute Answer Accuracy
            accuracy = self.compute_accuracy_with_logits(logits, answer).sum() / self.hparams.bsz

            return {'val_loss': loss, 'val_acc': accuracy}

    def validation_epoch_end(self, outputs):
        # Outputs --> List of Individual Step Outputs
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        avg_acc = torch.stack([x['val_acc'] for x in outputs]).mean()

        log = {'val_loss': avg_loss, 'val_acc': avg_acc}
        return {'progress_bar': log, 'log': log}