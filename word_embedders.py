import string
from typing import List
from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F
from allennlp.modules.elmo import batch_to_ids, Elmo
from transformers import AutoModel, AutoTokenizer
from torchtext import vocab

from utils import FFNNScore, Span


class BaseEmbedder(ABC):
    @abstractmethod
    def tokenize_train_data(self, doc):
        pass

    @abstractmethod
    def tokenize_test_data(self, doc):
        pass



class Bert(nn.Module, BaseEmbedder):
    def __init__(self, model_path, chunk_len=128, stride_for_overlap=0, word_embed_dim=768):
        super().__init__()

        self.model = AutoModel.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

        self.chunk_len = chunk_len
        self.stride = stride_for_overlap
        self.word_embed_dim = word_embed_dim

    def tokenize_train_data(self, doc):
        doc = doc['sentences']
        # Document tokenization
        doc = self.join_sents(doc)
        tokenized = self.tokenizer( doc['tokens'],
                                    is_split_into_words=True,
                              )

        ids = tokenized.word_ids()
        word_inds = [[] for _ in range(len(doc['tokens']))]
        for i, w_ind in enumerate(ids):
            if w_ind is not None:
                word_inds[w_ind].append(i)

        # Creating answers
        cspans = {}
        for span in doc['coref_spans']:
            cspans[span[0]] = cspans.get(span[0], []) + \
                              [Span(word_inds[span[1]][0], word_inds[span[2]][-1])]

        gold_antecedents = {}
        for cluster, spans in cspans.items():
            for i in range(len(spans)-1,0, -1):
                gold_antecedents[spans[i]] = spans[:i]

        # Split whole document on parts with len = self.chunk_len
        inputs = self.split_doc(tokenized)

        return inputs, gold_antecedents,doc['tokens']

    def tokenize_test_data(self, doc):
        doc = doc['sentences']
        # Document tokenization
        doc = self.join_sents(doc)
        tokenized = self.tokenizer( doc['tokens'],
                                    is_split_into_words=True,
                              )

        ids = tokenized.word_ids()
        word_inds = [[] for _ in range(len(doc['tokens']))]
        for i, w_ind in enumerate(ids):
            if w_ind is not None:
                word_inds[w_ind].append(i)

        # Creating answers
        cspans = {}
        for span in doc['coref_spans']:
            cspans[span[0]] = cspans.get(span[0], []) + \
                              [(word_inds[span[1]][0], word_inds[span[2]][-1])]

        mention_to_cluster = {}
        clusters = []
        for cluster_id, spans in cspans.items():
            spans = tuple(spans)
            mention_to_cluster.update({span: spans for span in spans})
            clusters.append(spans)

        # Split whole document on parts with len = self.chunk_len
        inputs = self.split_doc(tokenized)

        return inputs, clusters, mention_to_cluster, doc['tokens']

    def split_doc(self, doc):
        window_len = self.chunk_len - 2
        start_ind = 1
        end_ind = window_len + 1
        inputs = {key: [] for key in doc.keys()}
        while start_ind < len(doc['input_ids']):
            for key in inputs.keys():
                part_list = [doc[key][0]] + \
                            doc[key][start_ind:end_ind] + \
                            [doc[key][-1]]
                inputs[key].append(part_list)
            start_ind += window_len
            end_ind += window_len
        # Padding last segment to self.chunk_len
        if len(inputs['input_ids'][-1]) < self.chunk_len:
            pad_len = (self.chunk_len - len(inputs['input_ids'][-1]))
            inputs['input_ids'][-1] = inputs['input_ids'][-1] + \
                                      [self.tokenizer.convert_tokens_to_ids(self.tokenizer.pad_token)] * pad_len
            inputs['token_type_ids'][-1] = inputs['token_type_ids'][-1] + [0] * pad_len
            inputs['attention_mask'][-1] = inputs['attention_mask'][-1] + [0] * pad_len

        for key in inputs.keys():
            inputs[key] = torch.tensor(inputs[key])

        return inputs

    def join_sents(self, sents: List[str]):
        """
        Joining sentences and updating coref_spans for new document
        :param sents:
        :return: dict
        """
        tokens = []
        coref_spans = []
        n = 0
        for sent in sents:
            tokens.extend(sent['words'])
            for span in sent['coref_spans']:
                coref_spans.append((span[0], span[1] + n, span[2] + n))
            n += len(sent['words'])
        return {'tokens': tokens,
                'coref_spans': coref_spans}

    def forward(self, input_ids, token_type_ids, attention_mask):
        output = self.model(input_ids=input_ids,
                            token_type_ids=token_type_ids,
                            attention_mask=attention_mask)
        return output['last_hidden_state']



class ELMO(nn.Module, BaseEmbedder):
    def __init__(self,
                 model_size='small',
                 num_output_representations=1,
                 dropout=0.5,
                 requires_grad=False,
                 device=None):
        super().__init__()
        if model_size == 'small':
            self.word_embed_dim = 256
            self.options_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x1024_128_2048cnn_1xhighway/elmo_2x1024_128_2048cnn_1xhighway_options.json"
            self.weight_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x1024_128_2048cnn_1xhighway/elmo_2x1024_128_2048cnn_1xhighway_weights.hdf5"
        elif model_size == 'medium':
            self.word_embed_dim = 512
            self.options_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x2048_256_2048cnn_1xhighway/elmo_2x2048_256_2048cnn_1xhighway_options.json"
            self.weight_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x2048_256_2048cnn_1xhighway/elmo_2x2048_256_2048cnn_1xhighway_weights.hdf5"
        elif model_size == 'original':
            self.word_embed_dim = 1024
            self.options_file = 'https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json'
            self.weight_file = 'https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5'
        else:
            raise ValueError("Correct values for model are 'small', 'medium', 'original'")
        self.word_emb = Elmo(self.options_file, self.weight_file,
                             num_output_representations=num_output_representations,
                             dropout=dropout,
                             requires_grad=requires_grad)
        self.device = device
        if not self.device:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def tokenize_train_data(self, doc):
        ids = []
        n = 0
        span_indices = []
        sentences = doc['sentences']
        # # посмотреть формат ids
        for s in sentences:
            ids.append(batch_to_ids([s['words']]))
            # embed.append(self.word_emb(sids)['elmo_representations'][0])
            span_indices.append(list(range(len(s))))
        clusters = {}
        gold_antecedents = {}
        l = 0
        sents = []
        for sent in sentences:
            sents.append(sent['words'])

            for span in sent['coref_spans']:
                clusters[span[0]] = clusters.get(span[0], []) + [
                    Span(span[1] + l, span[2] + l, sent['words'][span[1]:(span[2] + 1)])]
            l += len(sent['words'])

        for cluster, spans in clusters.items():
            for i in range(len(spans)-1,0, -1):
                gold_antecedents[spans[i]] = spans[:i]

        return ids, gold_antecedents, sents

    def tokenize_test_data(self, doc):
        ids = []
        n = 0
        span_indices = []
        sentences = doc['sentences']
        # посмотреть формат ids
        for s in sentences:
            ids.append(batch_to_ids([s['words']]))
            # embed.append(self.word_emb(sids)['elmo_representations'][0])
            span_indices.append(list(range(len(s))))

        cspans = {}
        gold_antecedents = {}
        l = 0
        sents = []
        for sent in sentences:
            sents.append(sent['words'])

            for span in sent['coref_spans']:
                cspans[span[0]] = cspans.get(span[0], []) + [
                    (span[1] + l, span[2] + l)]
            l += len(sent['words'])

        mention_to_cluster = {}
        clusters = []
        for cluster_id, spans in cspans.items():
            spans = tuple(spans)
            mention_to_cluster.update({span: spans for span in spans})
            clusters.append(spans)

        return ids, clusters, mention_to_cluster, sents

    def forward(self, ids):
        word_embed = self.word_emb(ids)['elmo_representations'][0].squeeze(dim=0)
        return word_embed




class UntrainedEmbedder(nn.Module):
    def __init__(self,
                 lstm_hidden=200,
                 num_layers=2,
                 lstm_dropout=0.5,
                 char_emb_dim=8,
                 cnn_n_filters=50,
                 kernel_sizes=(3, 4, 5),
                 embedding_dim=300,
                 device=None):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.lstm_hidden = lstm_hidden
        self.num_layers = num_layers

        self.char_emb_dim = char_emb_dim
        self.cnn_n_filters = cnn_n_filters
        self.word_embed_dim = self.lstm_hidden * 2

        self.glove = vocab.GloVe(name='6B', dim=self.embedding_dim)
        self.cnn_emb = CharCNN(self.char_emb_dim, self.cnn_n_filters, kernel_sizes, device=device)

        self.lstm = nn.LSTM(input_size=self.embedding_dim + self.cnn_n_filters * len(kernel_sizes),
                            hidden_size=self.lstm_hidden,
                            num_layers=self.num_layers,
                            dropout=lstm_dropout,
                            batch_first=True,
                            bidirectional=True)

        self.device = device
        if not self.device:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def forward(self, sent):
        glove_emb = torch.stack([self.glove.get_vecs_by_tokens(word).to(self.device) for word in sent])
        cnn_emb = self.cnn_emb(sent)

        embed = torch.cat((glove_emb, cnn_emb), dim=1)  # [len(sent),embedding_dim+n_filters*kernel_size]
        output, _ = self.lstm(embed)  # [1, n_tokens, emb_dim]
        span_indices = list(range(len(sent)))
        return output

    def tokenize_train_data(self, doc):
        clusters = {}
        gold_antecedents = {}
        l = 0
        sents = []
        sentences = doc['sentences']
        for sent in sentences:
            sents.append(sent['words'])

            for span in sent['coref_spans']:
                clusters[span[0]] = clusters.get(span[0], []) + [
                    Span(span[1] + l, span[2] + l, sent['words'][span[1]:(span[2] + 1)])]
            l += len(sent['words'])

        for cluster, spans in clusters.items():
            for i in range(len(spans)-1,0, -1):
                gold_antecedents[spans[i]] = spans[:i]

        return sents, gold_antecedents, sents

    def tokenize_test_data(self, doc):
        cspans = {}
        gold_antecedents = {}
        l = 0
        sents = []
        sentences = doc['sentences']
        for sent in sentences:
            sents.append(sent['words'])

            for span in sent['coref_spans']:
                cspans[span[0]] = cspans.get(span[0], []) + [
                    Span(span[1] + l, span[2] + l, sent['words'][span[1]:(span[2] + 1)])]
            l += len(sent['words'])

        mention_to_cluster = {}
        clusters = []
        for cluster_id, spans in cspans.items():
            spans = tuple(spans)
            mention_to_cluster.update({span: spans for span in spans})
            clusters.append(spans)

        return sents, clusters, mention_to_cluster, sents


class CharCNN(nn.Module):
    def __init__(self, embedding_dim, n_filters, kernel_sizes, max_len=15, device='cpu'):
        super().__init__()

        self.build_vocab()
        self.embedding_dim = embedding_dim
        self.n_filters = n_filters
        self.n_convs = len(kernel_sizes)
        self.end_dim = self.n_filters * self.n_convs
        self.max_len = 15
        self.device = device

        self.emb = nn.Embedding(num_embeddings=self.vocab_size, embedding_dim=self.embedding_dim)
        self.convs = nn.ModuleList([nn.Conv2d(in_channels=1,
                                              out_channels=self.n_filters,
                                              kernel_size=(n, self.embedding_dim)) for n in kernel_sizes])

    def forward(self, x):
        words = self.make_vectors(x)
        embed = self.emb(words)
        embed = embed.unsqueeze(1)
        conveds = [F.relu(conv(embed)).squeeze(-1) for conv in self.convs]
        pooled = [F.max_pool1d(conved, kernel_size=conved.shape[-1]).squeeze(-1) for conved in conveds]
        united = torch.stack(pooled, dim=1).view((len(x), self.end_dim))

        return united

    def make_vectors(self, sent):
        """
        Transform sentence(that is list of words in string format) into tensor
        """
        token_ids = [[self.token2id.get(sym, 1) for sym in word[:self.max_len]] for word in sent]
        padded = [F.pad(torch.tensor(w_ids), pad=(0, self.max_len - len(w_ids))) for w_ids in token_ids]

        return torch.stack(padded).to(self.device)

    def build_vocab(self):
        self.token2id = {sym: i for i, sym in enumerate(list(string.printable), start=2)}
        self.token2id['<pad>'] = 0
        self.token2id['<unk>'] = 1
        self.vocab_size = len(self.token2id.keys())