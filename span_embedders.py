from random import sample

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import FFNNScore, Span


class SpanEmbedder(nn.Module):
    def __init__(self,
                 embedding_model,
                 embedding_dim=300,
                 l=10,
                 attn_hidden=150,
                 max_segments=50,
                 device=None):
        super().__init__()
        self.embedding_model = embedding_model

        self.l = l
        self.max_segments = max_segments

        self.word_emb = embedding_model
        self.attn = FFNNScore(embedding_dim, attn_hidden)

        self.device = device
        if not self.device:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'


    def forward(self, doc):
        spans = []
        start_ind = 0
        segments_count = len(doc)
        if segments_count > self.max_segments:
            exclude_inds = sorted(sample(range(segments_count), segments_count-self.max_segments))
        # ind_mask = [True]*segments_count
        # for i in exclude_inds:
        #     ind_mask[i] = False
            for i in exclude_inds[-1::-1]:
                del doc[i]
        for sent in doc:
            # if not ind_mask[i]:
            #     continue
            n = len(sent)
            word_embed = self.word_emb(sent)
            attn_score = self.attn(word_embed)
            spans.extend([(Span(i + start_ind, j + start_ind),
                      self.span_embeder(attn_score[i:j + 1],
                                        word_embed[i:j + 1]))
                     for i in range(n) for j in range(i, min(i + self.l, n))])
            start_ind += n
        return spans, start_ind


    def span_embeder(self, attn_scores, word_embeds):
        head_emb = torch.sum(F.softmax(attn_scores) * word_embeds, dim=0)
        return torch.cat((word_embeds[0], word_embeds[-1], head_emb))


class SpanEmbedderElmo(SpanEmbedder):
    def forward(self, doc):
        spans = []
        start_ind = 0
        segments_count = len(doc)
        if segments_count > self.max_segments:
            exclude_inds = sorted(sample(range(segments_count), segments_count - self.max_segments))
            for i in exclude_inds[-1::-1]:
                del doc[i]
        for sent in doc:
            n = sent.shape[1]
            word_embed = self.word_emb(sent)
            attn_score = self.attn(word_embed)
            spans.extend([(Span(i + start_ind, j + start_ind),
                           self.span_embeder(attn_score[i:j + 1],
                                             word_embed[i:j + 1]))
                          for i in range(n) for j in range(i, min(i + self.l, n))])
            start_ind += n
        return spans, start_ind

class SpanEmbedderBert(SpanEmbedder):
    def forward(self, doc):
        spans = []
        start_ind = 0
        segments_count = doc['input_ids'].shape[0]
        if segments_count > self.max_segments:
            indexes = sample(range(segments_count), self.max_segments)
            for key in doc.keys():
                doc[key] = doc[key].index_select(dim=0, index=indexes)
        word_embeds = self.word_emb(**doc)
        attn_scores = self.attn(word_embeds)
        for word_embed, attn_score in zip(word_embeds, attn_scores) :
            n = word_embed.shape[0]
            spans.extend([(Span(i + start_ind, j + start_ind),
                      self.span_embeder(attn_score[i:j + 1],
                                        word_embed[i:j + 1]))
                     for i in range(1,n-1) for j in range(i, min(i + self.l, n-1))])
            start_ind += n
        return spans, start_ind



