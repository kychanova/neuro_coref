import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import Span, FFNNScore


class CoreferenceModel(nn.Module):
    def __init__(self,
                 span_embedder,
                 word_emb_dim=256,
                 p_lambda=0.4,
                 k=250, device=None):
        super().__init__()
        self.span_emb = span_embedder
        self.mention_scorer = FFNNScore(word_emb_dim * 3)
        self.p_lambda = p_lambda
        self.k = k
        self.pairwise_scorer = FFNNScore(word_emb_dim * 9)

        self.device = device
        if not self.device:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def forward(self, doc):
        def get_antecedent_scores(i):
            spans_count = min(self.k, n - i - 1)
            span_repeated = vectors_with_scores[i, :-1].repeat((spans_count, 1))
            possible_corefs = vectors_with_scores[(i + 1):(i + spans_count + 1), :-1]
            pair_repr = torch.cat((span_repeated,
                                   possible_corefs,
                                   span_repeated * possible_corefs),
                                  dim=1)
            pairwise_score = self.pairwise_scorer(pair_repr)
            antecedents_scores = torch.cat((pairwise_score.squeeze() + \
                                            vectors_with_scores[(i + 1):(i + spans_count + 1), -1] + \
                                            vectors_with_scores[i, -1].repeat(spans_count),
                                            torch.tensor([0]).to(self.device)), dim=-1)
            spans_probs = F.softmax(antecedents_scores)
            return spans_probs

        # get all spans representations
        spans = []
        spans, t = self.span_emb(doc=doc)

        # computing mention scores
        spans, vectors = zip(*spans)
        vectors = torch.stack(vectors)
        mention_scores = self.mention_scorer(vectors)

        # pruning by mention scores
        mention_scores, sort_inds = mention_scores.sort(dim=0, descending=True)
        mention_scores = mention_scores[:int(self.p_lambda * t)]
        sort_inds = sort_inds[:int(self.p_lambda * t)].squeeze()
        if sort_inds.size() == torch.Size([]):
            sort_inds = torch.tensor([sort_inds.item()]).to(self.device)
        vectors = vectors.index_select(dim=0, index=sort_inds)
        spans = [spans[i] for i in sort_inds]

        # sort by span indices
        sort_inds, spans = zip(*sorted(enumerate(spans), key=lambda x: (x[1].i_start, x[1].i_end), reverse=True))
        sort_inds = torch.tensor(sort_inds).to(self.device)
        vectors_with_scores = torch.column_stack([vectors, mention_scores])
        vectors_with_scores = vectors_with_scores.index_select(dim=0, index=sort_inds)

        n = vectors_with_scores.shape[0]
        output = [get_antecedent_scores(i) for i in range(n - 1)]

        return spans, output