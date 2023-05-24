from torch import nn


class FFNNScore(nn.Module):
    def __init__(self, input_dim, hidden_dim=150):
        super().__init__()
        self.pipeline = nn.Sequential(nn.Linear(input_dim, hidden_dim),
                                      nn.ReLU(),
                                      nn.Dropout(0.2),
                                      nn.Linear(hidden_dim, hidden_dim),
                                      nn.ReLU(),
                                      nn.Dropout(0.2),
                                      nn.Linear(hidden_dim, 1))

    def forward(self, x):
        return self.pipeline(x)


class Span:
    def __init__(self, i_start=-1, i_end=-1, word=' ', vector=None):
        self.i_start = i_start
        self.i_end = i_end
        self.words = tuple(word)
        self.vector = vector

    def __str__(self):
        return f'[{self.i_start}:{self.i_end}]'

    def __repr__(self):
        return f'Span([{self.i_start}:{self.i_end}], {self.word})'

    def __eq__(self, other):
        return self.i_start == other.i_start and self.i_end == other.i_end

    def __hash__(self):
        return hash((self.i_start, self.i_end))

