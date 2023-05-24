from random import shuffle
from prettytable import PrettyTable

import torch
from tqdm import tqdm
import matplotlib.pyplot as plt

from utils import Span
from evaluation import evaluate


class Trainer:
    def __init__(self, model, optimizer, scheduler, device=None):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler

        self.device = device
        if not self.device:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def train(self, train_dataset, val_dataset, savepath, num_epoch=150):
        self.model.train()
        losses = []
        for epoch in range(num_epoch):
            print(f'{epoch} epoch')
            indexes = list(range(len(train_dataset)))
            shuffle(indexes)
            epoch_losses = []
            n = 0
            for i in tqdm(indexes):
                self.optimizer.zero_grad()
                tokenized, gold_antecedents, sents = train_dataset[i]

                spans, prev_spans_probs = self.model(tokenized)
                if len(prev_spans_probs) < 1:
                    continue

                loss = self.coref_loss(gold_antecedents, spans, prev_spans_probs)
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()
                epoch_losses.append(loss.item())
                n += 1
            losses.append(epoch_losses)
            print(f'[{epoch + 1}] loss: {self.mean(epoch_losses):.3f} ')
            result = evaluate(val_dataset, self.model)
            self.print_metrics_table(result)

        torch.save(self.model.state_dict(), savepath)


    def print_metrics_table(self, metrics):
        t = PrettyTable(['', 'Precision', 'Recall', 'F1-score'])
        t.add_row(['MUC']+[round(score, 3) for score in metrics['muc']])
        t.add_row(['B-cubed']+[round(score, 3) for score in metrics['b_cubed']])
        t.add_row(['CEAF']+[round(score, 3) for score in metrics['ceafe']], divider=True)
        t.add_row(['AVG'] + [round(score, 3) for score in metrics['avg']])
        print(t)

    def mean(self, l):
        return sum(l)/len(l)

    def print_res(self, ls):
        fig = plt.figure(figsize=(18,9))
        plt.scatter(range(len(ls)), ls, s=2)
        plt.show()

    def coref_loss(self, gold_antecedents, spans, prev_spans_probs):
        epsilon_span = Span(-1, -1)
        loss = 0
        for i in range(len(spans)-1):
            gold_i = gold_antecedents.get(spans[i], [epsilon_span])
            spans_count = len(prev_spans_probs[i])
            antecedents = {span: span_probs
                           for span, span_probs in zip(spans[i + 1:i + spans_count + 1], prev_spans_probs[i])}
            antecedents[epsilon_span] = prev_spans_probs[i][-1]
            current_loss = torch.sum(torch.stack([antecedents.get(g, torch.tensor(0).to(self.device)) for g in gold_i]))
            if current_loss == 0:
                current_loss = antecedents.get(epsilon_span, torch.tensor(0).to(self.device))
            current_loss = -torch.log(current_loss)
            loss += current_loss

        return loss