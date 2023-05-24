from tqdm import tqdm
from typing import List, Any

import torch
from allennlp_models.coref.metrics.conll_coref_scores import Scorer


def get_predicted_clusters(spans: List[Any], spans_probs: List[torch.Tensor]):
    """
    Function for predicting clusters.
    :param spans: List[Span] - list of top spans after all pruning steps
    :param spans_probs: List[torch.Tensor] - list of antecedents probabilities for every span from spans
    :return:final_clusters: List[Tuple[int, int]] - list of clusters,
            mention_to_cluster: Dict[Tuple[int,int]] - dict of span to whole cluster list
    """
    clusters = []
    mention_to_cluster_id = {}
    for i in range(len(spans)-2, -1, -1):
        span = (spans[i].i_start, spans[i].i_end)
        span_probs = spans_probs[i]
        antecedents = spans[i + 1:i + len(span_probs) + 1]
        antecedent_ind = torch.argmax(span_probs)
        if antecedent_ind == len(span_probs) - 1:
            # mention_to_cluster_id[span] = len(clusters)
            # clusters.append([span])
            continue

        predicted_antecedent = antecedents[antecedent_ind]
        predicted_antecedent = (predicted_antecedent.i_start, predicted_antecedent.i_end)
        if bool(mention_to_cluster_id.get(predicted_antecedent, -1)+1):
            cluster_id = mention_to_cluster_id[predicted_antecedent]
        else:
            cluster_id = len(clusters)
            mention_to_cluster_id[predicted_antecedent] = cluster_id
            clusters.append([predicted_antecedent])

        mention_to_cluster_id[span] = cluster_id
        clusters[cluster_id].append(span)

    final_clusters = [tuple(cluster) for cluster in clusters]
    mention_to_cluster = {
        mention: final_clusters[cluster_id]
        for mention, cluster_id in mention_to_cluster_id.items()
    }

    return final_clusters, mention_to_cluster


def evaluate(dataset, model):
    """
    Evaluation function
    :param dataset: test dataset, every example contain inputs for model,
                    clusters: List[Tuple[int,int]] - list of clusters, span - tuple (start_ind, end_ind)
                    gold_mention_to_cluster: Dict[Tuple[int,int]] - dict: key - span, item - whole cluster from clusters
                    tokens - all document tokens
    :param model:  model, that should be evaluated
    :return: results: Dict[str] - key - metric name,
                                  item: List[float, float, float] - list of precision, recall, f1-score results for this metric
    """
    scorers = [Scorer(m) for m in (Scorer.muc, Scorer.b_cubed, Scorer.ceafe)]
    score_names = ['muc', 'b_cubed', 'ceafe']
    for data in tqdm(dataset):
        inputs, gold_clusters, gold_mention_to_cluster, tokens = data
        with torch.no_grad():
            spans, prev_spans_probs = model(inputs)
        predicted_clusters, predicted_mention_to_cluster = get_predicted_clusters(spans, prev_spans_probs)

        for scorer in scorers:
            scorer.update(
                predicted_clusters, gold_clusters, predicted_mention_to_cluster, gold_mention_to_cluster
            )
    metrics = (lambda e: e.get_precision(), lambda e: e.get_recall(), lambda e: e.get_f1())
    results = {}
    for scorer, sname in zip(scorers, score_names):
        results[sname] = [metric(scorer) for metric in metrics]
    results['avg'] = [ sum(metric(e) for e in scorers) / len(scorers) for metric in metrics]
    return results