import yaml

import torch
from datasets import load_dataset
from torch import optim

import span_embedders
from models import CoreferenceModel
from trainer import Trainer
import word_embedders

with open('config.yml', 'r') as file:
   config_data = yaml.safe_load(file)

if __name__ == '__main__':
    print('Initialization')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if config_data['embedding_model'] == 'bert':
        bert_config = config_data['bert']
        word_embedder = word_embedders.Bert(model_path=bert_config['model_path'],
                                            chunk_len=bert_config['chunk_len'],
                                            stride_for_overlap=bert_config['stride'],
                                            word_embed_dim=bert_config['token_embed_dim']
                                            )
        span_embedder = span_embedders.SpanEmbedderBert(embedding_model=word_embedder,
                                                    embedding_dim=bert_config['token_embed_dim'],
                                                    l=config_data['max_span_len'],
                                                    attn_hidden=config_data['ffnn_size'],
                                                    device=device
                                                    )
        word_embed_dim = word_embedder.word_embed_dim
    elif config_data['embedding_model'] == 'elmo':
        elmo_config = config_data['elmo']
        word_embedder = word_embedders.ELMO(model_size=elmo_config['model_size'],
                                            num_output_representations=elmo_config['num_output_representations'],
                                            dropout=elmo_config['dropout'],
                                            requires_grad=elmo_config['requires_grad'],
                                            device=device)
        word_embed_dim = word_embedder.word_embed_dim
        span_embedder = span_embedders.SpanEmbedderElmo(embedding_model=word_embedder,
                                                    embedding_dim=word_embed_dim,
                                                    l=config_data['max_span_len'],
                                                    attn_hidden=config_data['ffnn_size'],
                                                    device=device
                                                    )
    elif config_data['embedding_model'] == 'untrained':
        untrained_config = config_data['untrained']
        word_embedder = word_embedders.UntrainedEmbedder(lstm_hidden=untrained_config['lstm_hidden'],
                                                         num_layers=untrained_config['num_lstm_layers'],
                                                         lstm_dropout=untrained_config['lstm_dropout_rate'],
                                                         char_emb_dim=untrained_config['char_embedding_size'],
                                                         cnn_n_filters=untrained_config['filter_size'],
                                                         kernel_sizes=untrained_config['filter_widths'])
        word_embed_dim = word_embedder.word_embed_dim
        span_embedder = span_embedders.SpanEmbedder(embedding_model=word_embedder,
                                                    embedding_dim=word_embed_dim,
                                                    l=config_data['max_span_len'],
                                                    attn_hidden=config_data['ffnn_size'],
                                                    device=device
                                                    )
    else:
        raise ValueError("Incorrect value for 'embedding_model' parameter. Valid values: bert, elmo, untrained.")

    # Data preprocessing
    train_dataset = load_dataset("conll2012_ontonotesv5", 'english_v12', split='train')
    cashed_train = [word_embedder.tokenize_train_data(train_dataset.__getitem__(i))
                    for i in range(train_dataset.__len__())]

    val_dataset = load_dataset("conll2012_ontonotesv5", 'english_v12', split='validation')
    cashed_val = [word_embedder.tokenize_test_data(train_dataset.__getitem__(i))
                  for i in range(val_dataset.__len__())]

    print('Models init')

    coref_model = CoreferenceModel(span_embedder=span_embedder, word_emb_dim=word_embed_dim, device=device).to(device)
    optimizer = optim.Adam(coref_model.parameters(), lr=config_data['learning_rate'])
    scheduler = optim.lr_scheduler.StepLR(optimizer,
                                          step_size=config_data['lr_decay_step'],
                                          gamma=config_data['lr_decay_gamma'])
    trainer = Trainer(coref_model, optimizer, scheduler)

    trainer.train(cashed_train, cashed_val, config_data['savepath'], num_epoch=config_data['num_epoch'])