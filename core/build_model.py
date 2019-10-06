"""
Created on 2019/3/2 15:58

@author: zhouweixin
@note: 模型实例化函数
"""

from model.narre import NARRE
from model.narre_fa import NARRE_FA
from model.narre_soa import NARRE_SOA

# 评论特征提取方法
from model.narre_lstm import NARRE_LSTM
from model.narre_lstm_sa import NARRE_LSTM_SA
from model.narre_cnn_lstm import NARRE_CNN_LSTM
from model.narre_cnn_lstm_sa import NARRE_CNN_LSTM_SA
from model.hanci import HANCI


def build_NARRE(train_dset, hyperparas):
    model = NARRE(train_dset=train_dset,
                  word_embedding_size=hyperparas['embedding_size'],
                  id_embedding_size=hyperparas['id_embedding_size'],
                  filter_sizes=hyperparas['filter_sizes'],
                  num_filters=hyperparas['num_filters'],
                  attention_size=hyperparas['attention_size'],
                  num_latent=hyperparas['num_latent'],
                  dropout=hyperparas['dropout'])
    return model


def build_NARRE_FA(train_dset, hyperparas):
    model = NARRE_FA(train_dset=train_dset,
                     word_embedding_size=hyperparas['embedding_size'],
                     id_embedding_size=hyperparas['id_embedding_size'],
                     filter_sizes=hyperparas['filter_sizes'],
                     num_filters=hyperparas['num_filters'],
                     attention_size=hyperparas['attention_size'],
                     num_latent=hyperparas['num_latent'],
                     dropout=hyperparas['dropout'])
    return model


def build_NARRE_SOA(train_dset, hyperparas):
    model = NARRE_SOA(train_dset=train_dset,
                      word_embedding_size=hyperparas['embedding_size'],
                      id_embedding_size=hyperparas['id_embedding_size'],
                      filter_sizes=hyperparas['filter_sizes'],
                      num_filters=hyperparas['num_filters'],
                      attention_size=hyperparas['attention_size'],
                      num_latent=hyperparas['num_latent'],
                      dropout=hyperparas['dropout'],
                      soa_size=hyperparas['soa_size'])
    return model


def build_NARRE_LSTM(train_dset, hyperparas):
    model = NARRE_LSTM(train_dset=train_dset,
                       word_embedding_size=hyperparas['embedding_size'],
                       id_embedding_size=hyperparas['id_embedding_size'],
                       hidden_size=hyperparas['hidden_size'],
                       num_layers=hyperparas['num_layers'],
                       bidirectional=hyperparas['bidirectional'],
                       attention_size=hyperparas['attention_size'],
                       num_latent=hyperparas['num_latent'],
                       dropout=hyperparas['dropout'])
    return model


def build_NARRE_LSTM_SA(train_dset, hyperparas):
    model = NARRE_LSTM_SA(train_dset=train_dset,
                          word_embedding_size=hyperparas['embedding_size'],
                          id_embedding_size=hyperparas['id_embedding_size'],
                          hidden_size=hyperparas['hidden_size'],
                          num_layers=hyperparas['num_layers'],
                          bidirectional=hyperparas['bidirectional'],
                          da=hyperparas['da'],
                          r=hyperparas['r'],
                          attention_size=hyperparas['attention_size'],
                          num_latent=hyperparas['num_latent'],
                          dropout=hyperparas['dropout'], )
    return model


def build_HANCI(train_dset, hyperparas):
    model = HANCI(train_dset=train_dset,
                  word_embedding_size=hyperparas['embedding_size'],
                  id_embedding_size=hyperparas['id_embedding_size'],
                  hidden_size=hyperparas['hidden_size'],
                  num_layers=hyperparas['num_layers'],
                  bidirectional=hyperparas['bidirectional'],
                  da=hyperparas['da'],
                  r=hyperparas['r'],
                  attention_size=hyperparas['attention_size'],
                  num_latent=hyperparas['num_latent'],
                  dropout=hyperparas['dropout'],
                  soa_size=hyperparas['soa_size'], )
    return model


def build_NARRE_CNN_LSTM(train_dset, hyperparas):
    model = NARRE_CNN_LSTM(train_dset=train_dset,
                           word_embedding_size=hyperparas['embedding_size'],
                           id_embedding_size=hyperparas['id_embedding_size'],
                           filter_sizes=hyperparas['filter_sizes'],
                           num_filters=hyperparas['num_filters'],
                           hidden_size=hyperparas['hidden_size'],
                           num_layers=hyperparas['num_layers'],
                           bidirectional=hyperparas['bidirectional'],
                           attention_size=hyperparas['attention_size'],
                           num_latent=hyperparas['num_latent'],
                           dropout=hyperparas['dropout'], )
    return model


def build_NARRE_CNN_LSTM_SA(train_dset, hyperparas):
    model = NARRE_CNN_LSTM_SA(train_dset=train_dset,
                              word_embedding_size=hyperparas['embedding_size'],
                              id_embedding_size=hyperparas['id_embedding_size'],
                              filter_sizes=hyperparas['filter_sizes'],
                              num_filters=hyperparas['num_filters'],
                              hidden_size=hyperparas['hidden_size'],
                              num_layers=hyperparas['num_layers'],
                              bidirectional=hyperparas['bidirectional'],
                              da=hyperparas['da'],
                              r=hyperparas['r'],
                              attention_size=hyperparas['attention_size'],
                              num_latent=hyperparas['num_latent'],
                              dropout=hyperparas['dropout'], )
    return model
