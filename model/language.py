"""
Created on 2019/3/2 15:58

@author: zhouweixin
@note: 词嵌入和特征提取
"""

import numpy as np
import torch
import torch.nn as nn
from model.attention import SelfAttention
from core import config


class WordEmbedding(nn.Module):
    def __init__(self, ntoken, emb_dim=300, dropout=0.5):
        super(WordEmbedding, self).__init__()

        self.ntoken = ntoken
        self.emb_dim = emb_dim
        self.dropout = dropout

        self.embedding = nn.Embedding(ntoken + 1, emb_dim, padding_idx=ntoken)
        self.dropout = nn.Dropout(p=dropout)

    def init_embedding(self, np_file=config.glove_init_file):
        weight_init = torch.from_numpy(np.load(np_file))
        assert weight_init.shape == (self.ntoken, self.emb_dim)

        self.embedding.weight.data[:self.ntoken] = weight_init

    def init_embedding_with_one(self, value=0.001):
        self.embedding.weight.data = torch.ones_like(self.embedding.weight) * value

    def forward(self, x):
        x = self.embedding(x)
        x = self.dropout(x)
        return x


class ReviewCNN(nn.Module):
    """
    CNN 从评论里提取特征
    """

    def __init__(self, review_len, embedding_size=300, filter_sizes=[3], num_filters=100, dropout=0.):
        super(ReviewCNN, self).__init__()

        self.filter_sizes = filter_sizes
        self.num_filters = num_filters
        self.convs = nn.Sequential()

        for i, filter_size in enumerate(filter_sizes):
            conv = nn.Sequential(
                # x: [batch, review_num, review_len, embedding_size]
                # ->x: [batch*review_num, 1, review_len, embedding_size]
                # conv: [batch*review_num, num_filters, review_len-filter_size+1, 1]
                # pool: [batch*review_num, num_filters, 1, 1]
                nn.Conv2d(in_channels=1, out_channels=num_filters, kernel_size=(filter_size, embedding_size), stride=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=(review_len - filter_size + 1, 1), stride=1)
            )
            self.convs.add_module(str(i), conv)

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        """
        :param x: [batch, review_num, review_len, embedding_size]
        :return:
        """
        batch = x.size(0)
        review_num = x.size(1)
        review_len = x.size(2)
        embedding_size = x.size(3)

        # 1.卷积层
        x = x.view(-1, 1, review_len, embedding_size)  # [batch*review_num, 1, review_len, embedding_size]

        outputs = []
        for conv in self.convs:
            x1 = conv(x)  # [batch*review_num, num_filters, 1, 1]
            x1 = x1.view(batch, review_num, self.num_filters)  # [batch, review_num, num_filters]
            outputs.append(x1)

        # output: [len(filter_sizes), batch, review_num, num_filters]
        output = torch.cat(outputs, dim=2)  # [batch, review_num, num_fileters*len(filter_sizes)]
        output = self.dropout(output)

        return output


class ReviewLSTM(nn.Module):
    """
    LSTM 从评论里提取特征
    """

    def __init__(self, embedding_size=300, dropout=0.,
                 hidden_size=512, num_layers=1, bidirectional=False):
        super(ReviewLSTM, self).__init__()

        self.lstm = nn.LSTM(
            input_size=embedding_size, hidden_size=hidden_size, num_layers=num_layers,
            bidirectional=bidirectional,
            dropout=dropout,
            batch_first=True)

        self.ndirections = 1 + int(bidirectional)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        """
        :param x: [batch, review_num, review_len, embedding_size]
        :return:
        """
        batch = x.size(0)
        review_num = x.size(1)
        review_len = x.size(2)
        embedding_size = x.size(3)

        x = x.view(-1, review_len, embedding_size)  # [batch*review_num, review_len, embedding_size]
        # output: [batch*review_num, review_len, hidden_layers*ndirections]
        # hn: [ndirections*num_layers, batch*review_num, hidden_layers]
        output, (hn, cn) = self.lstm(x)
        # [batch*review_num, hidden_layers*ndirections]
        output = output[:, -1]
        # [batch, review_num, hidden_layers*ndirections]
        output = output.view(batch, review_num, -1)

        return output


class ReviewLSTMSA(nn.Module):
    """
    LSTM + self-attention从评论里提取特征
    """

    def __init__(self, embedding_size=300, dropout=0.,
                 hidden_size=512, num_layers=1, bidirectional=False, da=100, r=10):
        super(ReviewLSTMSA, self).__init__()

        self.lstm = nn.LSTM(
            input_size=embedding_size, hidden_size=hidden_size, num_layers=num_layers,
            bidirectional=bidirectional,
            dropout=dropout,
            batch_first=True)

        self.ndirections = 1 + int(bidirectional)
        self.dropout = nn.Dropout(p=dropout)

        self.review_hidden_size = hidden_size * (1 + int(bidirectional))
        self.da = da
        self.r = r

        self.review_att = SelfAttention(self.review_hidden_size, da, r)

    def forward(self, x):
        """
        :param x: [batch, review_num, review_len, embedding_size]
        :return:
        """
        batch = x.size(0)
        review_num = x.size(1)
        review_len = x.size(2)
        embedding_size = x.size(3)

        # [batch*review_num, review_len, embedding_size]
        x = x.view(-1, review_len, embedding_size)

        # output: [batch*review_num, review_len, review_hidden_size]
        # hn: [ndirections*num_layers, batch*review_num, hidden_layers]
        output, (hn, cn) = self.lstm(x)

        review_embedding = output # [batch*review_num, review_len, review_hidden_size]
        attention = self.review_att(review_embedding) # [batch*review_num, r, review_len]
        review_embedding = attention @ review_embedding  # [batch*review_num, r, review_hidden_size]
        review_embedding = torch.sum(review_embedding, 1) / self.r  # [batch*review_num, review_hidden_size]
        review_embedding = review_embedding.view(batch, review_num, self.review_hidden_size) # [batch, review_num, review_hidden_size]

        att = torch.mean(attention, 1) # [batch*review_num, review_len]
        att = att.view(batch, review_num, review_len) # [batch, review_num, review_len]

        return review_embedding, att


class ReviewCNNLSTM(nn.Module):
    """
    CNN-LSTM 从评论里提取特征
    """

    def __init__(self, embedding_size=300, filter_sizes=[3], num_filters=100, dropout=0.,
                 hidden_size=512, num_layers=1, bidirectional=False):
        super(ReviewCNNLSTM, self).__init__()

        self.filter_sizes = filter_sizes
        self.num_filters = num_filters
        self.convs = nn.Sequential()

        for i, filter_size in enumerate(filter_sizes):
            conv = nn.Sequential(
                # x: [batch, review_num, review_len, embedding_size]
                # ->x: [batch*review_num, 1, review_len, embedding_size]
                # conv: [batch*review_num, num_filters, review_len-filter_size+1, 1]
                # pool: [batch*review_num, num_filters, 1, 1]
                nn.Conv2d(in_channels=1, out_channels=num_filters, kernel_size=(filter_size, embedding_size), stride=1),
                nn.ReLU(),
                # nn.MaxPool2d(kernel_size=(review_len - filter_size + 1, 1), stride=1)
            )
            self.convs.add_module(str(i), conv)

        self.dropout = nn.Dropout(p=dropout)

        self.lstm = nn.LSTM(
            input_size=num_filters*len(filter_sizes), hidden_size=hidden_size, num_layers=num_layers,
            bidirectional=bidirectional,
            dropout=dropout,
            batch_first=True)

        self.ndirections = 1 + int(bidirectional)

    def forward(self, x):
        """
        :param x: [batch, review_num, review_len, embedding_size]
        :return:
        """
        batch = x.size(0)
        review_num = x.size(1)
        review_len = x.size(2)
        embedding_size = x.size(3)

        # 1.卷积层
        x = x.view(-1, 1, review_len, embedding_size)  # [batch*review_num, 1, review_len, embedding_size]

        outputs = []
        for conv in self.convs:
            x1 = conv(x)  # [batch*review_num, num_filters, review_len - filter_size + 1, 1]
            x1 = x1.view(batch*review_num, -1, self.num_filters)  # [batch*review_num, review_len - filter_size + 1, num_filters]
            outputs.append(x1)

        # output: [len(filter_sizes), batch*review_num, review_len - filter_size + 1, num_filters]
        cnn_output = torch.cat(outputs, dim=2)  # [batch*review_num, (review_len - filter_size + 1), num_filters*len(filter_sizes)]
        cnn_output = self.dropout(cnn_output)

        # output: [batch*review_num, (review_len - filter_size + 1), hidden_layers*ndirections]
        # hn: [ndirections*num_layers, batch*review_num, hidden_layers]
        output, (hn, cn) = self.lstm(cnn_output)

        # [batch*review_num, hidden_layers*ndirections]
        lstm_output = output[:, -1]
        # [batch, review_num, hidden_layers*ndirections]
        lstm_output = lstm_output.view(batch, review_num, -1)
        return lstm_output


class ReviewCNNLSTMSA(nn.Module):
    """
    CNN-LSTM + self-attention从评论里提取特征
    """

    def __init__(self, embedding_size=300, filter_sizes=[3], num_filters=100, dropout=0.,
                 hidden_size=512, num_layers=1, bidirectional=False, da=100, r=10):
        super(ReviewCNNLSTMSA, self).__init__()

        self.filter_sizes = filter_sizes
        self.num_filters = num_filters
        self.convs = nn.Sequential()

        for i, filter_size in enumerate(filter_sizes):
            conv = nn.Sequential(
                # x: [batch, review_num, review_len, embedding_size]
                # ->x: [batch*review_num, 1, review_len, embedding_size]
                # conv: [batch*review_num, num_filters, review_len-filter_size+1, 1]
                # pool: [batch*review_num, num_filters, 1, 1]
                nn.Conv2d(in_channels=1, out_channels=num_filters, kernel_size=(filter_size, embedding_size), stride=1),
                nn.ReLU(),
                # nn.MaxPool2d(kernel_size=(review_len - filter_size + 1, 1), stride=1)
            )
            self.convs.add_module(str(i), conv)

        self.dropout = nn.Dropout(p=dropout)

        self.lstm = nn.LSTM(
            input_size=num_filters*len(filter_sizes), hidden_size=hidden_size, num_layers=num_layers,
            bidirectional=bidirectional,
            dropout=dropout,
            batch_first=True)

        self.ndirections = 1 + int(bidirectional)

        self.review_hidden_size = hidden_size * (1 + int(bidirectional))
        self.da = da
        self.r = r

        self.review_att = SelfAttention(self.review_hidden_size, da, r)

    def forward(self, x):
        """
        :param x: [batch, review_num, review_len, embedding_size]
        :return:
        """
        batch = x.size(0)
        review_num = x.size(1)
        review_len = x.size(2)
        embedding_size = x.size(3)

        # 1.卷积层
        x = x.view(-1, 1, review_len, embedding_size)  # [batch*review_num, 1, review_len, embedding_size]

        outputs = []
        for conv in self.convs:
            x1 = conv(x)  # [batch*review_num, num_filters, review_len - filter_size + 1, 1]
            x1 = x1.view(batch*review_num, -1, self.num_filters)  # [batch*review_num, review_len - filter_size + 1, num_filters]
            outputs.append(x1)

        # output: [len(filter_sizes), batch*review_num, review_len - filter_size + 1, num_filters]
        cnn_output = torch.cat(outputs, dim=2)  # [batch*review_num, (review_len - filter_size + 1), num_filters*len(filter_sizes)]
        cnn_output = self.dropout(cnn_output)

        # output: [batch*review_num, (review_len - filter_size + 1), review_hidden_size]
        # hn: [ndirections*num_layers, batch*review_num, hidden_layers]
        output, (hn, cn) = self.lstm(cnn_output)

        # [batch*review_num, (review_len - filter_size + 1), review_hidden_size]
        lstm_output = output

        attention = self.review_att(lstm_output)  # [batch*review_num, r, (review_len - filter_size + 1)]
        review_embedding = attention @ lstm_output  # [batch*review_num, r, review_hidden_size]
        review_embedding = torch.sum(review_embedding, 1) / self.r  # [batch*review_num, review_hidden_size]
        review_embedding = review_embedding.view(batch, review_num, self.review_hidden_size)  # [batch, review_num, review_hidden_size]
        return review_embedding