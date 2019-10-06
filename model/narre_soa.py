"""
Created on 2019/3/2 15:58

@author: zhouweixin
@note: 论文的原型：用CNN提取特征 + soft-attention
"""

import torch
from torch import nn
from model.language import WordEmbedding, ReviewCNN
from model.attention import Attention, SoftAttention
from util import utils


class NARRE_SOA(nn.Module):
    """
    CNN 提取评论用户偏好和项目特征 + soft-attention
    """

    def __init__(self, train_dset, word_embedding_size=300, id_embedding_size=32,
                 filter_sizes=[3], num_filters=100, attention_size=32, num_latent=32, dropout=0.5,
                 soa_size=50):
        super(NARRE_SOA, self).__init__()

        self.review_num_filters = num_filters * len(filter_sizes)
        self.user_word_emb = WordEmbedding(train_dset.dictionary.ntoken, word_embedding_size)
        self.item_word_emb = WordEmbedding(train_dset.dictionary.ntoken, word_embedding_size)

        self.user_review_emb = ReviewCNN(train_dset.user_review_len, embedding_size=word_embedding_size,
                                         filter_sizes=filter_sizes, num_filters=num_filters)
        self.item_review_emb = ReviewCNN(train_dset.item_review_len, embedding_size=word_embedding_size,
                                         filter_sizes=filter_sizes, num_filters=num_filters)

        self.user_id_emb = WordEmbedding(train_dset.user_num, id_embedding_size, dropout=0.)
        self.item_id_emb = WordEmbedding(train_dset.item_num, id_embedding_size, dropout=0.)
        self.user_rid_emb = WordEmbedding(train_dset.item_num, id_embedding_size, dropout=0.)
        self.item_rid_emb = WordEmbedding(train_dset.user_num, id_embedding_size, dropout=0.)

        self.user_reviews_att = Attention(self.review_num_filters, id_embedding_size, attention_size)
        self.item_reviews_att = Attention(self.review_num_filters, id_embedding_size, attention_size)

        self.user_soft_attention = SoftAttention(self.review_num_filters, id_embedding_size, soa_size)
        self.item_soft_attention = SoftAttention(self.review_num_filters, id_embedding_size, soa_size)

        self.user_latent = nn.Linear(self.review_num_filters, num_latent)
        self.item_latent = nn.Linear(self.review_num_filters, num_latent)
        self.relu = nn.ReLU()

        self.classify = nn.Linear(num_latent, 1, bias=False)
        self.dropout = nn.Dropout(dropout)

        # 分别用作user和item的偏置
        self.user_bias = WordEmbedding(ntoken=train_dset.user_num, emb_dim=1, dropout=0.)
        self.item_bias = WordEmbedding(ntoken=train_dset.item_num, emb_dim=1, dropout=0.)
        self.global_bias = nn.Parameter(torch.Tensor([0.1]))

        # 初始化
        self.user_word_emb.init_embedding()
        self.item_word_emb.init_embedding()
        self.user_bias.init_embedding_with_one(0.1)
        self.item_bias.init_embedding_with_one(0.1)

    def forward(self, user_id, item_id, user_reviews, item_reviews, user_rids, item_rids, rating, norm_lambda=0.001, save_att=False):
        """
        :param user_id: [batch, 1]
        :param item_id: [batch, 1]
        :param user_reviews: [batch, user_review_num, user_review_len]
        :param item_reviews: [batch, item_review_num, item_review_len]
        :param user_rids: [batch, user_review_num]
        :param item_rids: [batch, item_review_num]
        :param rating: [batch, 1]
        :return:
        """
        # 1.word_embedding
        user_word_emb = self.user_word_emb(
            user_reviews)  # [batch, user_review_num, user_review_len, word_embedding_size]
        item_word_emb = self.item_word_emb(
            item_reviews)  # [batch, item_review_num, item_review_len, word_embedding_size]

        # 2.review_embedding
        user_reviews_emb = self.user_review_emb(user_word_emb)  # [batch, user_review_num, review_num_filters]
        item_reviews_emb = self.item_review_emb(item_word_emb)  # [batch, item_review_num, review_num_filters]

        # 3.id_embedding
        user_id_emb = self.user_id_emb(user_id)  # [batch, 1, id_embedding_size]
        item_id_emb = self.item_id_emb(item_id)  # [batch, 1, id_embedding_size]
        user_rids_emb = self.user_rid_emb(user_rids)  # [batch, user_review_num, id_embedding_size]
        item_rids_emb = self.item_rid_emb(item_rids)  # [batch, item_review_num, id_embedding_size]

        user_id_emb = user_id_emb.view(-1, user_id_emb.size(2)) # [batch, id_embedding_size]
        item_id_emb = item_id_emb.view(-1, item_id_emb.size(2)) # [batch, id_embedding_size]

        # 4.attention
        user_reviews_att = self.user_reviews_att(user_reviews_emb, user_rids_emb)  # [batch, user_review_num, 1]
        item_reviews_att = self.item_reviews_att(item_reviews_emb, item_rids_emb)  # [batch, item_review_num, 1]

        # 5.add_reviews
        user = self.dropout(torch.sum(user_reviews_emb * user_reviews_att, 1))  # [batch, review_num_filters]
        item = self.dropout(torch.sum(item_reviews_emb * item_reviews_att, 1))  # [batch, review_num_filters]

        # soft-attention
        user = self.item_soft_attention(user, item_id_emb) # [batch, review_num_filters]
        item = self.user_soft_attention(item, user_id_emb) # [batch, review_num_filters]

        # 6.LFM
        user_latent = self.user_latent(user) + user_id_emb  # [batch, num_latent]
        item_latent = self.item_latent(item) + item_id_emb  # [batch, num_latent]

        # 方法二：
        output = self.relu(user_latent * item_latent)  # [batch, num_latent]
        output = self.dropout(output)
        output = self.classify(output)  # [batch, 1]
        user_bias = self.user_bias(user_id).view(-1, 1)  # [batch, 1]
        item_bias = self.item_bias(item_id).view(-1, 1)  # [batch, 1]
        predictions = output + user_bias + item_bias + self.global_bias  # [batch, 1]

        # loss
        l2_loss = 0.
        l2_loss += utils.l2_loss(self.user_reviews_att.review_linear.weight)
        l2_loss += utils.l2_loss(self.user_reviews_att.id_linear.weight)
        l2_loss += utils.l2_loss(self.item_reviews_att.review_linear.weight)
        l2_loss += utils.l2_loss(self.item_reviews_att.id_linear.weight)

        loss = utils.l2_loss(predictions, rating) + norm_lambda * l2_loss

        return predictions, loss
