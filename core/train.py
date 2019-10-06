"""
Created on 2019/3/2 15:58

@author: zhouweixin
@note: 训练函数
"""

import os
import sys
import time
import numpy as np
import torch
from torch import nn, optim
from torch.autograd import Variable
from core import config
from util import utils

MSELoss = nn.MSELoss()


def train(model, train_loader, eval_loader=None, hyperparas=None, output_path=config.result_output_path):
    num_epoches = hyperparas['num_epoches']
    lr = hyperparas['lr']
    gpu = hyperparas['gpu']
    is_eval = hyperparas['is_eval']
    norm_lambda = hyperparas['norm_lambda']
    weight_decay = hyperparas['weight_decay']

    if gpu:
        model = model.cuda()

    optimizer = optim.Adamax(model.parameters(), lr=lr, weight_decay=weight_decay)

    utils.create_dir(output_path)
    logger = utils.Logger(os.path.join(output_path, 'log.txt'), hyperparas)
    best_rmse = None

    for epoch in range(num_epoches):
        torch.cuda.empty_cache()
        total_loss = 0.
        train_mse_total = 0.

        start_time = time.time()
        for i, (user_id, item_id, user_reviews, item_reviews, user_rids, item_rids, rating) in enumerate(train_loader):
            sys.stdout.write('\rtrain: %d / %d' % (i+1, len(train_loader)))

            if gpu:
                user_id = Variable(user_id).cuda()
                item_id = Variable(item_id).cuda()
                user_reviews = Variable(user_reviews).cuda()
                item_reviews = Variable(item_reviews).cuda()
                user_rids = Variable(user_rids).cuda()
                item_rids = Variable(item_rids).cuda()
                rating = Variable(rating).cuda()
            else:
                user_id = Variable(user_id)
                item_id = Variable(item_id)
                user_reviews = Variable(user_reviews)
                item_reviews = Variable(item_reviews)
                user_rids = Variable(user_rids)
                item_rids = Variable(item_rids)
                rating = Variable(rating)

            rating_pred, loss = model(user_id, item_id, user_reviews, item_reviews, user_rids, item_rids, rating, norm_lambda)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * rating.size(0)
            if (i+1) % 100 == 0:
                sys.stdout.write('\r')
                print('Epoch %d, iter %05d, loss = %.6f' % (epoch + 1, (i+1), loss.item()))

        loss = total_loss / len(train_loader.dataset)
        train_mse = loss

        if is_eval:
            # 验证集
            model.train(False)
            val_mse, val_rmse = evaluate(model, eval_loader, gpu)
            model.train(True)

            if best_rmse is None:
                best_rmse = val_rmse

            if best_rmse > val_rmse:
                model_path = os.path.join(output_path, 'model.pth')
                torch.save(model.state_dict(), model_path)
                best_rmse = val_rmse

        logger.write('Epoch %d, loss = %.6f, time: %.2f' % (epoch + 1, loss, time.time() - start_time))
        logger.write('\ttrain_mse = %.6f, train_rmse = %.6f' % (train_mse, np.sqrt(train_mse)))
        if is_eval:
            logger.write('\tval_mse = %.6f, val_rmse = %.6f, best_rmse = %.6f' % (val_mse, val_rmse, best_rmse))


def evaluate(model, eval_loader, gpu, save_att=False):
    total_mse = 0

    for i, (user_id, item_id, user_reviews, item_reviews, user_rids, item_rids, rating) in enumerate(eval_loader):
        sys.stdout.write('eval: %d / %d     \r' % (i + 1, len(eval_loader)))

        if gpu:
            user_id = Variable(user_id).cuda()
            item_id = Variable(item_id).cuda()
            user_reviews = Variable(user_reviews).cuda()
            item_reviews = Variable(item_reviews).cuda()
            user_rids = Variable(user_rids).cuda()
            item_rids = Variable(item_rids).cuda()
            rating = Variable(rating).cuda()
        else:
            user_id = Variable(user_id)
            item_id = Variable(item_id)
            user_reviews = Variable(user_reviews)
            item_reviews = Variable(item_reviews)
            user_rids = Variable(user_rids)
            item_rids = Variable(item_rids)
            rating = Variable(rating)

        rating_pred, _ = model(user_id, item_id, user_reviews, item_reviews, user_rids, item_rids, rating, save_att=save_att)
        mse = MSELoss(rating_pred, rating)
        total_mse += mse.item() * rating.size(0)

    mse = total_mse / len(eval_loader.dataset)
    rmse = np.sqrt(mse)
    return mse, rmse


class MyLoss(nn.Module):
    def __init__(self):
        super(MyLoss, self).__init__()

    def forward(self, model, prediction, rating, norm_lambda):

        l2_loss = 0.
        l2_loss += utils.l2_loss(model.user_reviews_att.review_linear.weight)
        l2_loss += utils.l2_loss(model.user_reviews_att.id_linear.weight)
        l2_loss += utils.l2_loss(model.item_reviews_att.review_linear.weight)
        l2_loss += utils.l2_loss(model.item_reviews_att.id_linear.weight)

        loss = utils.l2_loss(prediction, rating)

        return loss + norm_lambda * l2_loss