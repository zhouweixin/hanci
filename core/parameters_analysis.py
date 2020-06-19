"""
Created on 2020/3/25 12:44

@author: zhouweixin
@note: 
"""

import torchvision.models as models
import torch
from ptflops import get_model_complexity_info

def compute_parameters(model):
    print(model)
    params = list(model.parameters())
    total = 0
    for i in params:
        n = 1
        print("该层的结构：" + str(list(i.size())))
        for j in i.size():
            n *= j
        print("该层参数和：" + str(n))
        total += n

    return total

def compute_fops(model, train_loader):
    for i, (user_id, item_id, user_reviews, item_reviews, user_rids, item_rids, rating) in enumerate(train_loader):
        torch.cat([torch.tensor(user_id.size()), torch.tensor(item_id.size()), torch.tensor(user_reviews.size()),
                       torch.tensor(item_reviews.size()), torch.tensor(user_rids.size()), torch.tensor(item_rids.size()),
                   torch.tensor(rating.size()), torch.tensor([1])], 0)
        size = tuple(list(user_id.size()) + list(item_id.size()) + list(user_reviews.size()) + list(item_reviews.size()) + list(user_rids.size()) + list(item_rids.size()) + list(rating.size()) + [1])
        model = model
        flops, params = get_model_complexity_info(model, size, as_strings=True, print_per_layer_stat=True)
        # print('{:<30}  {:<8}'.format('Computational complexity: ', flops))
        # print('{:<30}  {:<8}'.format('Number of parameters: ', params))
        break


def compute_fops1(model, train_loader):
    from thop import profile

    for i, (user_id, item_id, user_reviews, item_reviews, user_rids, item_rids, rating) in enumerate(train_loader):
        hereflops, params = profile(model, (
        user_id, item_id, user_reviews, item_reviews, user_rids, item_rids, rating, 0.1))
        print(hereflops)
        print("*" * 20)
        print(params)
        break