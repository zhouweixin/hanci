"""
Created on 2019/3/2 15:58

@author: zhouweixin
@note: 主函数
"""

from torch.utils.data import DataLoader

from core.dataset import Dictionary, NARREDataset
from core.train import train
from core import build_model
from core.parameters_analysis import compute_parameters, compute_fops, compute_fops1

hyperparas = {
    # config
    'gpu': True,
    'is_eval': True,

    # common
    # 模型选择(注：SA表示self-attention, FA表示feature-attention)
    'model': 'NARRE', # HANCI, NARRE_LSTM_SA, NARRE, NARRE_FA
    'embedding_size': 300,
    'id_embedding_size': 32,
    'attention_size': 32,

    'num_latent': 32,
    'dropout': 0.5,
    'num_epoches': 1,
    'batch_size': 1,
    'lr': 0.01,
    'norm_lambda': 0.1,
    'weight_decay': 0.002,

    # cnn
    'filter_sizes': [3],
    'num_filters': 100,

    # lstm
    'hidden_size': 100,
    'num_layers': 2,
    'bidirectional': True,

    # word-level attention
    'da': 100,
    'r': 10,

    # feature-level attention
    'soa_size': 50,
}

dictionary = Dictionary.load_from_file()

train_dset = NARREDataset('train', dictionary)
val_dset = NARREDataset('val', dictionary)

train_loader = DataLoader(train_dset, batch_size=hyperparas['batch_size'], shuffle=True)
val_loader = DataLoader(val_dset, batch_size=hyperparas['batch_size'], shuffle=True)

constructor = 'build_' + hyperparas['model']
model = getattr(build_model, constructor)(train_dset, hyperparas)

# train(model, train_loader, val_loader, hyperparas)

print(compute_parameters(model))
# print(model)
# compute_fops(model, train_loader)
# compute_fops1(model, train_loader)

from prepro.FLOPs_counter import print_model_parm_flops

for i, input in enumerate(train_loader):
    print_model_parm_flops(model, input, detail=True)
    break



