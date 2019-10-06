"""
Created on 2019/3/2 15:58

@author: zhouweixin
@note: 主函数
"""

from torch.utils.data import DataLoader
from core.dataset import Dictionary, NARREDataset
from core.train import evaluate
from core import build_model
import torch

hyperparas = {
    # config
    'gpu': True,
    'is_eval': True,

    # common
    # 模型选择(注：SA表示self-attention, FA表示feature-attention)
    'model': 'NARRE_LSTM_SA_SOA',  # NARRE, NARRE_FA, NARRE_SOA, NARRE_LSTM, NARRE_LSTM_SA, NARRE_LSTM_SA_SOA, NARRE_CNN_LSTM, NARRE_CNN_LSTM_SA
    'embedding_size': 300,
    'id_embedding_size': 32,
    'attention_size': 32,

    'num_latent': 32,
    'dropout': 0.5,
    'num_epoches': 20,
    'batch_size': 100,
    'lr': 0.01,
    'norm_lambda': 0.1,
    'weight_decay': 0.001,

    # cnn
    'filter_sizes': [3],
    'num_filters': 100,

    # lstm
    'hidden_size': 100,
    'num_layers': 1,
    'bidirectional': True,

    # self-attention
    'da': 100,
    'r': 10,

    # soft-attention
    'soa_size': 50,
}

dictionary = Dictionary.load_from_file()

train_dset = NARREDataset('train', dictionary)
# val_dset = NARREDataset('val', dictionary)
test_dset = NARREDataset('test', dictionary)

train_loader = DataLoader(train_dset, batch_size=hyperparas['batch_size'], shuffle=True)
# val_loader = DataLoader(val_dset, batch_size=hyperparas['batch_size'], shuffle=True)
test_loader = DataLoader(test_dset, batch_size=hyperparas['batch_size'], shuffle=True)
constructor = 'build_' + hyperparas['model']

model_path = '../../narre_all_data/toy_game/result/narre-lstm-sa-soa-0.8513/model.pth'
model = getattr(build_model, constructor)(train_dset, hyperparas)
model.load_state_dict(torch.load(model_path))
model.train(False)
# train(model, train_loader, val_loader, hyperparas)
model = model.cuda()
val_mse, val_rmse = evaluate(model, test_loader, True, save_att=True)
print(val_rmse)

