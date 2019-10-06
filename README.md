# HANCI
---

## Environments
- python 3.6.6
- pytorch 1.0.1
- numpy
- pandas
- json

## Dataset
### Datasets url:
- [Amazon 5-core](http://jmcauley.ucsd.edu/data/amazon/)
- [Yelp Challenge 2017](https://www.yelp.com/dataset)

### Word corpus url:
- [word2vec](https://code.google.com/archive/p/word2vec)
- [GloVe](https://nlp.stanford.edu/projects/glove)

### Data diretory structure
```sh
├── data
│   ├── glove
│   │   └── GoogleNews-vectors-negative300.bin
│   ├── instrument
│   │   └── reviews_Musical_Instruments_5.json
│   └── stopwords.txt
```

## Load data
```sh
python load_data.py
```

files
```sh
├── data
│   ├── glove
│   │   └── GoogleNews-vectors-negative300.bin
│   ├── instrument
│   │   ├── item_reviews
│   │   │   ├── 0.json
│   │   │   ├── 1.json
│   │   │   └── ...
│   │   ├── review_num_len.json
│   │   ├── reviews_Musical_Instruments_5.json
│   │   ├── test.csv
│   │   ├── train.csv
│   │   ├── user_item_num.json
│   │   ├── user_reviews
│   │   │   ├── 0.json
│   │   │   ├── 1.json
│   │   │   └── ...
│   │   └── val.csv
│   └── stopwords.txt
```

## Preprocess
```sh
python create_dictionary.py
```

```sh
├── data
│   ├── glove
│   │   └── GoogleNews-vectors-negative300.bin
│   ├── instrument
│   │   ├── dictionary.pkl
│   │   ├── glove.init.npy
│   │   ├── item_reviews
│   │   │   ├── 0.json
│   │   │   ├── 1.json
│   │   │   └── ...
│   │   ├── item_reviews_token
│   │   │   ├── 0.json
│   │   │   ├── 1.json
│   │   │   └── ...
│   │   ├── review_num_len.json
│   │   ├── reviews_Musical_Instruments_5.json
│   │   ├── test.csv
│   │   ├── train.csv
│   │   ├── user_item_num.json
│   │   ├── user_reviews
│   │   │   ├── 0.json
│   │   │   ├── 1.json
│   │   │   └── ...
│   │   ├── user_reviews_token
│   │   │   ├── 0.json
│   │   │   ├── 1.json
│   │   │   └── ...
│   │   └── val.csv
│   └── stopwords.txt
```

## Train
```sh
python main.py
```

```text
loading dictionary from ../data/instrument/dictionary.pkl
loading train dataset
reading ../data/instrument/train.csv
loading val dataset
reading ../data/instrument/val.csv
=============hyperparas=============
gpu: True
is_eval: True
model: HANCI
embedding_size: 300
id_embedding_size: 32
attention_size: 32
num_latent: 32
dropout: 0.5
num_epoches: 20
batch_size: 100
lr: 0.01
norm_lambda: 0.1
weight_decay: 0.002
filter_sizes: [3]
num_filters: 100
hidden_size: 100
num_layers: 2
bidirectional: True
da: 100
r: 10
soa_size: 50
====================================

Epoch 1, loss = 431.957176, time: 25.83
	train_mse = 431.957176, train_rmse = 20.783579
	val_mse = 3.154101, val_rmse = 1.775979, best_rmse = 1.775979
Epoch 2, loss = 178.305362, time: 25.81
	train_mse = 178.305362, train_rmse = 13.353103
	val_mse = 2.509649, val_rmse = 1.584187, best_rmse = 1.584187
```
