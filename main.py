import argparse
import dgl
import torch
from trainflow.link_prediction import LinkPrediction
from utils.logger import Logger
import pandas as pd
import numpy as np
import os
if __name__ == '__main__':
    if not os.path.exists("./output"):
        os.mkdir("./output")
    if not os.path.exists("./checkpoints"):
        os.mkdir("./checkpoints")
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', default='CompGCN', type=str, help='name of models')
    parser.add_argument('--task', '-t', default='link_prediction', type=str, help='name of task')
    # link_prediction / node_classification
    parser.add_argument('--dataset', '-d', default='demo_graph', type=str, help='name of datasets')
    # parser.add_argument('--gpu', '-g', default='0', type=int, help='-1 means cpu')
    parser.add_argument('--use_best_config', action='store_true', help='will load utils.best_config')
    parser.add_argument('--load_from_pretrained', action='store_true', help='load model from the checkpoint')
    # parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
    # parser.add_argument('--max_epoch', default=500, type=int, help='max training epoch')

    args = parser.parse_args()
    best_config = {
        'seed': 0, 
        'validation': True, 
        # 'evaluation_metric': 'acc',
        'lr': 0.01, 
        'weight_decay': 0.0001, 
        'max_epoch': 500,
        'hidden_dim': 32, 
        'num_layers': 2, 
        'dropout': 0.2, 
        'comp_fn': 'sub', 
        'batch_size': 128,
        'patience': 100,
        'mini_batch_flag': False,
        "output_dir": "./checkpoints",
        "model_name": "CompGCN",
        "dataset_name": "demo_graph",
        "task": "link_prediction",
        "device": "cuda:0" if torch.cuda.is_available() else 'cpu',
        'optimizer': 'Adam'
    }
    if args.use_best_config:
        for key, value in best_config.items():
            args.__setattr__(key, value)

    torch.cuda.empty_cache()
    logger = Logger(args)
    args.logger = logger
    flow = LinkPrediction(args)
    flow.train()

    # Get csv file
    score = np.load("./output/score.npy")
    labels = score >= 0.5
    indexes = np.arange(len(labels)).reshape((-1,1))
    labels = labels.astype(np.int).reshape((-1,1))
    labels_df = pd.DataFrame(np.concatenate([indexes, labels],axis=1), columns=["Index", "Predicted"])
    labels_df.to_csv("./output/CompGCN_results.csv", index=False, header=True)
