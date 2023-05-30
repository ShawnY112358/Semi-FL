import shutil
import data_loader
from data_loader import save_data, iid, load_from_file
import config
import torch
import numpy as np
import os
import argparse
import random

conf = config.conf

def generate_data(partition='n_way_k_shot'):
    if os.path.exists('./data'):
        shutil.rmtree('./data')
    os.mkdir('./data')

    train_set, test_set = data_loader.load_dataset(name=conf.dataset), data_loader.load_dataset(name=conf.dataset)
    train_set = [train_set[i] for i in range(len(train_set))]
    random.shuffle(train_set)

    proxy_dataset = train_set[: conf.num_proxy_data]
    x = torch.stack([proxy_dataset[i][0] for i in range(len(proxy_dataset))], dim=0)
    y = torch.from_numpy(np.array([proxy_dataset[i][1] for i in range(len(proxy_dataset))]))
    torch.save(x, './data/proxy_data.pt')
    torch.save(y, './data/proxy_label.pt')

    train_set = train_set[conf.num_proxy_data: ]

    if partition == 'n_way_k_shot':
        train_sets, test_sets = data_loader.n_way_k_shot(train_set, test_set)
    elif partition == 'pathological':
        train_sets, test_sets = data_loader.pathological(train_set, test_set)
    elif partition == 'iid':
        train_sets, test_sets = data_loader.iid(train_set, test_set)

    for i, data in enumerate(zip(train_sets, test_sets)):
        train_data, test_data = data
        random.shuffle(train_data)
        data_loader.save_data(train_data, test_data, i)

# parser = argparse.ArgumentParser()
# parser.add_argument('-p', '--partition', help='partitioning methods')
# args = parser.parse_args()
generate_data(conf.partition)