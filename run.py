import json

import data_loader
from data_loader import save_data, iid, load_from_file
import os
import config
# from FedAvg.server import avg_Server
# from FedAvg.client import avg_Client
# from Local.client import local_Client
from Semi.client import semi_Client
from Semi.server import semi_Server
# from MAFSSL.server import mafssl_Server
# from MAFSSL.client import mafssl_Client
import random
import shutil
import torch
conf = config.conf

def init_Local():
    if os.path.exists('./Local/log'):
        shutil.rmtree("./Local/log")
    os.mkdir('./Local/log')
    if os.path.exists('./Local/model'):
        shutil.rmtree("./Local/model")
    os.mkdir('./Local/model')

    clients = []
    for i in range(conf.num_clients):
        train_data, test_data = load_from_file(i)
        client = local_Client(i, train_data, test_data)
        clients.append(client)

    return clients

def run_Local():
    clients = init_Local()
    for g_epoch in range(conf.nums_g_epoch):
        for client in clients:
            client.train()
        for client in clients:
            client.test()

    avg_acc = [sum([client.test_acc[i] for client in clients]) / len(clients) for i in range(len(clients[0].test_acc))]
    with open('./Local/log/test_acc_avg.txt', 'w') as fp:
        json.dump(avg_acc, fp=fp)


def init_FedAvg():

    if os.path.exists('./FedAvg/log'):
        shutil.rmtree("./FedAvg/log")
    os.mkdir('./FedAvg/log')
    if os.path.exists('./FedAvg/model'):
        shutil.rmtree("./FedAvg/model")
    os.mkdir('./FedAvg/model')

    clients = []
    server = avg_Server()

    for i in range(conf.num_clients):
        train_data, test_data = load_from_file(i)
        client = avg_Client(i, train_data, test_data, server)
        server.clients.append(client)
        clients.append(client)

    return clients, server

def run_FedAvg(finetune=False):
    clients, server = init_FedAvg()
    for g_epoch in range(conf.nums_g_epoch):
        # for client in clients:
        #     client.down_model()
        group = random.sample(clients, int(conf.num_clients * conf.select_rate))
        for client in group:
            print("global_epoch: %d" % g_epoch)
            client.train()
        server.aggregate()
        server.test()
    torch.save(server.extractor, './FedAvg/model/extractor.pt')
    torch.save(server.classifier, './FedAvg/model/classifier.pt')

def init_Semi():

    if os.path.exists('./Semi/log'):
        shutil.rmtree("./Semi/log")
    os.mkdir('./Semi/log')
    if os.path.exists('./Semi/model'):
        shutil.rmtree("./Semi/model")
    os.mkdir('./Semi/model')

    clients = []
    server = semi_Server()

    for i in range(conf.num_clients):
        train_data, test_data = load_from_file(i)
        client = semi_Client(i, train_data, test_data, server)
        server.clients.append(client)
        clients.append(client)

    return clients, server

def run_Semi():
    clients, server = init_Semi()
    server.aggregate()
    for g_epoch in range(conf.nums_g_epoch):
        for client in clients:
            client.down_model()
        group = random.sample(clients, int(conf.num_clients * conf.select_rate))
        # print("global_epoch: %d" % g_epoch)
        # clients[0].train(g_epoch)
        for client in group:
            print("global_epoch: %d" % g_epoch)
            client.train(g_epoch)
        server.aggregate()
        server.test()
    torch.save(server.extractor, './Semi/model/extractor.pt')
    torch.save(server.classifier, './Semi/model/classifier.pt')

def init_MAFSSL():

    if os.path.exists('./MAFSSL/log'):
        shutil.rmtree("./MAFSSL/log")
    os.mkdir('./MAFSSL/log')
    if os.path.exists('./MAFSSL/model'):
        shutil.rmtree("./MAFSSL/model")
    os.mkdir('./MAFSSL/model')

    clients = []
    server = mafssl_Server()

    for i in range(conf.num_clients):
        train_data, test_data = load_from_file(i)
        client = mafssl_Client(i, train_data, test_data, server)
        server.clients.append(client)
        clients.append(client)

    return clients, server

def run_MAFSSL(finetune=False):
    clients, server = init_MAFSSL()
    server.aggregate()
    for g_epoch in range(conf.nums_g_epoch):
        for client in clients:
            client.down_model()
        group = random.sample(clients, int(conf.num_clients * conf.select_rate))
        for client in group:
            print("global_epoch: %d" % g_epoch)
            client.train(g_epoch=g_epoch)
        server.aggregate()
    torch.save(server.extractor, './MAFSSL/model/extractor.pt')
    torch.save(server.classifier, './MAFSSL/model/classifier.pt')