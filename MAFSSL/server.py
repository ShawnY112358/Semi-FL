import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from model import load_extractor, load_classifier
import json
from data_loader import load_dataset
import config
import copy

conf = config.conf

class mafssl_Server():
    def __init__(self):
        self.clients = []

        self.extractor = load_extractor().to(conf.device)
        self.classifier = load_classifier().to(conf.device)

        self.test_set = load_dataset(name=conf.dataset, is_train=False)
        self.test_loader = torch.utils.data.DataLoader(self.test_set, batch_size=conf.batchsize, shuffle=False)

        self.test_acc = []
        self.extractor_list, self.classifier_list, self.model_list = [], [], []
        self.prototype = nn.Embedding(conf.num_classes, conf.prototype_size).weight

    def init_classifier(self):
        self.classifier = load_classifier().to(conf.device)

    def aggregate(self):
        self.extractor_list, self.classifier_list, self.model_list = [], [], []
        for client in self.clients:
            self.extractor_list.append(copy.deepcopy(client.extractor))
            self.classifier_list.append(copy.deepcopy(client.classifier))
            self.model_list.append(copy.deepcopy(client.model))

        prototype = torch.zeros_like(self.prototype)
        for c in range(conf.num_classes):
            N = 0
            for client in self.clients:
                if client.class_count[c] != 0:
                    prototype[c] += client.feature[c] * client.class_count[c]
                    N += client.class_count[c]
            if N != 0:
                self.prototype[c] = prototype[c] / N

        for key in self.extractor.state_dict().keys():
            if 'num_batches_tracked' in key:
                self.extractor.state_dict()[key].data.copy_(self.clients[0].extractor.state_dict()[key])
                continue
            temp = torch.zeros_like(self.extractor.state_dict()[key]).to(conf.device)
            N = 0
            for i in range(len(self.clients)):
                temp += self.clients[i].num_data * self.clients[i].extractor.state_dict()[key]
                N += self.clients[i].num_data
            self.extractor.state_dict()[key].data.copy_(temp / N)

        for key in self.classifier.state_dict().keys():
            if 'num_batches_tracked' in key:
                self.classifier.state_dict()[key].data.copy_(self.clients[0].classifier.state_dict()[key])
                continue
            temp = torch.zeros_like(self.classifier.state_dict()[key]).to(conf.device)
            N = 0
            for i in range(len(self.clients)):
                temp += self.clients[i].num_data * self.clients[i].classifier.state_dict()[key]
                N += self.clients[i].num_data
            self.classifier.state_dict()[key].data.copy_(temp / N)

    def test(self):
        self.extractor.eval()
        self.classifier.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data in self.test_loader:
                inputs, labels = data
                inputs, labels = inputs.to(conf.device), labels.to(conf.device)
                output = self.classifier(self.extractor(inputs))
                _, predicted = torch.max(output.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        print('accuracy:%d/%d = %f' % (correct, total, correct / total))
        self.test_acc.append(correct / total)
        with open('./MAFSSL/log/test_acc.txt', 'w') as fp:
            json.dump(self.test_acc, fp)