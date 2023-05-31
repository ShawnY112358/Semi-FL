import torch
import numpy as np
import torch.nn.functional as F
import torch.optim as optim

import wideresnet
from model import load_model
import json
from data_loader import load_dataset
import config

conf = config.conf

class semi_Server():
    def __init__(self):
        self.clients = []

        self.model= load_model().to(conf.device)
        # self.model = wideresnet.WideResNet(num_classes=conf.num_classes).to(conf.device)

        self.test_set = load_dataset(name=conf.dataset, is_train=False)
        self.test_loader = torch.utils.data.DataLoader(self.test_set, batch_size=conf.batchsize, shuffle=False)

        x = torch.load('./data/proxy_data.pt')
        y = torch.load('./data/proxy_label.pt').long()
        self.proxy_data = [(x[i], y[i]) for i in range(x.shape[0])]

        self.test_acc = []

    def init_classifier(self):
        self.classifier = load_classifier().to(conf.device)

    def aggregate(self, finetune=False):

        # for key in self.extractor.state_dict().keys():
        #     if 'num_batches_tracked' in key:
        #         self.extractor.state_dict()[key].data.copy_(self.clients[0].extractor.state_dict()[key])
        #         continue
        #     temp = torch.zeros_like(self.extractor.state_dict()[key]).to(conf.device)
        #     N = 0
        #     for i in range(len(self.clients)):
        #         temp += self.clients[i].num_data * self.clients[i].extractor.state_dict()[key]
        #         N += self.clients[i].num_data
        #     self.extractor.state_dict()[key].data.copy_(temp / N)
        #
        # for key in self.classifier.state_dict().keys():
        #     if 'num_batches_tracked' in key:
        #         self.classifier.state_dict()[key].data.copy_(self.clients[0].classifier.state_dict()[key])
        #         continue
        #     temp = torch.zeros_like(self.classifier.state_dict()[key]).to(conf.device)
        #     N = 0
        #     for i in range(len(self.clients)):
        #         temp += self.clients[i].num_data * self.clients[i].classifier.state_dict()[key]
        #         N += self.clients[i].num_data
        #     self.classifier.state_dict()[key].data.copy_(temp / N)

        for key in self.model.state_dict().keys():
            if 'num_batches_tracked' in key:
                self.model.state_dict()[key].data.copy_(self.clients[0].model.state_dict()[key])
                continue
            temp = torch.zeros_like(self.model.state_dict()[key]).to(conf.device)
            N = 0
            for i in range(len(self.clients)):
                temp += self.clients[i].num_data * self.clients[i].model.state_dict()[key]
                N += self.clients[i].num_data
            self.model.state_dict()[key].data.copy_(temp / N)

        dataloader = torch.utils.data.DataLoader(self.proxy_data, batch_size=conf.batchsize, shuffle=False, drop_last=False)

        self.proxy_labels = []
        with torch.no_grad():
            for i, data in enumerate(dataloader):
                inputs, _ = data
                inputs = inputs.to(conf.device)
                output = self.model(inputs)
                _, predicted = torch.max(output.data, 1)
                self.proxy_labels.append(predicted)
        self.proxy_labels = torch.cat(self.proxy_labels, dim=0)
        self.proxy_data = [(self.proxy_data[i][0], self.proxy_labels[i]) for i in range(len(self.proxy_data))]
        self.proxy_data.sort(key=lambda i: i[1])


        self.data_dict = dict()
        for i in range(conf.num_classes):
            self.data_dict[i] = []
        start = 0
        for i in range(1, len(self.proxy_data)):
            if self.proxy_data[i][1] != self.proxy_data[i - 1][1]:
                self.data_dict[self.proxy_data[i - 1][1].item()] = self.proxy_data[start: i]
                start = i
        self.data_dict[self.proxy_data[-1][1].item()] = self.proxy_data[start:]

    def test(self):
        # FedAvg
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data in self.test_loader:
                inputs, labels = data
                inputs, labels = inputs.to(conf.device), labels.to(conf.device)
                output = self.model(inputs)
                _, predicted = torch.max(output.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        print('accuracy:%d/%d = %f' % (correct, total, correct / total))
        self.test_acc.append(correct / total)
        with open('./Semi/log/test_acc.txt', 'w') as fp:
            json.dump(self.test_acc, fp)