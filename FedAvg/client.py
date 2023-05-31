import wideresnet
from model import load_model
import torch
import numpy as np
from data_loader import load_dataset
import torch.optim as optim
import torch.nn as nn
import config
import json
conf = config.conf
import random
import data_loader

class avg_Client():
    def __init__(self, index, data, test_data, server):
        self.index = index
        data = data_loader.load_dataset(name='cifar', is_train=True)
        random.shuffle(data)

        self.labeled_data = data
        self.test_data = test_data

        self.num_data = len(self.labeled_data)
        self.server = server

        self.model = load_model().to(conf.device)
        self.num_l_epochs = conf.l_epoch
        self.batch_size = conf.batchsize
        self.lr = conf.learning_rate
        self.trainloader = torch.utils.data.DataLoader(self.labeled_data, batch_size=self.batch_size, shuffle=True)
        self.train_iter = iter(self.trainloader)

        self.test_acc = []
        self.grad = []


    def train(self):
        print("FedAvg:")
        self.model.train()
        optimizer= optim.SGD(self.model.parameters(), lr=self.lr)
        criterion = nn.CrossEntropyLoss().to(conf.device)
        # criterion = nn.CrossEntropyLoss(weight=self.server.sample_weight).to(conf.device)   # sample re-weighting

        for l_iter in range(conf.num_l_iteration):
            try:
                l_data = self.train_iter.__next__()
            except:
                self.train_iter = iter(self.trainloader)
                l_data = self.train_iter.__next__()
            x, y = l_data
            optimizer.zero_grad()
            x, y = x.to(conf.device), y.to(conf.device)
            output = self.model(x)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()

            print("client: %d\t iteration: (%d/%d)\t loss: %f"
                  % (self.index, l_iter, conf.num_l_iteration, loss.cpu().item()))

    def down_model(self):

        for key in self.model.state_dict().keys():
            self.model.state_dict()[key].data.copy_(self.server.extractor.state_dict()[key])


    def test(self):
        self.model.eval()

        test_loader = torch.utils.data.DataLoader(self.test_data, batch_size=self.batch_size, shuffle=True)
        total = 0
        correct = 0
        with torch.no_grad():
            for data in test_loader:
                inputs, labels = data
                inputs, labels = inputs.to(conf.device), labels.to(conf.device)
                # output = self.classifier(self.extractor(inputs))
                output = self.model(inputs)
                _, predicted = torch.max(output.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        print('accuracy:%d/%d = %f' % (correct, total, correct / total))
        self.test_acc.append(correct / total)
        with open('./FedAvg/log/test_acc_%d.txt' % self.index, 'w') as fp:
            json.dump(self.test_acc, fp)

        self.model.train()

    def save_model(self):
        torch.save(self.model, './FedAvg/model/model_%d.pt')