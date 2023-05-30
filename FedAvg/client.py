import wideresnet
from model import load_extractor, load_classifier
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

        self.labeled_data = data[: int(conf.labeled_rate * len(data))]
        self.unlabeled_data = data[int(conf.labeled_rate * len(data)): ]
        self.test_data = test_data

        self.num_data = len(self.labeled_data)
        self.server = server

        self.extractor = load_extractor().to(conf.device)
        self.classifier = load_classifier().to(conf.device)
        self.num_l_epochs = conf.l_epoch
        self.batch_size = conf.batchsize
        self.lr = conf.learning_rate
        self.test_acc = []
        self.grad = []

    def train(self):
        print("FedAvg:")
        self.extractor.train()
        self.classifier.train()
        optimizer_e = optim.SGD(self.extractor.parameters(), lr=self.lr)
        optimizer_c = optim.SGD(self.classifier.parameters(), lr=self.lr)
        criterion = nn.CrossEntropyLoss().to(conf.device)
        # criterion = nn.CrossEntropyLoss(weight=self.server.sample_weight).to(conf.device)   # sample re-weighting

        self.loss_avg = []
        trainloader = torch.utils.data.DataLoader(self.labeled_data, batch_size=self.batch_size, shuffle=True)
        l_iter = iter(trainloader)
        for l_epoch in range(self.num_l_epochs):
            avg_loss = 0
            for i in range(conf.l_iteration_per_epoch):
                try:
                    l_data = l_iter.__next__()
                except:
                    l_iter = iter(trainloader)
                    l_data = l_iter.__next__()
                x, y = l_data
                optimizer_e.zero_grad()
                optimizer_c.zero_grad()
                x, y = x.to(conf.device), y.to(conf.device)
                # output = self.classifier(self.extractor(x))
                output = self.classifier(self.extractor(x))
                loss = criterion(output, y)
                loss.backward()
                optimizer_e.step()
                optimizer_c.step()

                avg_loss += loss.cpu().item()
            if (l_epoch + 1) % 10 == 0:
                self.test()
            avg_loss /= conf.l_iteration_per_epoch
            print("client: %d\t epoch: (%d/%d)\t loss: %f"
                  % (self.index, l_epoch, self.num_l_epochs, avg_loss))
            self.loss_avg.append(avg_loss)


    def down_model(self):

        for key in self.extractor.state_dict().keys():
            self.extractor.state_dict()[key].data.copy_(self.server.extractor.state_dict()[key])

        for key in self.classifier.state_dict().keys():
            self.classifier.state_dict()[key].data.copy_(self.server.classifier.state_dict()[key])


    def test(self):
        self.extractor.eval()
        self.classifier.eval()

        test_loader = torch.utils.data.DataLoader(self.test_data, batch_size=self.batch_size, shuffle=True)
        total = 0
        correct = 0
        with torch.no_grad():
            for data in test_loader:
                inputs, labels = data
                inputs, labels = inputs.to(conf.device), labels.to(conf.device)
                # output = self.classifier(self.extractor(inputs))
                output = self.classifier(self.extractor(inputs))
                _, predicted = torch.max(output.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        print('accuracy:%d/%d = %f' % (correct, total, correct / total))
        self.test_acc.append(correct / total)
        with open('./FedAvg/log/test_acc_%d.txt' % self.index, 'w') as fp:
            json.dump(self.test_acc, fp)

        self.extractor.train()
        self.classifier.train()

    def save_model(self):
        torch.save(self.extractor, './FedAvg/model/extractor_%d.pt' % self.index)
        torch.save(self.classifier, './FedAvg/model/classifier_%d.pt' % self.index)