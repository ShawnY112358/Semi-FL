import torch
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
from model import load_model
import json
from data_loader import load_dataset
import config

conf = config.conf

class avg_Server():
    def __init__(self):
        self.clients = []

        self.model = load_model().to(conf.device)

        self.test_set = load_dataset(name=conf.dataset, is_train=False)
        self.test_loader = torch.utils.data.DataLoader(self.test_set, batch_size=conf.batchsize, shuffle=False)

        self.test_acc = []

    def aggregate(self, finetune=False):

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
        with open('./FedAvg/log/test_acc.txt', 'w') as fp:
            json.dump(self.test_acc, fp)