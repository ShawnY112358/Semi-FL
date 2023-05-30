import copy
import random

from model import load_extractor, load_classifier
import torch
import numpy as np
import data_loader
from data_loader import load_dataset, RandomFlip, RandomPadandCrop, ToTensor
import torch.optim as optim
import torch.nn as nn
import config
import json
conf = config.conf
import torchvision.transforms as transforms
import wideresnet
from utils import accuracy

augment = transforms.Compose([
    RandomPadandCrop(32),
    RandomFlip(),
    ToTensor(),
])

def linear_rampup(current, rampup_length):
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current / rampup_length, 0.0, 1.0)
        return float(current)


def interleave_offsets(batch, nu):
    groups = [batch // (nu + 1)] * (nu + 1)  # 一个batch分为batch_size个块，一共nu + 1个batch
    for x in range(batch - sum(groups)):  # 正常整除的话batch-sum(groups)=0,但是由于batch // (nu + 1)向下取整
        groups[-x - 1] += 1
    # 相当于想把第一个batch中的数据均分给所有batch，但是不能整除，剩了一些，剩的分给末尾的那些batch，一人一个

    offsets = [0]
    for g in groups:
        offsets.append(offsets[-1] + g)
    # offsets[i]表示：第i个batch可以分到的数据编号从offsets[i]开始
    assert offsets[-1] == batch
    return offsets


def interleave(xy, batch):
    nu = len(xy) - 1  # len(xy) 为batch数
    offsets = interleave_offsets(batch, nu)
    xy = [[v[offsets[p]:offsets[p + 1]] for p in range(nu + 1)] for v in xy]
    for i in range(1, nu + 1):
        xy[0][i], xy[i][i] = xy[i][i], xy[0][i]
    # 先把每个batch分成同样规格的块，再把第一个batch中的块穿插到其他batch中，也就是把带标签的块穿插到不带标签的batch中
    return [torch.cat(v, dim=0) for v in xy]

class Data(torch.utils.data.Dataset):
    def __init__(self, data, k):
        self.data = data
        self.num_augment = k

    def __getitem__(self, item):
        img, label = self.data[item][0], self.data[item][1]

        label = nn.functional.one_hot(torch.tensor(label), num_classes=conf.num_classes)
        if self.num_augment == 2:
            return augment(img), augment(img), label
        else:
            return augment(img), label

    def __len__(self):
        return len(self.data)

class mafssl_Client():
    def __init__(self, index, data, test_data, server):
        self.index = index
        self.extractor = load_extractor().to(conf.device)
        self.classifier = load_classifier().to(conf.device)
        self.model = wideresnet.WideResNet(num_classes=conf.num_classes).to(conf.device)

        self.labeled_data = Data(data[: int(conf.labeled_rate * len(data))], k=1)
        self.unlabeled_data = Data(data[int(conf.labeled_rate * len(data)): ], k=2)

        self.test_data = test_data
        # TODO: num_data = ?
        self.num_data = len(self.labeled_data)
        self.server = server
        self.num_l_epochs = conf.l_epoch
        self.batch_size = conf.batchsize
        self.lr = conf.learning_rate
        self.test_acc = []
        self.grad = []

    def pseudo_label(self, data):
        # outputs of classifier
        with torch.no_grad():
            output = torch.zeros_like(data[-1]).float()
            for j in range(len(data) - 1):
                for model in self.server.model_list:
                    input = data[j].to(conf.device)
                    output += torch.softmax(model(input), dim=1).cpu()
                # for extractor, classifier in zip(self.server.extractor_list, self.server.classifier_list):
                #     input = data[j].to(conf.device)
                #     output += torch.softmax(classifier(extractor(input)), dim=1).cpu()
            p = output / (len(data) - 1) / len(self.server.extractor_list)
            pt = p ** (1 / conf.T)
            targets = pt / pt.sum(dim=1, keepdim=True)
            targets = targets.detach()

        # # outputs of FedAvg
        # with torch.no_grad():
        #     output = torch.zeros_like(data[-1]).float()
        #     for j in range(len(data) - 1):
        #         input = data[j].to(conf.device)
        #         output += torch.softmax(self.classifier(self.extractor(input)), dim=1).cpu()
        #     p = output / (len(data) - 1)
        #     pt = p ** (1 / conf.T)
        #     targets = pt / pt.sum(dim=1, keepdim=True)
        #     targets = targets.detach()

        return targets

    def cal_feature(self):
        feature = []
        self.class_count = [0 for i in range(conf.num_classes)]
        x = torch.tensor([self.labeled_data[i][0].numpy() for i in range(len(self.labeled_data))]).to(conf.device)
        y = torch.tensor([self.labeled_data[i][1] for i in range(len(self.labeled_data))]).to(conf.device)
        with torch.no_grad():
            output, _ = self.extractor(x).cpu()
            for i in range(conf.num_classes):
                f = torch.zeros(conf.prototype_size)
                count = 0
                for j in range(len(y)):
                    if y[j].cpu().item() == i:
                        f += output[j]
                        count += 1
                self.class_count[i] = count
                feature.append((f / count).numpy()) if count != 0 else feature.append(None)
        self.feature = feature


    def train(self, g_epoch):
        print("mafssl:")
        self.extractor.train()
        self.classifier.train()
        optimizer_e = optim.SGD(self.extractor.parameters(), lr=self.lr)
        optimizer_c = optim.SGD(self.classifier.parameters(), lr=self.lr)
        optimizer_m = optim.Adam(self.model.parameters(), lr=self.lr)


        criterion_x = nn.CrossEntropyLoss().to(conf.device)
        criterion_u = nn.MSELoss().to(conf.device)
        # criterion = nn.CrossEntropyLoss(weight=self.server.sample_weight).to(conf.device)   # sample re-weighting

        self.loss_avg = []
        l_trainloader = torch.utils.data.DataLoader(self.labeled_data, batch_size=self.batch_size, shuffle=True, drop_last=True)
        u_trainloader = torch.utils.data.DataLoader(self.unlabeled_data, batch_size=self.batch_size, shuffle=True, drop_last=True)
        l_iter, u_iter = iter(l_trainloader), iter(u_trainloader)
        for l_epoch in range(self.num_l_epochs):
            avg_loss_x, avg_loss_u = 0, 0
            # for i, (l_data, u_data) in enumerate(zip(l_trainloader, u_trainloader)):
            for i in range(conf.l_iteration_per_epoch):
                try:
                    l_data = l_iter.__next__()
                except:
                    l_iter = iter(l_trainloader)
                    l_data = l_iter.__next__()
                try:
                    u_data = u_iter.__next__()
                except:
                    u_iter = iter(u_trainloader)
                    u_data = u_iter.__next__()

                optimizer_e.zero_grad()
                optimizer_c.zero_grad()
                optimizer_m.zero_grad()

                batch_size = l_data[0].shape[0]
                l_xs, l_ys = torch.cat([l_data[j] for j in range(len(l_data) - 1)], dim=0), torch.cat([l_data[-1]] * (len(l_data) - 1), dim=0)


                # predict label
                targets = self.pseudo_label(u_data)
                u_xs = torch.cat([u_data[j] for j in range(len(u_data) - 1)], dim=0)
                u_ys = torch.cat([targets] * (len(u_data) - 1), dim=0)
                # mixup
                w_data, w_label = torch.cat([copy.deepcopy(l_xs), copy.deepcopy(u_xs)], dim=0), torch.cat([copy.deepcopy(l_ys), copy.deepcopy(u_ys)], dim=0)
                idx = torch.randperm(w_data.shape[0])
                w_data, w_label = w_data[idx], w_label[idx]

                l = np.random.beta(conf.alpha, conf.alpha)
                l = max(l, 1 - l)

                mixed_data_x, mixed_label_x = l * l_xs + (1 - l) * w_data[: l_xs.shape[0]], l * l_ys + (1 - l) * w_label[: l_ys.shape[0]]
                mixed_data_u, mixed_label_u = l * u_xs + (1 - l) * w_data[l_xs.shape[0]: ], l * u_ys + (1 - l) * w_label[l_ys.shape[0]: ]
                mixed_data_x, mixed_label_x, mixed_data_u, mixed_label_u = mixed_data_x.to(conf.device), mixed_label_x.to(conf.device), mixed_data_u.to(conf.device), mixed_label_u.to(conf.device)

                # interleave labeled and unlabed samples between batches to get correct batchnorm calculation
                mixed_input = torch.cat([mixed_data_x, mixed_data_u], dim=0)

                mixed_input = list(torch.split(mixed_input, batch_size))
                mixed_input = interleave(mixed_input, batch_size)

                # logits = [self.classifier(self.extractor(mixed_input[0]))]
                # for input in mixed_input[1:]:
                #     logits.append(self.classifier(self.extractor(input)))
                logits = [self.model(mixed_input[0])]
                for input in mixed_input[1:]:
                    logits.append(self.model(input))

                # put interleaved samples back
                logits = interleave(logits, batch_size)
                logits_x = logits[0]
                logits_u = torch.cat(logits[1:], dim=0)

                # loss_x = criterion_x(logits_x, mixed_label_x)
                # loss_u = criterion_u(torch.softmax(logits_u, dim=1), mixed_label_u)
                loss_u = torch.mean((torch.softmax(logits_u, dim=1) - mixed_label_u)**2)
                loss_x = -torch.mean(torch.sum(nn.functional.log_softmax(logits_x, dim=1) * mixed_label_x, dim=1))
                # TODO: dynamic lambda_u, rampup_length?
                lambda_u = conf.lambda_u * linear_rampup(self.num_l_epochs * g_epoch + l_epoch + i / conf.l_iteration_per_epoch, self.num_l_epochs * conf.nums_g_epoch)
                loss = loss_x + lambda_u * loss_u

                loss.backward()
                # optimizer_e.step()
                # optimizer_c.step()
                optimizer_m.step()

                avg_loss_x += loss_x.cpu().item()
                avg_loss_u += loss_u.cpu().item()

            if (l_epoch + 1) % 10 == 0:
                self.test()
            avg_loss_x /= conf.l_iteration_per_epoch
            avg_loss_u /= conf.l_iteration_per_epoch
            print("client: %d\t epoch: (%d/%d)\t loss_x/loss_u: %f/%f"
                  % (self.index, l_epoch, self.num_l_epochs, avg_loss_x, avg_loss_u))
            self.loss_avg.append(avg_loss_x + avg_loss_u)

        self.test()

    def down_model(self):
        # for key in self.extractor.state_dict().keys():
        #     self.extractor.state_dict()[key].data.copy_(self.server.extractor.state_dict()[key])
        #
        # for key in self.classifier.state_dict().keys():
        #     self.classifier.state_dict()[key].data.copy_(self.server.classifier.state_dict()[key])
        pass


    def test(self):
        self.extractor.eval()
        self.classifier.eval()
        self.model.eval()

        test_loader = torch.utils.data.DataLoader(self.test_data, batch_size=self.batch_size, shuffle=True)
        total = 0
        correct = 0
        with torch.no_grad():
            for data in test_loader:
                inputs, labels = data
                inputs, labels = inputs.to(conf.device), labels.to(conf.device)
                output = self.model(inputs)
                # output = self.classifier(self.extractor(inputs))
                _, predicted = torch.max(output.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        print('accuracy:%d/%d = %f' % (correct, total, correct / total))
        self.test_acc.append(correct / total)
        with open('./MAFSSL/log/test_acc_%d.txt' % self.index, 'w') as fp:
            json.dump(self.test_acc, fp)
        self.extractor.train()
        self.classifier.train()
        self.model.train()

    def save_model(self):
        torch.save(self.extractor, './MAFSSL/model/extractor_%d.pt' % self.index)
        torch.save(self.classifier, './MAFSSL/model/classifier_%d.pt' % self.index)