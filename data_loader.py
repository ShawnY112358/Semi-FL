import os
from PIL import Image
import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import random
import config
import json
import sys

conf = config.conf


cifar10_mean = (0.4914, 0.4822, 0.4465) # equals np.mean(train_set.train_data, axis=(0,1,2))/255
cifar10_std = (0.2471, 0.2435, 0.2616) # equals np.std(train_set.train_data, axis=(0,1,2))/255
transform_cifar = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize(cifar10_mean, cifar10_std)])
transform_mnist = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5,), (0.5,))])

def normalize(x, mean=cifar10_mean, std=cifar10_std):
    x, mean, std = [np.array(a, np.float32) for a in (x, mean, std)]
    x -= mean*255
    x *= 1.0/(255*std)
    return x

def transpose(x, source='NHWC', target='NCHW'):
    return x.transpose([source.index(d) for d in target])

def pad(x, border=4):
    return np.pad(x, [(0, 0), (border, border), (border, border)], mode='reflect')

class RandomPadandCrop(object):
    """Crop randomly the image.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, x):
        x = pad(x, 4)

        h, w = x.shape[1:]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        x = x[:, top: top + new_h, left: left + new_w]

        return x

class RandomFlip(object):
    """Flip randomly the image.
    """
    def __call__(self, x):
        if np.random.rand() < 0.5:
            x = x[:, :, ::-1]

        return x.copy()

class GaussianNoise(object):
    """Add gaussian noise to the image.
    """
    def __call__(self, x):
        c, h, w = x.shape
        x += np.random.randn(c, h, w) * 0.15
        return x

class ToTensor(object):
    """Transform the image to tensor.
    """
    def __call__(self, x):
        x = torch.from_numpy(x)
        return x

def load_dataset(name=conf.dataset, is_train=True):
    if name == 'cifar':
        dataset = torchvision.datasets.CIFAR10(root='./dataset', train=is_train, download=True, transform=transform_cifar)
    elif name == 'mnist':
        dataset = torchvision.datasets.MNIST(root='./dataset', train=is_train, download=True, transform=transform_mnist)
    dataset = [dataset[i] for i in range(len(dataset))]
    return dataset

def iid(train_set, test_set, num_samples=500):
    train_set = [train_set[i] for i in range(len(train_set))]
    random.shuffle(train_set)
    test_set = [test_set[i] for i in range(len(test_set))]
    random.shuffle(test_set)

    num_samples = int(len(train_set) / conf.num_clients)
    train_sets, test_sets = [], []
    for client_id in range(conf.num_clients):
        train_sets.append(train_set[client_id * num_samples: (client_id + 1) * num_samples])
    for client_id in range(conf.num_clients):
        test_sets.append(test_set)
    return train_sets, test_sets

# Non-overlap
def pathological(train_set, test_set, num_clients=conf.num_clients, num_classes=conf.num_classes, shard_size=500, pick_num=2):
    train_set = [train_set[i] for i in range(len(train_set))]
    train_set.sort(key=lambda i: i[1])
    test_set = [test_set[i] for i in range(len(test_set))]
    test_set.sort(key=lambda i: i[1])

    ends_tr = [0]
    for i in range(1, len(train_set)):
        if train_set[i][1] != train_set[i - 1][1]:
            ends_tr.append(i)
    ends_tr.append(len(train_set))
    ends_te = [0]
    for i in range(1, len(test_set)):
        if test_set[i][1] != test_set[i - 1][1]:
            ends_te.append(i)
    ends_te.append(len(test_set))

    shards = []
    shard_size = int(len(train_set) / (conf.num_clients * pick_num))
    for c in range(conf.num_classes):
        num_samples = ends_tr[c + 1] - ends_tr[c]
        num_shards = int(num_samples / shard_size)
        groups = [train_set[ends_tr[c] + j * shard_size: ends_tr[c] + (j + 1) * shard_size] for j in range(num_shards)]
        if num_shards * shard_size < num_samples:
            groups.append(train_set[ends_tr[c] + shard_size * num_shards: ends_tr[c] + num_samples])
        shards.extend(groups)

    random.shuffle(shards)
    print('NUM_SHARDS:' + str(len(shards)))

    train_sets, test_sets = [], []
    # each client take pick_num shards
    classes_list = [[0 for j in range(num_classes)] for i in range(num_clients)]

    for client_id in range(conf.num_clients):
        train = []
        for p in range(pick_num):
            train.extend(shards[client_id * pick_num + p])
            class_id = shards[client_id * pick_num + p][0][1]
            classes_list[client_id][class_id] += len(shards[client_id * pick_num + p])

        train_sets.append(train)

    rand_c = random.sample(range(num_clients), len(shards) - num_clients * pick_num)
    for s, c in zip(range(num_clients * pick_num, len(shards)), rand_c):
        train_sets[c].extend(shards[s])
        class_id = shards[s][0][1]
        classes_list[c][class_id] += len(shards[s])

    # build test_set
    for client_id in range(conf.num_clients):
        test = []
        num_samples = [ends_te[c + 1] - ends_te[c] for c in range(num_classes)]
        cardinal = min([int(num_samples[i] / classes_list[client_id][i]) if classes_list[client_id][i] != 0 else float('inf') for i in range(num_classes)])
        for c in range(num_classes):
            test.extend(random.sample(test_set[ends_te[c]: ends_te[c + 1]], cardinal * classes_list[client_id][c]))
        test_sets.append(test)

    print('Data distribution:')
    print(classes_list)

    return train_sets, test_sets

def n_way_k_shot(train_set, test_set, num_clients=conf.num_clients, num_classes=conf.num_classes):

    train_set = [train_set[i] for i in range(len(train_set))]
    train_set.sort(key=lambda i: i[1])
    test_set = [test_set[i] for i in range(len(test_set))]
    test_set.sort(key=lambda i: i[1])

    train_sets, test_sets = [], []

    # test samples
    ends_te = [0]
    for i in range(1, len(test_set)):
        if test_set[i][1] != test_set[i - 1][1]:
            ends_te.append(i)
    ends_te.append(len(test_set))
    num_samples = [ends_te[i + 1] - ends_te[i] for i in range(num_classes)]
    num_samples_test = min(num_samples)

    # train set
    ends_tr = [0]
    for i in range(1, len(train_set)):
        if train_set[i][1] != train_set[i - 1][1]:
            ends_tr.append(i)
    ends_tr.append(len(train_set))

    num_samples = [ends_tr[i + 1] - ends_tr[i] for i in range(num_classes)]
    k_list = [random.randint(50, int(min(num_samples) / 2)) for i in range(num_clients)] # k取值随机选取最少50个样本，最大为数量最少的类的样本数的一半
    n_list = [random.randint(2, num_clients) for i in range(num_clients)]

    for client_id in range(num_clients):
        classes = random.sample(range(num_classes), n_list[client_id])
        train, test = [], []
        for c in classes:
            rs = random.sample(train_set[ends_tr[c]: ends_tr[c + 1]], k_list[client_id])
            train.extend(rs)
            rs = random.sample(test_set[ends_te[c]: ends_te[c + 1]], num_samples_test)
            test.extend(rs)

        train_sets.append(train)
        test_sets.append(test)

    return train_sets, test_sets

def load_from_file(client_no):
    x = torch.load('./data/train_data_%d.pt' % client_no)
    y = torch.load('./data/train_label_%d.pt' % client_no).long()
    train_data = [(x[i], y[i]) for i in range(x.shape[0])]

    if os.path.exists('./data/test_data_%d.pt' % client_no):
        x2 = torch.load('./data/test_data_%d.pt' % client_no)
    else:
        x2 = None
    if os.path.exists('./data/test_label_%d.pt' % client_no):
        y2 = torch.load('./data/test_label_%d.pt' % client_no).long()
    else:
        y2 = None

    test_data = [(x2[i], y2[i]) for i in range(x2.shape[0])] if x2 != None and y2 != None else None
    return train_data, test_data


def save_data(train_data, test_data, index):
    x = torch.stack([train_data[i][0] for i in range(len(train_data))], dim=0)
    y = torch.from_numpy(np.array([train_data[i][1] for i in range(len(train_data))]))
    torch.save(x, './data/train_data_%d.pt' % index)
    torch.save(y, './data/train_label_%d.pt' % index)

    x = torch.stack([test_data[i][0] for i in range(len(test_data))], dim=0)
    y = torch.from_numpy(np.array([test_data[i][1] for i in range(len(test_data))]))
    torch.save(x, './data/test_data_%d.pt' % index)
    torch.save(y, './data/test_label_%d.pt' % index)
