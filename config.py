import torch

class Config():
    def __init__(self):
        self.num_classes = 10
        self.num_clients = 10
        self.prototype_size = 512
        self.select_rate = 1  # fraction of clients per communication round
        self.nums_g_epoch = 10000 # number of pretraining round
        self.nums_ft_epoch = 500 # number of fine-tuning round
        self.dataset = 'cifar'
        self.model = 'cnn'
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.batchsize = 64
        self.partition = 'iid'
        self.learning_rate = 0.002
        self.num_proxy_data = 1000 * self.num_classes
        self.l_epoch = 10  # local update
        self.l_iteration_per_epoch = 10
        self.num_l_iteration = 1
        self.prox_cfft = 0.01 # punish term coefficient of fedprox
        self.finetune = False
        self.algorithm = 'MAFSSL'
        self.m_class_fraction = 0.5
        self.labeled_rate = 0.005
        self.aug_times = 2
        self.alpha = 0.75
        self.lambda_u = 75
        self.T = 0.5

    def init_configuration(self, args):
        if args.algorithm:
            self.algorithm = args.algorithm
        if args.num_ft_epoch:
            self.finetune = True
            self.nums_ft_epoch = args.num_ft_epoch
        if args.num_g_epoch:
            self.nums_g_epoch = args.num_g_epoch
        if args.model:
            self.model = args.model


conf = Config()