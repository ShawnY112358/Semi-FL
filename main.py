import data_loader
from data_loader import pathological, load_from_file, save_data
import os
import config
from run import run_Semi # , run_FedAvg, run_Local, run_MAFSSL
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-a', '--algorithm', help='choose an algorithm')
parser.add_argument('-g', '--num_g_epoch', help='the number of communication rounds', type=int)
parser.add_argument('-f', '--num_ft_epoch', help='the number of fine-tune rounds', type=int)
parser.add_argument('-t', '--run_times', help='run times', type=int, default=1)
parser.add_argument('-m', '--model', help='model')

if not os.path.exists('./data'):
    os.mkdir('./data')
if not os.path.exists('./dataset'):
    os.mkdir('./dataset')
conf = config.conf
print("device:" + str(conf.device))

args = parser.parse_args()
conf.init_configuration(args)

conf_list = {'num_clients': conf.num_clients, 'dataset': conf.dataset, 'model': conf.model, 'algorithm': conf.algorithm}
print(conf_list)

for t in range(args.run_times):
    if conf.algorithm == 'FedAvg':
        run_FedAvg(finetune=conf.finetune)
    elif conf.algorithm == 'Local':
        run_Local()
    elif conf.algorithm == 'Semi':
        run_Semi()
    elif conf.algorithm == 'MAFSSL':
        run_MAFSSL()

    os.rename('./' + conf.algorithm + '/' + 'log', './' + conf.algorithm + '/' + 'log_' + str(t + 1))
    os.rename('./' + conf.algorithm + '/' + 'model', './' + conf.algorithm + '/' + 'model_' + str(t + 1))

print(conf_list)