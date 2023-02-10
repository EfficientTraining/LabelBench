import os
import argparse
import torch
import numpy as np

num_thread = '5'
os.environ["OMP_NUM_THREADS"] = num_thread
os.environ["OPENBLAS_NUM_THREADS"] = num_thread
os.environ["MKL_NUM_THREADS"] = num_thread

parser = argparse.ArgumentParser()
parser.add_argument("seed", type=int)
parser.add_argument("data", type=str)
parser.add_argument("wandb_name", type=str, default="")
parser.add_argument("model", type=str)
parser.add_argument("--alg", type=str, default="random")
parser.add_argument("--sub_procedure", nargs="*", default=[])
parser.add_argument("--batch_size", type=int, default=1000)
parser.add_argument("--num_batch", type=int, default=10)
args = parser.parse_args()

seed = args.seed
torch.manual_seed(seed)
np.random.seed(seed + 98765)

wandb_name = args.wandb_name
data_name = args.data
model_name = args.model
alg_name = args.alg
batch_size = args.batch_size
num_batch = args.num_batch
sub_procedures = args.sub_procedure

if data_name == "car":
    n_epoch = 15
elif data_name == "voc":
    n_epoch = 15
elif data_name == "coco":
    n_epoch = 10
elif data_name == "celeb":
    n_epoch = 8
elif data_name == "caltech":
    n_epoch = 15
elif data_name == "kuzushiji":
    n_epoch = 10
elif data_name == "cifar10_imb_2":
    n_epoch = 10
elif data_name == "cifar100_imb_2":
    n_epoch = 10
elif data_name == "svhn_imb_2":
    n_epoch = 10
elif data_name == "cifar100_imb_10":
    n_epoch = 10
elif data_name == "cifar10_imb_10":
    n_epoch = 20

if alg_name == "random":
    alg_str = "Random Sampling"
elif alg_name == "random_meta":
    alg_str = "Random Meta, %s" % str(sub_procedures)
elif alg_name == "thompson_pos":
    alg_str = "Thompson w/ #Pos, %s" % str(sub_procedures)
elif alg_name == "thompson_div":
    alg_str = "Thompson w/ Div, %s" % str(sub_procedures)
elif alg_name == "thompson_acc":
    alg_str = "Thompson w/ Acc, %s" % str(sub_procedures)
elif alg_name == "albl_meta":
    alg_str = "ALBL Sampling, %s" % str(sub_procedures)
