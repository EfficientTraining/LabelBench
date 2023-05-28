import numpy as np
import subprocess
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--wandb_name", type=str, help="Wandb user name.")
parser.add_argument("--dataset", type=str, help="Dataset name.")
parser.add_argument("--data_dir", type=str, help="Directory to store dataset.", default="./data")
parser.add_argument("--metric", type=str, help="Metric name.")
parser.add_argument("--batch_size", type=int,
                    help="We train neural network after collecting batch_size number of new labels.")
parser.add_argument("--num_batch", type=int, help="Number of batches of collected labels.")
parser.add_argument("--embed_model_config", type=str, help="Path to the encoder model configuration file.",
                    default="none.json")
parser.add_argument("--classifier_model_config", type=str, help="Path to model configuration file.")
parser.add_argument("--trainer_config", type=str, help="Path to trainer configuration file.")
parser.add_argument("--strategies", type=str, nargs="+", help="Path to AL strategy configuration files.")
parser.add_argument("--num_runs", type=int, default=4, help="Number of runs.")
parser.add_argument("--run_per_device", type=int, default=1, help="Number of runs on each GPU.")
parser.add_argument("--device_per_run", type=int, default=0, help="Number of GPUs for each run.")
parser.add_argument("--gpu_masks", type=int, nargs="+", help="Which devices to run on.")
parser.add_argument("--skip", type=int, default=0, help="Skip the specified number of experiments in the beginning.")
args = parser.parse_args()

python_dir = "python3"

counter = 0
processes = []
for strategy in args.strategies:
    for i, seed in enumerate(np.linspace(1234, 9999999, num=args.num_runs, dtype=int)):
        command = [python_dir, "main.py",
                   "--seed", str(seed),
                   "--wandb_name", args.wandb_name,
                   "--dataset", args.dataset,
                   "--data_dir", args.data_dir,
                   "--metric", args.metric,
                   "--batch_size", str(args.batch_size),
                   "--num_batch", str(args.num_batch),
                   "--embed_model_config", args.embed_model_config,
                   "--classifier_model_config", args.classifier_model_config,
                   "--strategy_config", strategy,
                   "--trainer_config", args.trainer_config
                   ]
        if counter >= args.skip:
            new_env = os.environ.copy()
            if args.device_per_run > 0:
                num_parallel_runs = len(args.gpu_masks) // args.device_per_run
                assert num_parallel_runs > 0
                new_env["CUDA_VISIBLE_DEVICES"] = \
                    "".join([str(counter % num_parallel_runs * args.device_per_run + i) + "," for i in
                             range(args.device_per_run)])[:-1]
                processes.append(subprocess.Popen(command, env=new_env))
                if len(processes) == num_parallel_runs:
                    for p in processes:
                        p.wait()
                    processes = []
            else:
                new_env["CUDA_VISIBLE_DEVICES"] = str(
                    args.gpu_masks[(counter % (len(args.gpu_masks) * args.run_per_device)) // args.run_per_device])
                processes.append(subprocess.Popen(command, env=new_env))
                if len(processes) == len(args.gpu_masks) * args.run_per_device:
                    for p in processes:
                        p.wait()
                    processes = []
        counter += 1

for p in processes:
    p.wait()
