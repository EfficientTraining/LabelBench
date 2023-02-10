import numpy as np
import subprocess

python_dir = "python"
wandb_name = "jifan"

all_algs = ["galaxy", "confidence", "badge", "mlp", "similar"]
datasets = list(zip(["caltech", "kuzushiji", "cifar10_imb_2", "svhn_imb_2", "cifar100_imb_10", "cifar10_imb_10"],
                    [1000, 1000, 2000, 7000, 2000, 2000, 2000], [20, 10, 20, 10, 20, 20, 20]))[-1:]
algorithms = list(
    zip(["random", "random_meta", "random_meta", "random_meta", "random_meta", "random_meta", "random_meta",
         "albl_meta", "thompson_div", "thompson_acc"],
        [None, ["galaxy"], ["confidence"], ["mlp"], ["similar"], ["badge"], all_algs, all_algs, all_algs, all_algs]))[-1:]

num_processes = 4

# for alg, sub_name in algorithms:
#     for data, batch_size, num_batch in datasets:
#         processes = []
#         for i, seed in list(enumerate(np.linspace(1234, 9999999, num=num_processes, dtype=int))):
#             command = [python_dir, "main.py", str(seed), data, wandb_name, "resnet18", "--alg", alg, "--batch_size",
#                        str(batch_size), "--num_batch", str(num_batch)]
#             if sub_name is not None:
#                 command.append("--sub_procedure")
#                 command = command + sub_name
#             processes.append(subprocess.Popen(command))
#             if i == 1 or i == 3:
#                 for p in processes:
#                     p.wait()
#                 processes = []

for alg, sub_name in algorithms:
    for data, batch_size, num_batch in datasets:
        processes = []
        for seed in np.linspace(1234, 9999999, num=num_processes, dtype=int):
            command = [python_dir, "main.py", str(seed), data, wandb_name, "resnet18", "--alg", alg, "--batch_size",
                       str(batch_size), "--num_batch", str(num_batch)]
            if sub_name is not None:
                command.append("--sub_procedure")
                command = command + sub_name
            processes.append(subprocess.Popen(command))
        for p in processes:
            p.wait()

# for alg, sub_name in algorithms:
#     for data, batch_size, num_batch in datasets:
#         for seed in np.linspace(1234, 9999999, num=num_processes, dtype=int)[2:]:
#             processes = []
#             command = [python_dir, "main.py", str(seed), data, wandb_name, "resnet18", "--alg", alg, "--batch_size",
#                        str(batch_size), "--num_batch", str(num_batch)]
#             if sub_name is not None:
#                 command.append("--sub_procedure")
#                 command = command + sub_name
#             processes.append(subprocess.Popen(command))
#             for p in processes:
#                 p.wait()
