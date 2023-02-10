import numpy as np
import subprocess

python_dir = "python"
wandb_name = "jifan"

project_names = ["caltech_256_classes", "kuzushiji_49_classes", "cifar10_imb_2_2_classes", "svhn_imb_2_2_classes",
                 "cifar100_imb_10_10_classes"]

num_processes = 4

processes = []
for project in project_names:
    for seed in np.linspace(1234, 9999999, num=num_processes, dtype=int)[3:4]:
        command = [python_dir, "tmp_eval.py", wandb_name, project, str(seed)]
        processes.append(subprocess.Popen(command))
for p in processes:
    p.wait()
