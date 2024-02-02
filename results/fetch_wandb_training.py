import wandb
import argparse
import pickle
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--wandb_name", type=str, help="Wandb user name.")
    parser.add_argument("--dataset", type=str, help="Dataset name.")
    parser.add_argument("--batch_size", type=int,
                        help="We train neural network after collecting batch_size number of new labels.")
    parser.add_argument("--embed_model", type=str, help="Substring of the embedding model name.")
    parser.add_argument("--classifier_model", type=str, help="Substring of the classifier model name.")
    parser.add_argument("--trainer_config", type=str, help="Substring of the trainer name.")
    parser.add_argument("--corrupter_config", type=str, help="Substring of the corrupter name.")
    args = parser.parse_args()

    wandb_project = "Active Learning, %s, Batch Size=%d" % (args.dataset, args.batch_size)
    api = wandb.Api(timeout=100)
    runs = api.runs("%s/%s" % (args.wandb_name, wandb_project))
    for run in runs:
        config = run.config
        if (args.embed_model in run.config["embed_model_config"]) and \
                (args.classifier_model in run.config["classifier_model_config"]) and \
                (args.trainer_config in run.config["trainer_config"]) and \
                ((("corrupter_config" not in run.config) and ("noiseless" in args.corrupter_config)) or
                 (("corrupter_config" in run.config) and (args.corrupter_config in run.config["corrupter_config"]))):
            run_dict = {}
            label_dict = {}
            print(run.name, config["seed"])
            for row in run.scan_history():
                if "Num Labeled" in row and row["Num Labeled"] is not None:
                    for key in row:
                        if key in run_dict:
                            run_dict[key].append(row[key])
                        else:
                            run_dict[key] = [row[key]]
                if "i" in row and row["i"] is not None:
                    label_dict[row["i"]] = row["Label index"]
            label_lst = []
            for i in range(len(label_dict)):
                try:
                    label_lst.append(label_dict[i])
                except:
                    print("WandB missing index: %d." % i)
                    label_lst.append(None)
            print(len(run_dict["Num Labeled"]), len(label_lst))

            if "noiseless" in args.corrupter_config:
                corrupter_str = ""
            else:
                corrupter_str = "_%s" % args.corrupter_config
            path_name = "./%s%s/bs_%d/embed_%s/model_%s/trainer_%s" % (
                args.dataset, corrupter_str, args.batch_size, args.embed_model, args.classifier_model,
                args.trainer_config)
            if not os.path.exists(path_name):
                os.makedirs(path_name)
            with open("%s/%s_%d.pkl" % (path_name, run.config["strategy_config"].split(".")[0], run.config["seed"]),
                      "wb") as file:
                pickle.dump([run_dict, label_lst], file)
